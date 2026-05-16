"""
ASL Sentence Builder — mediapipe 0.10.13+
==========================================
- Detects ASL letters (A-Y), numbers (1-5), and common signs
- Holds a sign for 1.5s to "type" it into a sentence
- SPACE sign (flat hand) adds a space
- BACKSPACE (thumbs down) deletes last character
- CLEAR (fist held 2s) clears the sentence
- Press Q to quit
- Streams results to web UI at ws://localhost:8765

Requirements:  pip install opencv-python mediapipe websockets
Run:           python asl_sentence_builder.py
"""

import cv2, math, urllib.request, os, time, threading, json
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── WebSocket server (runs in background thread) ─────────────────────────────
try:
    import asyncio
    import websockets

    ws_clients  = set()
    ws_loop     = None
    ws_queue    = []

    async def _ws_handler(ws):
        ws_clients.add(ws)
        # send current sentence on connect
        await ws.send(json.dumps({"type": "sentence", "sentence": sentence}))
        try:
            await ws.wait_closed()
        finally:
            ws_clients.discard(ws)

    async def _ws_server():
        async with websockets.serve(_ws_handler, "localhost", 8765):
            print("WebSocket ready → ws://localhost:8765  (open index.html)")
            await asyncio.Future()          # run forever

    def _start_ws():
        global ws_loop
        ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(ws_loop)
        ws_loop.run_until_complete(_ws_server())

    threading.Thread(target=_start_ws, daemon=True).start()

    def ws_broadcast(msg: dict):
        if ws_loop and ws_clients:
            data = json.dumps(msg)
            for client in list(ws_clients):
                asyncio.run_coroutine_threadsafe(client.send(data), ws_loop)

    WS_ENABLED = True

except ImportError:
    print("websockets not installed — running without web UI bridge.")
    print("Install with:  pip install websockets")
    def ws_broadcast(msg): pass
    WS_ENABLED = False

# ── Model download ───────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task (~5 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

# ── Landmark indices ─────────────────────────────────────────────────────────
WRIST=0; THUMB_TIP=4; THUMB_IP=3; THUMB_MCP=2; THUMB_CMC=1
INDEX_TIP=8;  INDEX_DIP=7;  INDEX_PIP=6;  INDEX_MCP=5
MIDDLE_TIP=12;MIDDLE_DIP=11;MIDDLE_PIP=10;MIDDLE_MCP=9
RING_TIP=16;  RING_DIP=15;  RING_PIP=14;  RING_MCP=13
PINKY_TIP=20; PINKY_DIP=19; PINKY_PIP=18; PINKY_MCP=17

# ── Geometry helpers ─────────────────────────────────────────────────────────
def d(lm,a,b):
    return math.hypot(lm[a].x-lm[b].x, lm[a].y-lm[b].y)

def tip_up(lm,tip,pip):   return lm[tip].y < lm[pip].y
def tip_down(lm,tip,pip): return lm[tip].y > lm[pip].y

def curl(lm, tip, pip, mcp):
    """True if finger is curled (tip below mcp level roughly)."""
    return lm[tip].y > lm[mcp].y

def finger_states(lm, side):
    if side=="Right": thumb = lm[THUMB_TIP].x < lm[THUMB_MCP].x
    else:             thumb = lm[THUMB_TIP].x > lm[THUMB_MCP].x
    i = tip_up(lm, INDEX_TIP,  INDEX_PIP)
    m = tip_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    r = tip_up(lm, RING_TIP,   RING_PIP)
    p = tip_up(lm, PINKY_TIP,  PINKY_PIP)
    return thumb, i, m, r, p

def all_curled(lm):
    return (curl(lm,INDEX_TIP,INDEX_PIP,INDEX_MCP) and
            curl(lm,MIDDLE_TIP,MIDDLE_PIP,MIDDLE_MCP) and
            curl(lm,RING_TIP,RING_PIP,RING_MCP) and
            curl(lm,PINKY_TIP,PINKY_PIP,PINKY_MCP))

# ── Master classifier ────────────────────────────────────────────────────────
def classify(lm, side):
    t, i, m, r, p = finger_states(lm, side)

    # ── Common words / phrases ───────────────────────────────────────────────
    if t and i and m and r and p:           return "HELLO"
    if not t and i and m and r and p:       return "HI"

    if t and i and m and r and not p:       return "THANK YOU"

    if t and i and not m and not r and p:   return "I LOVE YOU"

    if not t and not i and not m and not r and not p:
        if lm[THUMB_TIP].y < lm[WRIST].y:  return "SORRY"
        return "FIST"

    if not t and i and m and not r and not p:
        if d(lm,INDEX_TIP,THUMB_TIP)<0.08: return "NO"
        return "PEACE / V"

    if t and not i and not m and not r and not p:
        if lm[THUMB_TIP].y > lm[WRIST].y:  return "THUMBS DOWN"
        return "THUMBS UP / YES"

    if not t and i and m and r and not p:   return "STOP"

    if (d(lm,INDEX_TIP,THUMB_TIP)<0.06 and
        d(lm,MIDDLE_TIP,THUMB_TIP)<0.08):  return "MORE"

    if d(lm,INDEX_TIP,THUMB_TIP)<0.05 and not m and not r and not p:
        return "EAT / FOOD"

    if not t and i and m and r and not p:
        spread = d(lm,INDEX_TIP,RING_TIP)
        if spread > 0.15:                   return "WATER / W"

    if not t and i and not m and not r and not p: return "WHERE / POINT"

    if not t and not i and m and r and not p:     return "WHY"

    if d(lm,THUMB_TIP,INDEX_TIP)<0.07 and m and r and p: return "OK"

    if t and not i and not m and not r and p:     return "CALL ME"

    if not t and i and not m and not r and p:     return "ROCK ON"

    # ── ASL Numbers ──────────────────────────────────────────────────────────
    if not t and i and not m and not r and not p: return "1"

    if t and i and m and not r and not p:         return "3"

    if not t and i and m and r and p:             return "4"

    # ── ASL Letters (static) ─────────────────────────────────────────────────
    if (not i and not m and not r and not p and
        lm[THUMB_TIP].x > lm[INDEX_MCP].x):      return "A"

    if not t and i and m and r and p:             return "B"

    avg_curl = (d(lm,INDEX_TIP,INDEX_MCP) + d(lm,MIDDLE_TIP,MIDDLE_MCP)) / 2
    if 0.10 < avg_curl < 0.18 and not all_curled(lm): return "C"

    if (i and not m and not r and not p and
        d(lm,MIDDLE_TIP,THUMB_TIP)<0.08):         return "D"

    if (d(lm,INDEX_TIP,THUMB_TIP)<0.06 and
        m and r and p):                           return "F"

    if (i and not m and not r and not p and
        abs(lm[INDEX_TIP].y - lm[INDEX_MCP].y)<0.04): return "G"

    if (i and m and not r and not p and
        abs(lm[INDEX_TIP].y-lm[WRIST].y) >
        abs(lm[MIDDLE_TIP].y-lm[WRIST].y)*0.8):  return "H"

    if not t and not i and not m and not r and p: return "I"

    if (t and i and m and not r and not p and
        d(lm,THUMB_TIP,MIDDLE_PIP)<0.08):         return "K"

    if t and i and not m and not r and not p:      return "L"

    if (not t and not i and not m and not r and not p): return "M/N/S"

    if (d(lm,INDEX_TIP,THUMB_TIP)<0.06 and
        d(lm,MIDDLE_TIP,THUMB_TIP)<0.07 and
        d(lm,RING_TIP,THUMB_TIP)<0.08):           return "O"

    if (i and m and not r and not p and
        d(lm,INDEX_TIP,MIDDLE_TIP)<0.04):         return "R"

    if (not i and not m and not r and not p and
        d(lm,THUMB_TIP,INDEX_PIP)<0.06):          return "T"

    if (i and m and not r and not p and
        d(lm,INDEX_TIP,MIDDLE_TIP)<0.04):         return "U"

    if not t and i and m and r and not p:         return "W"

    if (not t and not m and not r and not p and
        lm[INDEX_TIP].y > lm[INDEX_PIP].y and
        lm[INDEX_TIP].y < lm[INDEX_MCP].y):       return "X"

    if t and not i and not m and not r and p:      return "Y"

    return "..."

# ── Draw skeleton ────────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
def draw_hand(frame, landmarks, w, h, color):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    for a,b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x,y in pts:
        cv2.circle(frame,(x,y),4,(255,255,255),-1)
        cv2.circle(frame,(x,y),4,color,1)

# ── Smoothing ────────────────────────────────────────────────────────────────
history = {0: deque(maxlen=10), 1: deque(maxlen=10)}
def smooth(idx, label):
    history[idx].append(label)
    return Counter(history[idx]).most_common(1)[0][0]

# ── Sentence state ───────────────────────────────────────────────────────────
sentence      = ""
last_sign     = ""
sign_start    = 0.0
HOLD_TIME     = 1.5   # seconds to hold a sign before it's typed
CLEAR_TIME    = 2.0   # hold FIST for 2s to clear
typed_flash   = ""
flash_until   = 0.0

def add_to_sentence(sign):
    global sentence, typed_flash, flash_until
    if sign == "THUMBS DOWN":
        sentence = sentence[:-1]
        typed_flash = "BACKSPACE"
        ws_broadcast({"type": "typed", "sign": "", "action": "backspace", "sentence": sentence})
    elif sign in ("HI","HELLO") and len(sentence)>0 and sentence[-1]==" ":
        sentence += sign + " "
        typed_flash = f"+ '{sign}'"
        ws_broadcast({"type": "typed", "sign": sign, "action": "type", "sentence": sentence})
    elif sign == "FIST":
        pass  # handled separately for clear
    elif sign == "...":
        pass
    else:
        if len(sign)==1:
            sentence += sign
        else:
            if sentence and sentence[-1] != " ":
                sentence += " "
            sentence += sign + " "
        typed_flash = f"+ '{sign}'"
        ws_broadcast({"type": "typed", "sign": sign, "action": "type", "sentence": sentence})
    flash_until = time.time() + 1.2

# ── Detector setup ───────────────────────────────────────────────────────────
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.55,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("ASL Sentence Builder — hold a sign for 1.5s to type it | Q = quit")

while True:
    ok, frame = cap.read()
    if not ok: break

    frame    = cv2.flip(frame, 1)
    H, W     = frame.shape[:2]
    now      = time.time()

    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result   = detector.detect(mp_img)

    current_sign = "..."
    progress     = 0.0

    if result.hand_landmarks:
        lm         = result.hand_landmarks[0]
        side       = result.handedness[0][0].category_name
        confidence = result.handedness[0][0].score
        color      = (0,220,120) if side=="Right" else (80,160,255)

        draw_hand(frame, lm, W, H, color)
        raw_sign     = classify(lm, side)
        current_sign = smooth(0, raw_sign)

        # Progress bar logic
        if current_sign == last_sign and current_sign not in ("...",):
            elapsed  = now - sign_start
            hold     = CLEAR_TIME if current_sign=="FIST" else HOLD_TIME
            progress = min(elapsed / hold, 1.0)

            if progress >= 1.0:
                if current_sign == "FIST":
                    sentence = ""
                    typed_flash = "CLEARED"
                    flash_until = now + 1.2
                    ws_broadcast({"type": "typed", "sign": "", "action": "clear", "sentence": ""})
                else:
                    add_to_sentence(current_sign)
                last_sign  = ""   # reset so it doesn't re-fire
                sign_start = now
        else:
            last_sign  = current_sign
            sign_start = now

        # Broadcast current sign to web UI
        ws_broadcast({
            "type":       "sign",
            "sign":       current_sign,
            "confidence": round(float(confidence), 3),
            "progress":   round(progress, 3),
            "hand":       side,
        })

        # Bounding box + label
        xs = [int(p.x*W) for p in lm]
        ys = [int(p.y*H) for p in lm]
        x1,y1 = max(min(xs)-20,0), max(min(ys)-20,0)
        x2,y2 = min(max(xs)+20,W), min(max(ys)+20,H)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        label_txt = f"{side}: {current_sign}"
        (tw,th),_ = cv2.getTextSize(label_txt,cv2.FONT_HERSHEY_SIMPLEX,0.85,2)
        cv2.rectangle(frame,(x1,y1-th-14),(x1+tw+10,y1),color,-1)
        cv2.putText(frame,label_txt,(x1+5,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,0),2)

        # Progress bar
        if progress > 0:
            bar_w = int((x2-x1)*progress)
            cv2.rectangle(frame,(x1,y2),(x1+bar_w,y2+8),(0,255,255),-1)
            cv2.rectangle(frame,(x1,y2),(x2,y2+8),(255,255,255),1)

    else:
        # No hand — broadcast idle
        ws_broadcast({"type": "sign", "sign": "...", "confidence": 0, "progress": 0, "hand": ""})

    # ── Sentence panel at bottom ─────────────────────────────────────────────
    panel_h = 110
    panel   = frame[H-panel_h:H, 0:W]
    overlay = panel.copy()
    cv2.rectangle(overlay,(0,0),(W,panel_h),(20,20,20),-1)
    cv2.addWeighted(overlay,0.75,panel,0.25,0,panel)
    frame[H-panel_h:H,0:W] = panel

    cv2.putText(frame,"SENTENCE:",(12,H-panel_h+28),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(180,180,180),1)

    # Word-wrap sentence display
    display = sentence if sentence else "_"
    words   = display.split(" ")
    line, lines = "", []
    for w_word in words:
        test = (line+" "+w_word).strip()
        (tw2,_),_ = cv2.getTextSize(test,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)
        if tw2 > W-30 and line:
            lines.append(line); line = w_word
        else:
            line = test
    if line: lines.append(line)
    lines = lines[-2:]  # show last 2 lines max
    for li, txt in enumerate(lines):
        cv2.putText(frame, txt, (12, H-panel_h+58+li*34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,180), 2)

    # Flash feedback
    if now < flash_until:
        cv2.putText(frame, typed_flash, (W-280, H-panel_h+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,230,255), 2)

    # ── HUD ──────────────────────────────────────────────────────────────────
    cv2.putText(frame,"ASL Sentence Builder  |  Hold 1.5s to type  |  Fist 2s = Clear  |  Q = Quit",
                (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

    cv2.imshow("ASL Sentence Builder", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release()
cv2.destroyAllWindows()
detector.close()