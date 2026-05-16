"""
ws_bridge.py  —  WebSocket bridge for SignBridge UI
====================================================
Wraps asl_sentence_builder.py detection logic and streams
JSON events to the web UI at ws://localhost:8765

Install:  pip install opencv-python mediapipe websockets
Run:      python ws_bridge.py
Then open index.html in your browser.

JSON messages sent to UI:
  {"type":"sign",  "sign":"A", "confidence":0.92}
  {"type":"progress", "sign":"A", "progress":0.65}
  {"type":"typed", "sign":"A", "action":"type"}
  {"type":"typed", "sign":"", "action":"backspace"}
  {"type":"typed", "sign":"", "action":"clear"}
  {"type":"sentence", "sentence":"HELLO WORLD"}
"""

import asyncio, json, math, time, urllib.request, os
from collections import deque, Counter
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import websockets

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

# ── Indices ──────────────────────────────────────────────────────────────────
WRIST=0;THUMB_TIP=4;THUMB_IP=3;THUMB_MCP=2;THUMB_CMC=1
INDEX_TIP=8;INDEX_DIP=7;INDEX_PIP=6;INDEX_MCP=5
MIDDLE_TIP=12;MIDDLE_DIP=11;MIDDLE_PIP=10;MIDDLE_MCP=9
RING_TIP=16;RING_DIP=15;RING_PIP=14;RING_MCP=13
PINKY_TIP=20;PINKY_DIP=19;PINKY_PIP=18;PINKY_MCP=17

def d(lm,a,b):
    return math.hypot(lm[a].x-lm[b].x, lm[a].y-lm[b].y)

def tip_up(lm,tip,pip):   return lm[tip].y < lm[pip].y
def curl(lm,tip,pip,mcp): return lm[tip].y > lm[mcp].y

def finger_states(lm, side):
    if side=="Right": thumb = lm[THUMB_TIP].x < lm[THUMB_MCP].x
    else:             thumb = lm[THUMB_TIP].x > lm[THUMB_MCP].x
    i = tip_up(lm,INDEX_TIP,INDEX_PIP)
    m = tip_up(lm,MIDDLE_TIP,MIDDLE_PIP)
    r = tip_up(lm,RING_TIP,RING_PIP)
    p = tip_up(lm,PINKY_TIP,PINKY_PIP)
    return thumb,i,m,r,p

def all_curled(lm):
    return (curl(lm,INDEX_TIP,INDEX_PIP,INDEX_MCP) and
            curl(lm,MIDDLE_TIP,MIDDLE_PIP,MIDDLE_MCP) and
            curl(lm,RING_TIP,RING_PIP,RING_MCP) and
            curl(lm,PINKY_TIP,PINKY_PIP,PINKY_MCP))

def classify(lm, side):
    t,i,m,r,p = finger_states(lm, side)
    if t and i and m and r and p:           return "HELLO"
    if not t and i and m and r and p:       return "HI"
    if t and i and m and r and not p:       return "THANK YOU"
    if t and i and not m and not r and p:   return "I LOVE YOU"
    if not t and not i and not m and not r and not p:
        if lm[THUMB_TIP].y < lm[WRIST].y:  return "SORRY"
        return "FIST"
    if not t and i and m and not r and not p:
        if d(lm,INDEX_TIP,THUMB_TIP)<0.08: return "NO"
        return "PEACE"
    if t and not i and not m and not r and not p:
        if lm[THUMB_TIP].y > lm[WRIST].y:  return "THUMBS DOWN"
        return "THUMBS UP"
    if not t and i and m and r and not p:   return "STOP"
    if (d(lm,INDEX_TIP,THUMB_TIP)<0.06 and
        d(lm,MIDDLE_TIP,THUMB_TIP)<0.08):   return "MORE"
    if d(lm,INDEX_TIP,THUMB_TIP)<0.05 and not m and not r and not p:
        return "EAT"
    if t and not i and not m and not r and p: return "CALL ME"
    if not t and i and not m and not r and p: return "ROCK ON"
    if not t and i and not m and not r and not p: return "1"
    if t and i and m and not r and not p:   return "3"
    if not t and i and m and r and p:       return "4"
    if not t and not i and not m and not r and not p:
        return "M/N/S"
    if (not i and not m and not r and not p and
        lm[THUMB_TIP].x > lm[INDEX_MCP].x): return "A"
    avg = (d(lm,INDEX_TIP,INDEX_MCP)+d(lm,MIDDLE_TIP,MIDDLE_MCP))/2
    if 0.10<avg<0.18 and not all_curled(lm): return "C"
    if i and not m and not r and not p and d(lm,MIDDLE_TIP,THUMB_TIP)<0.08: return "D"
    if d(lm,INDEX_TIP,THUMB_TIP)<0.06 and m and r and p: return "F"
    if i and not m and not r and not p and abs(lm[INDEX_TIP].y-lm[INDEX_MCP].y)<0.04: return "G"
    if i and m and not r and not p and abs(lm[INDEX_TIP].y-lm[WRIST].y)>abs(lm[MIDDLE_TIP].y-lm[WRIST].y)*0.8: return "H"
    if not t and not i and not m and not r and p: return "I"
    if t and i and m and not r and not p and d(lm,THUMB_TIP,MIDDLE_PIP)<0.08: return "K"
    if t and i and not m and not r and not p: return "L"
    if d(lm,INDEX_TIP,THUMB_TIP)<0.06 and d(lm,MIDDLE_TIP,THUMB_TIP)<0.07 and d(lm,RING_TIP,THUMB_TIP)<0.08: return "O"
    if i and m and not r and not p and d(lm,INDEX_TIP,MIDDLE_TIP)<0.04: return "R"
    if not i and not m and not r and not p and d(lm,THUMB_TIP,INDEX_PIP)<0.06: return "T"
    if i and m and not r and not p and d(lm,INDEX_TIP,MIDDLE_TIP)<0.04: return "U"
    if not t and i and m and r and not p:   return "W"
    if t and not i and not m and not r and p: return "Y"
    return "..."

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
               (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

def draw_hand(frame, landmarks, w, h, color):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    for a,b in CONNECTIONS:
        cv2.line(frame,pts[a],pts[b],color,2)
    for x,y in pts:
        cv2.circle(frame,(x,y),5,(255,255,255),-1)
        cv2.circle(frame,(x,y),5,color,2)

history = {0: deque(maxlen=10)}
def smooth(idx, label):
    history[idx].append(label)
    return Counter(history[idx]).most_common(1)[0][0]

# ── State ────────────────────────────────────────────────────────────────────
sentence   = ""
last_sign  = ""
sign_start = 0.0
HOLD_TIME  = 1.5
CLEAR_TIME = 2.0
clients    = set()

async def broadcast(msg: dict):
    if clients:
        data = json.dumps(msg)
        await asyncio.gather(*[c.send(data) for c in clients], return_exceptions=True)

async def ws_handler(ws):
    clients.add(ws)
    try:
        await ws.send(json.dumps({"type":"sentence","sentence":sentence}))
        await ws.wait_closed()
    finally:
        clients.discard(ws)

# ── Detector ─────────────────────────────────────────────────────────────────
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.55,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

async def run_detection(loop):
    global sentence, last_sign, sign_start

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("SignBridge backend running — ws://localhost:8765")
    print("Open index.html in your browser | Q = quit")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        now   = time.time()

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        current_sign = "..."
        progress     = 0.0
        confidence   = 0.0

        if result.hand_landmarks:
            lm   = result.hand_landmarks[0]
            side = result.handedness[0][0].category_name
            score = result.handedness[0][0].score
            color = (0,220,120) if side=="Right" else (80,160,255)

            draw_hand(frame, lm, W, H, color)
            raw = classify(lm, side)
            current_sign = smooth(0, raw)
            confidence   = float(score)

            # Bounding box
            xs = [int(p.x*W) for p in lm]; ys = [int(p.y*H) for p in lm]
            x1,y1 = max(min(xs)-20,0), max(min(ys)-30,0)
            x2,y2 = min(max(xs)+20,W), min(max(ys)+20,H)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            lbl = f"{side}: {current_sign}"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.85,2)
            cv2.rectangle(frame,(x1,y1-th-14),(x1+tw+10,y1),color,-1)
            cv2.putText(frame,lbl,(x1+5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,0),2)

            # Progress
            if current_sign == last_sign and current_sign not in ("...",):
                elapsed  = now - sign_start
                hold     = CLEAR_TIME if current_sign=="FIST" else HOLD_TIME
                progress = min(elapsed/hold, 1.0)
                if progress >= 1.0:
                    if current_sign == "FIST":
                        sentence = ""
                        asyncio.run_coroutine_threadsafe(
                            broadcast({"type":"typed","sign":"","action":"clear","sentence":sentence}), loop)
                    elif current_sign == "THUMBS DOWN":
                        sentence = sentence[:-1]
                        asyncio.run_coroutine_threadsafe(
                            broadcast({"type":"typed","sign":"","action":"backspace","sentence":sentence}), loop)
                    elif current_sign not in ("...",):
                        s = current_sign
                        if len(s)==1:
                            sentence += s
                        else:
                            if sentence and sentence[-1]!=" ": sentence += " "
                            sentence += s + " "
                        asyncio.run_coroutine_threadsafe(
                            broadcast({"type":"typed","sign":s,"action":"type","sentence":sentence}), loop)
                    last_sign  = ""
                    sign_start = now
            else:
                last_sign  = current_sign
                sign_start = now

            # Progress bar on frame
            if progress > 0:
                bw = int((x2-x1)*progress)
                cv2.rectangle(frame,(x1,y2),(x1+bw,y2+8),(0,255,200),-1)
                cv2.rectangle(frame,(x1,y2),(x2,y2+8),(200,200,200),1)

        # Broadcast sign state
        asyncio.run_coroutine_threadsafe(
            broadcast({"type":"sign","sign":current_sign,
                       "confidence":round(confidence,3),
                       "progress":round(progress,3)}), loop)

        # Sentence overlay
        panel_h = 90
        overlay = frame[H-panel_h:H].copy()
        cv2.rectangle(overlay,(0,0),(W,panel_h),(15,18,28),-1)
        cv2.addWeighted(overlay,0.8,frame[H-panel_h:H],0.2,0,frame[H-panel_h:H])
        cv2.putText(frame,sentence if sentence else "_",
                    (12,H-panel_h+54),cv2.FONT_HERSHEY_SIMPLEX,0.95,(0,230,155),2)
        cv2.putText(frame,"SignBridge  |  Hold 1.5s to type  |  Thumbs Down = Backspace  |  Fist 2s = Clear  |  Q = Quit",
                    (10,26),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

        cv2.imshow("SignBridge — ASL Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

async def main():
    loop = asyncio.get_event_loop()
    server = await websockets.serve(ws_handler, "localhost", 8765)
    print("WebSocket server started at ws://localhost:8765")
    await asyncio.gather(server.wait_closed(), run_detection(loop))

if __name__ == "__main__":
    asyncio.run(main())