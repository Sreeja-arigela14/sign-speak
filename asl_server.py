import asyncio, base64, json, math, os, time, urllib.request
from collections import deque, Counter
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import websockets
import threading

# ── Model download ────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task (~5 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST=0;THUMB_TIP=4;THUMB_IP=3;THUMB_MCP=2;THUMB_CMC=1
INDEX_TIP=8;INDEX_DIP=7;INDEX_PIP=6;INDEX_MCP=5
MIDDLE_TIP=12;MIDDLE_DIP=11;MIDDLE_PIP=10;MIDDLE_MCP=9
RING_TIP=16;RING_DIP=15;RING_PIP=14;RING_MCP=13
PINKY_TIP=20;PINKY_DIP=19;PINKY_PIP=18;PINKY_MCP=17

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def d(lm,a,b):
    return math.hypot(lm[a].x-lm[b].x, lm[a].y-lm[b].y)

def tip_up(lm,tip,pip): return lm[tip].y < lm[pip].y
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
    t,i,m,r,p = finger_states(lm,side)

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
        return "YES"

    if not t and i and m and r and not p:   return "STOP"

    if d(lm,INDEX_TIP,THUMB_TIP)<0.06 and d(lm,MIDDLE_TIP,THUMB_TIP)<0.08: return "MORE"
    if d(lm,INDEX_TIP,THUMB_TIP)<0.05 and not m and not r and not p: return "EAT"
    if d(lm,THUMB_TIP,INDEX_TIP)<0.07 and m and r and p: return "OK"

    if t and not i and not m and not r and p:   return "CALL ME"
    if not t and i and not m and not r and p:   return "ROCK ON"
    if not t and not i and m and r and not p:   return "WHY"

    if not t and i and m and r and p and d(lm,INDEX_TIP,PINKY_TIP)<0.12: return "HELP"

    avg_all = (d(lm,INDEX_TIP,INDEX_MCP)+d(lm,MIDDLE_TIP,MIDDLE_MCP)+
               d(lm,RING_TIP,RING_MCP)+d(lm,PINKY_TIP,PINKY_MCP))/4
    if 0.07 < avg_all < 0.13 and not t: return "WANT"

    if not t and i and not m and not r and not p and lm[INDEX_TIP].y > lm[INDEX_PIP].y: return "NEED"

    if t and i and m and r and p and d(lm,INDEX_TIP,PINKY_TIP)<0.10: return "PLEASE"

    if t and i and m and r and not p and lm[THUMB_TIP].y < lm[INDEX_MCP].y: return "GOOD"

    if t and not i and not m and not r and not p and lm[THUMB_TIP].y > lm[WRIST].y+0.05: return "BAD"

    if not t and i and m and r and p and d(lm,INDEX_TIP,PINKY_TIP)>0.18: return "WAIT"

    if not t and i and not m and not r and not p and lm[INDEX_TIP].x < lm[INDEX_MCP].x: return "COME"

    if not t and i and not m and not r and not p and lm[INDEX_TIP].x > lm[INDEX_MCP].x: return "GO"

    if t and i and m and r and p and lm[WRIST].y > lm[MIDDLE_MCP].y+0.1: return "HAPPY"

    if not t and not i and m and r and p: return "SAD"

    if not t and i and m and r and p and avg_all < 0.09: return "TIRED"

    avg_curl_val = (d(lm,INDEX_TIP,INDEX_MCP)+d(lm,MIDDLE_TIP,MIDDLE_MCP))/2
    if t and 0.08<avg_curl_val<0.15 and not p: return "HOT"

    if not t and not i and not m and not r and not p and d(lm,THUMB_TIP,INDEX_MCP)<0.08: return "COLD"

    if not t and i and not m and not r and not p and lm[INDEX_TIP].x < lm[WRIST].x: return "PAIN"

    if not t and i and not m and not r and not p: return "1"
    if t and i and m and not r and not p:         return "3"
    if not t and i and m and r and p:             return "4"

    if not i and not m and not r and not p and lm[THUMB_TIP].x > lm[INDEX_MCP].x: return "A"
    avg_curl2=(d(lm,INDEX_TIP,INDEX_MCP)+d(lm,MIDDLE_TIP,MIDDLE_MCP))/2
    if 0.10<avg_curl2<0.18 and not all_curled(lm): return "C"
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
    if not t and i and m and r and not p: return "W"
    if t and not i and not m and not r and p: return "Y"

    return "..."

def draw_skeleton(frame, landmarks, W, H, color=(0,220,120)):
    pts = [(int(lm.x*W), int(lm.y*H)) for lm in landmarks]
    for a,b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x,y in pts:
        cv2.circle(frame,(x,y),5,(255,255,255),-1)
        cv2.circle(frame,(x,y),5,color,2)
    return pts

def frame_to_b64(frame, quality=60):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()

# ── Smoothing ─────────────────────────────────────────────────────────────────
history = deque(maxlen=10)
def smooth(label):
    history.append(label)
    return Counter(history).most_common(1)[0][0]

# ── Detector ──────────────────────────────────────────────────────────────────
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.55,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ── Claude AI sentence expansion ──────────────────────────────────────────────
def expand_signs_with_claude(signs: list) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic()
        signs_str = ", ".join(signs)
        print(f"[Claude] Calling API for signs: {signs_str}")
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"These ASL signs were detected: {signs_str}. "
                    "Convert them into a single natural, fluent English sentence or two. "
                    "Keep it simple and conversational. "
                    "Only return the sentence, nothing else."
                )
            }]
        )
        result = response.content[0].text.strip()
        print(f"[Claude] Response: {result}")
        return result
    except ImportError:
        print("[Claude] anthropic package not installed, using fallback")
        return expand_signs_fallback(signs)
    except Exception as e:
        print(f"[Claude] API error: {e}, using fallback")
        return expand_signs_fallback(signs)

def expand_signs_fallback(signs):
    expansions = {
        "HELLO": "Hello", "HI": "Hi", "THANK YOU": "Thank you",
        "I LOVE YOU": "I love you", "SORRY": "I am sorry",
        "YES": "Yes", "NO": "No", "STOP": "Please stop",
        "MORE": "I want more", "WATER": "I need water",
        "EAT": "I want to eat", "OK": "Okay", "WHY": "Why",
        "CALL ME": "Call me", "ROCK ON": "Rock on",
        "HELP": "Please help me", "NEED": "I need",
        "WANT": "I want", "PLEASE": "Please",
        "GOOD": "Good", "BAD": "Bad",
        "HOT": "It is hot", "COLD": "It is cold",
        "TIRED": "I am tired", "HAPPY": "I am happy",
        "SAD": "I am sad", "PAIN": "I am in pain",
        "COME": "Come here", "GO": "I need to go",
        "WAIT": "Please wait",
    }
    parts = [expansions.get(s, s) for s in signs]
    return " ".join(parts).capitalize() + "."

# ── Process a single frame ────────────────────────────────────────────────────
def process_frame(frame):
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    sign = "..."
    confidence = 0.0

    if result.hand_landmarks:
        lm   = result.hand_landmarks[0]
        side = result.handedness[0][0].category_name
        conf = result.handedness[0][0].score
        color = (0,220,120) if side=="Right" else (80,160,255)

        draw_skeleton(frame, lm, W, H, color)

        raw  = classify(lm, side)
        sign = smooth(raw)
        confidence = float(conf)

        xs = [int(p.x*W) for p in lm]; ys = [int(p.y*H) for p in lm]
        x1,y1 = max(min(xs)-20,0), max(min(ys)-30,0)
        x2,y2 = min(max(xs)+20,W), min(max(ys)+20,H)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        lbl = f"{side}: {sign}"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        cv2.rectangle(frame,(x1,y1-th-12),(x1+tw+8,y1),color,-1)
        cv2.putText(frame,lbl,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    return frame, sign, confidence

# ── WebSocket server ──────────────────────────────────────────────────────────
clients = set()

async def ws_handler(ws):
    clients.add(ws)
    print(f"Browser connected. ({len(clients)} clients)")
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)

                # ── FIX: Browser can override sentence state (for Back/Clear buttons) ──
                if msg.get("type") == "set_sentence":
                    new_sentence = msg.get("sentence", "")
                    # Update the shared sentence in camera_loop via the global
                    global _sentence_override, _sentence_override_value
                    _sentence_override = True
                    _sentence_override_value = new_sentence
                    print(f"[SetSentence] Browser set sentence to: '{new_sentence}'")

                elif msg.get("type") == "combine_signs":
                    signs = msg.get("signs", [])
                    print(f"[Combine] Signs: {signs}")
                    loop = asyncio.get_event_loop()
                    try:
                        sentence = await asyncio.wait_for(
                            loop.run_in_executor(None, expand_signs_with_claude, signs),
                            timeout=15.0
                        )
                    except asyncio.TimeoutError:
                        print("[Combine] Claude timed out, using fallback")
                        sentence = expand_signs_fallback(signs)
                    await ws.send(json.dumps({"type": "combine_result", "sentence": sentence}))

            except Exception as e:
                print(f"[Handler] Error: {e}")
                await ws.send(json.dumps({"type":"error","message":str(e)}))
    finally:
        clients.discard(ws)

async def broadcast(msg: dict):
    if clients:
        data = json.dumps(msg)
        await asyncio.gather(*[c.send(data) for c in clients], return_exceptions=True)

# ── Sentence override globals (set by browser Back/Clear, consumed by camera_loop) ──
_sentence_override = False
_sentence_override_value = ""

# ── Live camera loop ──────────────────────────────────────────────────────────
async def camera_loop():
    global _sentence_override, _sentence_override_value

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sentence   = ""
    last_sign  = ""
    sign_start = 0.0
    HOLD_TIME  = 1.5
    CLEAR_TIME = 2.0

    print("Camera started. Open index.html in your browser.")

    while True:
        ok, frame = cap.read()
        if not ok:
            await asyncio.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        now   = time.time()

        # ── Apply any sentence override from browser Back/Clear buttons ──────
        if _sentence_override:
            sentence = _sentence_override_value
            _sentence_override = False
            _sentence_override_value = ""

        annotated, current_sign, confidence = process_frame(frame)
        progress = 0.0

        if current_sign == last_sign and current_sign not in ("...",):
            elapsed  = now - sign_start
            hold     = CLEAR_TIME if current_sign == "FIST" else HOLD_TIME
            progress = min(elapsed / hold, 1.0)

            if progress >= 1.0:
                if current_sign == "FIST":
                    sentence = ""
                    await broadcast({"type":"typed","action":"clear","sentence":""})
                elif current_sign == "THUMBS DOWN":
                    if sentence.endswith(' '):
                        trimmed = sentence.rstrip()
                        last_space = trimmed.rfind(' ')
                        sentence = trimmed[:last_space] + ' ' if last_space >= 0 else ''
                    else:
                        sentence = sentence[:-1]
                    await broadcast({"type":"typed","action":"backspace","sentence":sentence})
                else:
                    if len(current_sign) == 1:
                        sentence += current_sign
                    else:
                        if sentence and sentence[-1] != " ": sentence += " "
                        sentence += current_sign + " "
                    await broadcast({"type":"typed","action":"type","sign":current_sign,"sentence":sentence})
                last_sign  = ""
                sign_start = now
        else:
            last_sign  = current_sign
            sign_start = now

        if clients:
            b64 = frame_to_b64(annotated, quality=55)
            await broadcast({
                "type":       "frame",
                "image":      b64,
                "sign":       current_sign,
                "confidence": round(confidence, 3),
                "progress":   round(progress, 3),
                "sentence":   sentence,
            })

        await asyncio.sleep(0.033)

    cap.release()

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    server = await websockets.serve(ws_handler, "localhost", 8765)
    print("SignBridge server running at ws://localhost:8765")
    await asyncio.gather(server.wait_closed(), camera_loop())

if __name__ == "__main__":
    asyncio.run(main())