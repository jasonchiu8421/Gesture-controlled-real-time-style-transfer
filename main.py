import cv2
import torch
import time
import os
import numpy as np
import mediapipe as mp

import transformer
import utils

# ================= CONFIG =================
STYLE_LIST = [
    "transforms/mosaic.pth",
    "transforms/udnie.pth",
    "transforms/starry.pth"
]

WIDTH, HEIGHT = 640, 480
INPUT_SIZE = 256

device = torch.device("cpu")

# ================= OUTPUT DIR =================
SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_next_index():
    files = os.listdir(SAVE_DIR)
    nums = []
    for f in files:
        if f.endswith(".jpg"):
            name = f.split(".")[0]
            if name.isdigit():
                nums.append(int(name))
    return max(nums) + 1 if nums else 1

# ================= LOAD MODELS =================
models = []

def load_all_models():
    for p in STYLE_LIST:
        net = transformer.TransformerNetwork().to(device)
        net.load_state_dict(torch.load(p, map_location=device))
        net.eval()
        models.append(net)

load_all_models()

# ================= HAND TRACKING =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ================= SIMPLE GESTURE =================
def detect_gesture(lms):

    tips = [4, 8, 12, 16, 20]
    fingers = []

    # thumb
    fingers.append(1 if lms.landmark[4].x < lms.landmark[3].x else 0)

    # other fingers
    for i in range(1, 5):
        fingers.append(1 if lms.landmark[tips[i]].y < lms.landmark[tips[i]-2].y else 0)

    # ✌️ = screenshot (now style output)
    if fingers == [0,1,1,0,0]:
        return "peace"

    # 👊 = switch style
    if fingers == [0,0,0,0,0]:
        return "fist"

    # 👍 = enable
    if fingers == [1,0,0,0,0]:
        return "thumb"

    # ✋ = disable
    if fingers == [1,1,1,1,1]:
        return "open"

    return None

# ================= MAIN =================
def webcam():

    cam = cv2.VideoCapture(0)
    cam.set(3, WIDTH)
    cam.set(4, HEIGHT)

    idx = 0
    net = models[idx]
    enable_style = True

    last_action = 0

    print("🔥 Final Stable Style Transfer + Screenshot System")

    while True:

        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ================= HAND =================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = None

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand)

        # ================= CONTROL =================
        now = time.time()

        if gesture == "peace" and now - last_action > 0.8:

            # ⭐ SAVE STYLE OUTPUT (NOT RAW FRAME)
            idx_save = get_next_index()

            img_small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

            if enable_style:
                tensor = utils.itot(img_small).to(device)
                out = net(tensor)
                output = utils.ttoi(out.detach())
            else:
                output = img_small.copy()

            output = np.clip(output, 0, 255).astype(np.uint8)

            save_path = f"{SAVE_DIR}/{idx_save}.jpg"
            cv2.imwrite(save_path, output)

            print(f"📸 Saved stylized image: {save_path}")

            last_action = now

        elif gesture == "fist" and now - last_action > 0.8:
            idx = (idx + 1) % len(models)
            net = models[idx]
            print("🔄 Style switched")
            last_action = now

        elif gesture == "thumb" and now - last_action > 0.8:
            enable_style = True
            print("👍 Style ON")
            last_action = now

        elif gesture == "open" and now - last_action > 0.8:
            enable_style = False
            print("✋ Style OFF")
            last_action = now

        # ================= STYLE OUTPUT =================
        img_small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

        if enable_style:
            tensor = utils.itot(img_small).to(device)
            out = net(tensor)
            output = utils.ttoi(out.detach())
        else:
            output = img_small.copy()

        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (WIDTH, HEIGHT))

        # ================= UI =================
        cv2.putText(
            output,
            f"Gesture: {gesture} | Style: {idx} | ON: {enable_style}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.imshow("Final Gesture Style System", output)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam()