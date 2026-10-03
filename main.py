from gestures import detect_gesture
import cv2
import mediapipe as mp
import math
from PIL import Image, ImageSequence
import numpy as np
import time

GESTURE_IMAGES = {
    "POINT": "memes/point.jpg",
    "PEACE": "memes/peace-out.gif",
    "FIST": "memes/fist.jpeg",
    "OPEN PALM": "memes/open_palm.jpg",
    "THUMBS UP": "memes/thumbs_up.jpg"
}

def resize_meme(image, max_width=500, max_height=400):
    height, width = image.shape[:2]

    scale = min(max_width / width, max_height / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height))

def load_media(path):
    image = Image.open(path)

    frames = []

    if getattr(image, "is_animated", False):
        for frame in ImageSequence.Iterator(image):
            frame = frame.convert("RGB")
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
    else:
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frames.append(image)

    return frames

MEDIA = {
    gesture: load_media(path)
    for gesture, path in GESTURE_IMAGES.items()
}

MODEL_PATH = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Meme", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Meme", 500, 400)

current_gesture = None
current_frame = 0
last_frame_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not success:
            print("Could not access camera crodie")
            break

        # bgr to rgb conversion gng
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # convert from opencv to mediapipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # hand detection main part, hopefully works
        results = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )
        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            gesture = detect_gesture(hand)

            if gesture in MEDIA:

                # If we changed gestures, start the new meme from frame 0
                if gesture != current_gesture:
                    current_gesture = gesture
                    current_frame = 0
                    last_frame_time = time.time()

                frames = MEDIA[gesture]

                # Move to the next frame every 0.1 seconds
                if time.time() - last_frame_time >= 0.1:
                    current_frame = (current_frame + 1) % len(frames)
                    last_frame_time = time.time()

                meme = frames[current_frame]

                meme = resize_meme(meme)

                cv2.imshow("Meme", meme)

            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            cv2.putText(
                frame,
                gesture,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        

        frame_timestamp_ms += 33

        cv2.imshow("Gesture Meme", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()