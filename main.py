from gestures import detect_gesture
import cv2
import mediapipe as mp
import math

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
with HandLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while True:
        success, frame = cap.read()

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