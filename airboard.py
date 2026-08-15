import cv2
import mediapipe as mp
import numpy as np
import math
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.7,
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open webcam")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

drawing_layer = None
previous_point = None

PEN_COLOR = (0, 0, 255)
PEN_THICKNESS = 4

SMOOTHING = 0.25

smoothed_x = None
smoothed_y = None

def finger_extended(hand, tip, pip, mcp):
    wrist = hand[0]

    tip_point = hand[tip]
    pip_point = hand[pip]
    mcp_point = hand[mcp]

    tip_distance = math.hypot(
        tip_point.x - wrist.x,
        tip_point.y - wrist.y
    )

    pip_distance = math.hypot(
        pip_point.x - wrist.x,
        pip_point.y - wrist.y
    )

    long_enough = (
        tip_distance > pip_distance * 1.15
    )

    a = np.array([
        mcp_point.x,
        mcp_point.y
    ])

    b = np.array([
        pip_point.x,
        pip_point.y
    ])

    c = np.array([
        tip_point.x,
        tip_point.y
    ])

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return False

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = math.degrees(
        math.acos(cosine)
    )

    straight = angle > 145

    return long_enough and straight

def thumb_extended(hand):
    wrist = hand[0]
    thumb_tip = hand[4]
    thumb_ip = hand[3]

    tip_distance = math.hypot(
        thumb_tip.x - wrist.x,
        thumb_tip.y - wrist.y
    )

    ip_distance = math.hypot(
        thumb_ip.x - wrist.x,
        thumb_ip.y - wrist.y
    )

    return tip_distance > ip_distance * 1.15

def draw_hand(frame, hand):
    height, width, _ = frame.shape

    points = []

    for landmark in hand:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    for i, point in enumerate(points):

        if i == 8:

            cv2.circle(
                frame,
                point,
                10,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                point,
                13,
                (255, 255, 255),
                2
            )

        else:

            cv2.circle(
                frame,
                point,
                4,
                (0, 255, 0),
                -1
            )

    return points

start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Could not read webcam")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        if drawing_layer is None:

            drawing_layer = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        is_drawing = False
        gesture_name = "NO HAND"

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            points = draw_hand(
                frame,
                hand
            )

            index_open = finger_extended(
                hand,
                tip=8,
                pip=6,
                mcp=5
            )

            middle_open = finger_extended(
                hand,
                tip=12,
                pip=10,
                mcp=9
            )

            ring_open = finger_extended(
                hand,
                tip=16,
                pip=14,
                mcp=13
            )

            pinky_open = finger_extended(
                hand,
                tip=20,
                pip=18,
                mcp=17
            )

            thumb_open = thumb_extended(hand)

            index_only = (
                index_open
                and not middle_open
                and not ring_open
                and not pinky_open
                and not thumb_open
            )

            open_palm = (
                index_open
                and middle_open
                and ring_open
                and pinky_open
                and thumb_open
            )

            if index_only:

                is_drawing = True
                gesture_name = "✏ INDEX ONLY - DRAW"

            elif open_palm:

                is_drawing = False
                gesture_name = "✋ OPEN PALM - STOP"

                previous_point = None

            else:

                is_drawing = False
                gesture_name = "OTHER GESTURE"

                previous_point = None

            raw_x, raw_y = points[8]

            if smoothed_x is None:

                smoothed_x = float(raw_x)
                smoothed_y = float(raw_y)

            else:

                smoothed_x = (
                    SMOOTHING * raw_x
                    + (1 - SMOOTHING) * smoothed_x
                )

                smoothed_y = (
                    SMOOTHING * raw_y
                    + (1 - SMOOTHING) * smoothed_y
                )

            smooth_point = (
                int(smoothed_x),
                int(smoothed_y)
            )

            if is_drawing:

                if previous_point is not None:

                    movement = math.hypot(
                        smooth_point[0]
                        - previous_point[0],

                        smooth_point[1]
                        - previous_point[1]
                    )

                    if movement > 2:

                        cv2.line(
                            drawing_layer,
                            previous_point,
                            smooth_point,
                            PEN_COLOR,
                            PEN_THICKNESS,
                            cv2.LINE_AA
                        )

                previous_point = smooth_point

            cv2.circle(
                frame,
                smooth_point,
                7,
                (0, 0, 255),
                -1
            )

        else:

            previous_point = None

            smoothed_x = None
            smoothed_y = None

        drawing_mask = np.any(
            drawing_layer > 0,
            axis=2
        )

        frame[drawing_mask] = PEN_COLOR

        if is_drawing:

            status_color = (0, 255, 0)

        else:

            status_color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (10, 10),
            (390, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            gesture_name,
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "INDEX ONLY = DRAW",
            (20, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "OPEN PALM = STOP",
            (20, height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "AIRBOARD",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):

            drawing_layer = np.zeros_like(frame)

            previous_point = None

camera.release()
cv2.destroyAllWindows()