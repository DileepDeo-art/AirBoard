import cv2
import mediapipe as mp
import numpy as np
import math
import time


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

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


# =========================================================
# HAND CONNECTIONS
# =========================================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17),
]


# =========================================================
# COLORS
# OpenCV uses BGR
# =========================================================

COLORS = [
    ("RED", (0, 0, 255)),
    ("BLUE", (255, 0, 0)),
    ("GREEN", (0, 255, 0)),
    ("YELLOW", (0, 255, 255)),
    ("PURPLE", (255, 0, 255)),
    ("WHITE", (255, 255, 255)),
]

color_index = 0
PEN_COLOR = COLORS[color_index][1]


# =========================================================
# PEN SETTINGS
# =========================================================

PEN_THICKNESS = 6

MIN_THICKNESS = 3
MAX_THICKNESS = 25

SMOOTHING = 0.25


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():
    print("❌ Could not open webcam")
    exit()


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================================================
# DRAWING VARIABLES
# =========================================================

drawing_layer = None

previous_point = None

smoothed_x = None
smoothed_y = None


# =========================================================
# GESTURE VARIABLES
# =========================================================

last_color_change_time = 0
COLOR_CHANGE_DELAY = 0.8

pinch_active = False


# =========================================================
# FINGER DETECTION
# =========================================================

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

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = math.degrees(
        math.acos(cosine)
    )

    straight = angle > 145

    return long_enough and straight


# =========================================================
# THUMB DETECTION
# =========================================================

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


# =========================================================
# PINCH DISTANCE
# =========================================================

def thumb_index_distance(hand):

    thumb = hand[4]
    index = hand[8]

    return math.hypot(
        thumb.x - index.x,
        thumb.y - index.y
    )


# =========================================================
# DRAW HAND
# =========================================================

def draw_hand(frame, hand):

    height, width, _ = frame.shape

    points = []

    for landmark in hand:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections
    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Draw points
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


# =========================================================
# START
# =========================================================

start_time = time.time()


with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # =================================================
        # READ CAMERA
        # =================================================

        success, frame = camera.read()

        if not success:

            print("❌ Could not read webcam")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape


        # =================================================
        # CREATE DRAWING LAYER
        # =================================================

        if drawing_layer is None:

            drawing_layer = np.zeros_like(frame)


        # =================================================
        # MEDIAPIPE IMAGE
        # =================================================

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


        # =================================================
        # DEFAULT VALUES
        # =================================================

        is_drawing = False
        is_erasing = False

        gesture_name = "NO HAND"


        # =================================================
        # HAND FOUND
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # Draw hand skeleton
            points = draw_hand(
                frame,
                hand
            )


            # =================================================
            # FINGER STATES
            # =================================================

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


            # =================================================
            # GESTURES
            # =================================================

            # -----------------------------------------------
            # OPEN PALM
            # -----------------------------------------------

            open_palm = (
                index_open
                and middle_open
                and ring_open
                and pinky_open
                and thumb_open
            )


            # -----------------------------------------------
            # FIST
            # -----------------------------------------------

            fist = (
                not index_open
                and not middle_open
                and not ring_open
                and not pinky_open
            )


            # -----------------------------------------------
            # TWO FINGERS
            # Index + Middle
            # -----------------------------------------------

            two_fingers = (
                index_open
                and middle_open
                and not ring_open
                and not pinky_open
            )


            # -----------------------------------------------
            # INDEX ONLY
            # -----------------------------------------------

            index_only = (
                index_open
                and not middle_open
                and not ring_open
                and not pinky_open
                and not thumb_open
            )


            # -----------------------------------------------
            # PINCH
            # -----------------------------------------------

            pinch_distance = thumb_index_distance(hand)

            pinch = pinch_distance < 0.06


            # =================================================
            # PINCH = CHANGE PEN SIZE
            # =================================================

            if pinch and not two_fingers:

                gesture_name = "🤏 PINCH - SIZE"

                previous_point = None

                # Distance between thumb and index
                # controls thickness

                size_value = int(
                    np.interp(
                        pinch_distance,
                        [0.015, 0.06],
                        [MAX_THICKNESS, MIN_THICKNESS]
                    )
                )

                PEN_THICKNESS = max(
                    MIN_THICKNESS,
                    min(MAX_THICKNESS, size_value)
                )

                pinch_active = True


            else:

                pinch_active = False


            # =================================================
            # TWO FINGERS = CHANGE COLOR
            # =================================================

            if two_fingers and not pinch:

                current_time = time.time()

                if (
                    current_time
                    - last_color_change_time
                    > COLOR_CHANGE_DELAY
                ):

                    color_index = (
                        color_index + 1
                    ) % len(COLORS)

                    PEN_COLOR = COLORS[color_index][1]

                    last_color_change_time = current_time


                gesture_name = (
                    "✌ COLOR: "
                    + COLORS[color_index][0]
                )

                previous_point = None


            # =================================================
            # FIST = ERASER
            # =================================================

            elif fist:

                is_erasing = True

                gesture_name = (
                    "✊ ERASER"
                )


            # =================================================
            # INDEX = DRAW
            # =================================================

            elif index_only:

                is_drawing = True

                gesture_name = (
                    "☝ DRAW - "
                    + COLORS[color_index][0]
                )


            # =================================================
            # OPEN PALM = STOP
            # =================================================

            elif open_palm:

                is_drawing = False

                gesture_name = (
                    "✋ STOP"
                )

                previous_point = None


            # =================================================
            # OTHER GESTURE
            # =================================================

            else:

                is_drawing = False

                gesture_name = "OTHER GESTURE"

                previous_point = None


            # =================================================
            # SMOOTH INDEX POSITION
            # =================================================

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


            # =================================================
            # DRAW / ERASE
            # =================================================

            if is_drawing or is_erasing:

                if previous_point is not None:

                    movement = math.hypot(
                        smooth_point[0]
                        - previous_point[0],

                        smooth_point[1]
                        - previous_point[1]
                    )


                    if movement > 2:

                        # -----------------------------------
                        # ERASER
                        # -----------------------------------

                        if is_erasing:

                            cv2.line(
                                drawing_layer,
                                previous_point,
                                smooth_point,
                                (0, 0, 0),
                                PEN_THICKNESS * 2,
                                cv2.LINE_AA
                            )


                        # -----------------------------------
                        # PEN
                        # -----------------------------------

                        else:

                            cv2.line(
                                drawing_layer,
                                previous_point,
                                smooth_point,
                                PEN_COLOR,
                                PEN_THICKNESS,
                                cv2.LINE_AA
                            )


                previous_point = smooth_point

            else:

                previous_point = None


            # =================================================
            # SHOW CURSOR
            # =================================================

            if is_drawing:

                cv2.circle(
                    frame,
                    smooth_point,
                    7,
                    PEN_COLOR,
                    -1
                )

            elif is_erasing:

                cv2.circle(
                    frame,
                    smooth_point,
                    PEN_THICKNESS,
                    (255, 255, 255),
                    2
                )


        # =====================================================
        # NO HAND
        # =====================================================

        else:

            previous_point = None

            smoothed_x = None
            smoothed_y = None


        # =====================================================
        # PUT DRAWING ON CAMERA
        # =====================================================

        drawing_mask = np.any(
            drawing_layer > 0,
            axis=2
        )


        frame[drawing_mask] = (
            drawing_layer[drawing_mask]
        )


        # =====================================================
        # STATUS COLOR
        # =====================================================

        if is_drawing:

            status_color = (0, 255, 0)

        elif is_erasing:

            status_color = (255, 255, 255)

        else:

            status_color = (0, 0, 255)


        # =====================================================
        # TOP STATUS BOX
        # =====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (460, 115),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            gesture_name,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            status_color,
            2,
            cv2.LINE_AA
        )


        # Current color
        cv2.putText(
            frame,
            "COLOR: " + COLORS[color_index][0],
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            PEN_COLOR,
            2,
            cv2.LINE_AA
        )


        # Current size
        cv2.putText(
            frame,
            "SIZE: " + str(PEN_THICKNESS),
            (250, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # =====================================================
        # INSTRUCTIONS
        # =====================================================

        cv2.putText(
            frame,
            "INDEX = DRAW",
            (20, height - 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "TWO FINGERS = COLOR",
            (20, height - 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "PINCH = SIZE",
            (20, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "FIST = ERASER | PALM = STOP",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # =====================================================
        # SHOW
        # =====================================================

        cv2.imshow(
            "AIRBOARD",
            frame
        )


        # =====================================================
        # KEYBOARD CONTROLS
        # =====================================================

        key = cv2.waitKey(1) & 0xFF


        # Q = Quit
        if key == ord("q"):

            break


        # C = Clear
        if key == ord("c"):

            drawing_layer = np.zeros_like(frame)

            previous_point = None


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()