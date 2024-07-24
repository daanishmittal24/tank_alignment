from onvif import ONVIFCamera
import cv2
import threading
import time
import numpy as np
from collections import deque
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Constants
MASK_PERCENTAGE = 0.2
TRANSPARENCY_ALPHA = 0.3
GREEN_COLOR = (0, 255, 0)
ORANGE_COLOR = (255, 165, 0)
RED_COLOR = (0, 0, 255)
EDGE_COLOR = (128, 0, 128)
LINE_THICKNESS = 2
SMOOTHING_EPSILON_FACTOR = 0.05
FRAME_HISTORY = 2
TARGET_FPS = 5
FRAME_SKIP = 3  # Skip frames to reduce processing load

# Deque to store edge positions
edge_history1 = deque(maxlen=FRAME_HISTORY)
edge_history2 = deque(maxlen=FRAME_HISTORY)

# Camera connection details
camera1_details = {
    'host': '192.168.1.9',
    'port': 80,
    'user': 'admin',
    'pass': '123456'
}

camera2_details = {
    'host': '192.168.1.33',
    'port': 80,
    'user': 'admin',
    'pass': '123456'
}

# Initialize the ONVIF cameras
camera1 = ONVIFCamera(camera1_details['host'], camera1_details['port'], camera1_details['user'],
                      camera1_details['pass'])
camera2 = ONVIFCamera(camera2_details['host'], camera2_details['port'], camera2_details['user'],
                      camera2_details['pass'])

# Create media services
media_service1 = camera1.create_media_service()
media_service2 = camera2.create_media_service()

# Get the stream URIs
profiles1 = media_service1.GetProfiles()
profiles2 = media_service2.GetProfiles()

stream_uri1 = media_service1.GetStreamUri({
    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
    'ProfileToken': profiles1[0].token
}).Uri

stream_uri2 = media_service2.GetStreamUri({
    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
    'ProfileToken': profiles2[0].token
}).Uri

# Initialize the video capture objects with the stream URIs
cap1 = cv2.VideoCapture(stream_uri1)
cap2 = cv2.VideoCapture(stream_uri2)

if not cap1.isOpened():
    print("Error: Cannot open camera 1")
    exit()

if not cap2.isOpened():
    print("Error: Cannot open camera 2")
    exit()

# Desired frame width and height
desired_width = 640
desired_height = 480

# Variables to store frames and a flag to indicate a new frame is available
frame1 = None
frame2 = None
lock1 = threading.Lock()
lock2 = threading.Lock()
new_frame_available1 = False
new_frame_available2 = False


def load_model(config_file, weights_file):
    print("Loading model...")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.96  # Set a custom testing threshold
    print("Model loaded successfully.")
    return DefaultPredictor(cfg)


def mask_sides(image, mask_percentage):
    h, w = image.shape[:2]
    mask_width = int(w * mask_percentage)
    mask = np.ones(image.shape[:2], dtype=np.uint8)
    mask[:, :mask_width] = 0
    mask[:, -mask_width:] = 0
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask


def tint_screen(image, color, alpha):
    overlay = np.zeros_like(image)
    if color == 'green':
        overlay[:] = GREEN_COLOR
    elif color == 'orange':
        overlay[:] = ORANGE_COLOR
    else:
        overlay[:] = RED_COLOR
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def calculate_angle(vector1, vector2):
    if vector1[1] > 0:
        vector1 = -vector1
    if vector2[1] > 0:
        vector2 = -vector2
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(point, line_x):
    return abs(point[0] - line_x)


def find_right_edge(contour):
    epsilon = SMOOTHING_EPSILON_FACTOR * cv2.arcLength(contour, True)
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx_contour) > 4:
        approx_contour = cv2.convexHull(approx_contour)
        approx_contour = cv2.approxPolyDP(approx_contour, epsilon, True)
    approx_contour = approx_contour[:4]
    rightmost_points = sorted(approx_contour, key=lambda x: x[0][0])[-2:]
    right_edge_start, right_edge_end = sorted(rightmost_points, key=lambda x: x[0][1])
    return right_edge_start[0], right_edge_end[0]


def average_edge_position(edge_history):
    avg_start = np.mean([edge[0] for edge in edge_history], axis=0).astype(int)
    avg_end = np.mean([edge[1] for edge in edge_history], axis=0).astype(int)
    return tuple(avg_start), tuple(avg_end)


def create_colored_mask(mask, color):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 1] = color
    return colored_mask


def process_frame(frame, predictor, mask_percentage, edge_history):
    masked_frame, frame_mask = mask_sides(frame, mask_percentage)
    outputs = predictor(masked_frame)
    instances = outputs["instances"].to("cpu")
    ramp_instances = instances[instances.pred_classes == 1]
    draw_frame = masked_frame.copy()
    angle_degrees = None  # Initialize the angle variable

    for i in range(len(ramp_instances)):
        if ramp_instances.has("pred_masks"):
            mask = ramp_instances.pred_masks[i].numpy()
            mask = cv2.bitwise_and(mask.astype(np.uint8), frame_mask)

            # Create a colored mask for the segmented highlight
            highlight_color = (0, 255, 255)  # Yellow color
            colored_mask = create_colored_mask(mask, highlight_color)

            # Blend the colored mask with the original frame
            cv2.addWeighted(draw_frame, 1, colored_mask, 0.5, 0, draw_frame)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                right_edge_start, right_edge_end = find_right_edge(largest_contour)
                edge_history.append((right_edge_start, right_edge_end))
                if len(edge_history) == FRAME_HISTORY:
                    right_edge_start, right_edge_end = average_edge_position(edge_history)
                cv2.line(draw_frame, right_edge_start, right_edge_end, EDGE_COLOR, LINE_THICKNESS)

                mid_x = draw_frame.shape[1] // 2
                frame_height = draw_frame.shape[0]
                green_line_start = int(frame_height * 0.28)  # Starting point of the green line (28% from the top)
                green_line_end = frame_height  # Ending point of the green line (bottom of the frame)
                cv2.line(draw_frame, (mid_x, green_line_start), (mid_x, green_line_end), GREEN_COLOR, LINE_THICKNESS)

                vector_right_edge = np.array(
                    [right_edge_end[0] - right_edge_start[0], right_edge_end[1] - right_edge_start[1]])
                vector_green_line = np.array([0, -1])
                angle_degrees = calculate_angle(vector_right_edge, vector_green_line)  # Store the calculated angle

                distances = [calculate_distance(point, mid_x) for point in [right_edge_start, right_edge_end]]
                min_distance = min(distances)
                if angle_degrees < 5:
                    if min_distance < 5:
                        draw_frame = tint_screen(draw_frame, 'green', TRANSPARENCY_ALPHA)
                    elif min_distance < 15:
                        draw_frame = tint_screen(draw_frame, 'orange', TRANSPARENCY_ALPHA)
                    else:
                        draw_frame = tint_screen(draw_frame, 'red', TRANSPARENCY_ALPHA)
                elif angle_degrees < 15:
                    if min_distance < 15:
                        draw_frame = tint_screen(draw_frame, 'orange', TRANSPARENCY_ALPHA)
                    else:
                        draw_frame = tint_screen(draw_frame, 'red', TRANSPARENCY_ALPHA)
                else:
                    draw_frame = tint_screen(draw_frame, 'red', TRANSPARENCY_ALPHA)

    if angle_degrees is not None:
        angle_degrees = int(angle_degrees)  # Convert to integer
        angle_text = f"Angle: {angle_degrees} degrees"
        cv2.putText(draw_frame, angle_text, (10, draw_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_COLOR,
                    1, cv2.LINE_AA)

    return draw_frame


def capture_frames(cap, lock, frame_var_name, new_frame_flag, drop_frames=5):
    global frame, new_frame_available
    count = 0
    while True:
        ret, new_frame = cap.read()
        if ret:
            if count % drop_frames == 0:
                resized_frame = cv2.resize(new_frame, (desired_width, desired_height))
                with lock:
                    globals()[frame_var_name] = resized_frame
                    globals()[new_frame_flag] = True
            count += 1
        time.sleep(0.01)  # Small delay to reduce CPU usage


# Start threads to capture frames
thread1 = threading.Thread(target=capture_frames, args=(cap1, lock1, 'frame1', 'new_frame_available1'))
thread2 = threading.Thread(target=capture_frames, args=(cap2, lock2, 'frame2', 'new_frame_available2'))

thread1.start()
thread2.start()

# Load the model
config_file = "C:/Users/Daanish Mittal/OneDrive/Desktop/Tank_align/tank_alignment/army_ramp/loader.ramp/mask_rcnn_R_101_FPN_3x/2024-07-09-01-55-42/config.yaml"
weights_file = "C:/Users/Daanish Mittal/OneDrive/Desktop/Tank_align/tank_alignment/army_ramp/loader.ramp/mask_rcnn_R_101_FPN_3x/2024-07-09-01-55-42/model_final.pth"
predictor = load_model(config_file, weights_file)

last_process_time = time.time()

while True:
    # Check if enough time has passed to process the next frame
    current_time = time.time()
    if current_time - last_process_time >= 1 / TARGET_FPS:
        last_process_time = current_time

        # Get the latest frames from both cameras if a new frame is available
        with lock1:
            if new_frame_available1:
                frame_copy1 = frame1.copy()
                new_frame_available1 = False
            else:
                frame_copy1 = None

        with lock2:
            if new_frame_available2:
                frame_copy2 = frame2.copy()
                new_frame_available2 = False
            else:
                frame_copy2 = None

        if frame_copy1 is not None and frame_copy2 is not None:
            # Process the frames
            processed_frame1 = process_frame(frame_copy1, predictor, MASK_PERCENTAGE, edge_history1)
            processed_frame2 = process_frame(frame_copy2, predictor, MASK_PERCENTAGE, edge_history2)

            cv2.putText(processed_frame1, 'Left', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2, cv2.LINE_AA)
            cv2.putText(processed_frame2, 'Right', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2, cv2.LINE_AA)


            # Combine the frames side by side
            combined_frame = cv2.hconcat([processed_frame1, processed_frame2])

            # Display the combined frame
            cv2.imshow('Cameras', combined_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture objects and close the display window
cap1.release()
cap2.release()
cv2.destroyAllWindows()