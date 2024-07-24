from onvif import ONVIFCamera
import cv2
import threading
import time
import torch
from collections import deque
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np

# Constants
MASK_PERCENTAGE = 0.2
TRANSPARENCY_ALPHA = 0.3
GREEN_COLOR = (0, 255, 0)
ORANGE_COLOR = (255, 165, 0)
RED_COLOR = (0, 0, 255)
EDGE_COLOR = (128, 0, 128)
LINE_THICKNESS = 2
SMOOTHING_EPSILON_FACTOR = 0.05
FRAME_HISTORY = 5
TARGET_FPS = 5

# Deque to store edge positions
edge_history = deque(maxlen=FRAME_HISTORY)


def load_model(config_file, weights_file):
    print("Loading model...")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.97  # Set a custom testing threshold
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


def process_frame(frame, predictor, mask_percentage):
    masked_frame, frame_mask = mask_sides(frame, mask_percentage)
    outputs = predictor(masked_frame)
    instances = outputs["instances"].to("cpu")
    ramp_instances = instances[instances.pred_classes == 1]
    draw_frame = masked_frame.copy()
    for i in range(len(ramp_instances)):
        if ramp_instances.has("pred_masks"):
            mask = ramp_instances.pred_masks[i].numpy()
            mask = cv2.bitwise_and(mask.astype(np.uint8), frame_mask)
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
                angle_degrees = calculate_angle(vector_right_edge, vector_green_line)
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
    return draw_frame


# Camera connection details
camera_details = {
    'host': '192.168.1.9',
    'port': 80,
    'user': 'admin',
    'pass': '123456'
}

# Initialize the ONVIF camera
camera = ONVIFCamera(camera_details['host'], camera_details['port'], camera_details['user'], camera_details['pass'])

# Create media services
media_service = camera.create_media_service()

# Get the stream URI
profiles = media_service.GetProfiles()
stream_uri = media_service.GetStreamUri({
    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
    'ProfileToken': profiles[0].token
}).Uri

# Initialize the video capture object with the stream URI
cap = cv2.VideoCapture(stream_uri)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Desired frame width and height
desired_width = 640
desired_height = 480

# Variable to store frame and a flag to indicate a new frame is available
frame = None
lock = threading.Lock()
new_frame_available = False


def capture_frames(cap, lock, drop_frames=5):
    global frame, new_frame_available
    count = 0
    while True:
        ret, new_frame = cap.read()
        if ret:
            if count % drop_frames == 0:
                resized_frame = cv2.resize(new_frame, (desired_width, desired_height))
                with lock:
                    frame = resized_frame
                    new_frame_available = True
            count += 1
        time.sleep(0.01)  # Small delay to reduce CPU usage


# Start thread to capture frames
thread = threading.Thread(target=capture_frames, args=(cap, lock))
thread.start()

# Load the model
config_file = "C:/Users/Daanish Mittal/OneDrive/Desktop/Tank_align/tank_alignment/army_ramp/loader.ramp/mask_rcnn_R_101_FPN_3x/2024-07-09-01-55-42/config.yaml"
weights_file = "C:/Users/Daanish Mittal/OneDrive/Desktop/Tank_align/tank_alignment/army_ramp/loader.ramp/mask_rcnn_R_101_FPN_3x/2024-07-09-01-55-42/model_final.pth"
predictor = load_model(config_file, weights_file)

while True:
    # Get the latest frame from the camera if a new frame is available
    with lock:
        if new_frame_available:
            frame_copy = frame.copy()
            new_frame_available = False
        else:
            frame_copy = None

    if frame_copy is not None:
        # Process and display the frame
        processed_frame = process_frame(frame_copy, predictor, MASK_PERCENTAGE)
        cv2.imshow('Camera Feed', processed_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the display window
cap.release()
cv2.destroyAllWindows()
