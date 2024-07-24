# from onvif import ONVIFCamera
# import cv2
# import threading
# import time
#
# # Camera connection details
# camera1_details = {
#     'host': '192.168.1.32',
#     'port': 80,
#     'user': 'admin',
#     'pass': '123456'
# }
#
# camera2_details = {
#     'host': '192.168.1.33',
#     'port': 80,
#     'user': 'admin',
#     'pass': '123456'
# }
#
# # Initialize the ONVIF cameras
# camera1 = ONVIFCamera(camera1_details['host'], camera1_details['port'], camera1_details['user'], camera1_details['pass'])
# camera2 = ONVIFCamera(camera2_details['host'], camera2_details['port'], camera2_details['user'], camera2_details['pass'])
#
# # Create media services
# media_service1 = camera1.create_media_service()
# media_service2 = camera2.create_media_service()
#
# # Get the stream URIs
# profiles1 = media_service1.GetProfiles()
# profiles2 = media_service2.GetProfiles()
#
# stream_uri1 = media_service1.GetStreamUri({
#     'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
#     'ProfileToken': profiles1[0].token
# }).Uri
#
# stream_uri2 = media_service2.GetStreamUri({
#     'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
#     'ProfileToken': profiles2[0].token
# }).Uri
#
# # Initialize the video capture objects with the stream URIs
# cap1 = cv2.VideoCapture(stream_uri1)
# cap2 = cv2.VideoCapture(stream_uri2)
#
# if not cap1.isOpened():
#     print("Error: Cannot open camera 1")
#     exit()
#
# if not cap2.isOpened():
#     print("Error: Cannot open camera 2")
#     exit()
#
# # Desired frame width and height
# desired_width = 640
# desired_height = 480
#
# # Variables to store frames
# frame1 = None
# frame2 = None
# lock1 = threading.Lock()
# lock2 = threading.Lock()
#
# def capture_frames(cap, lock, frame_var_name, drop_frames=5):
#     global frame1, frame2
#     count = 0
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             if count % drop_frames == 0:
#                 frame = cv2.resize(frame, (desired_width, desired_height))
#                 with lock:
#                     globals()[frame_var_name] = frame
#             count += 1
#         time.sleep(0.01)  # Small delay to reduce CPU usage
#
# # Start threads to
# #
# # .
# #
# # capture frames
# thread1 = threading.Thread(target=capture_frames, args=(cap1, lock1, 'frame1'))
# thread2 = threading.Thread(target=capture_frames, args=(cap2, lock2, 'frame2'))
#
# thread1.start()
# thread2.start()
#
# while True:
#     # Get the latest frames from both cameras
#     with lock1:
#         frame1_copy = frame1.copy() if frame1 is not None else None
#     with lock2:
#         frame2_copy = frame2.copy() if frame2 is not None else None
#
#     if frame1_copy is not None and frame2_copy is not None:
#         # Combine the frames side by side
#         combined_frame = cv2.hconcat([frame1_copy, frame2_copy])
#         # Display the combined frame
#         cv2.imshow('Cameras', combined_frame)
#
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the capture objects and close the display window
# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()
#

from onvif import ONVIFCamera
import cv2
import threading
import time

# Camera connection details
camera_details = {
    'host': '192.168.1.33',
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

while True:
    # Get the latest frame from the camera if a new frame is available
    with lock:
        if new_frame_available:
            frame_copy = frame.copy()
            new_frame_available = False
        else:
            frame_copy = None

    if frame_copy is not None:
        # Display the frame
        cv2.imshow('Camera', frame_copy)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the display window
cap.release()
cv2.destroyAllWindows()
