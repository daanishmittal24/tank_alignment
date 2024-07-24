
import cv2

# Replace with your IP camera's RTSP URL, including authentication
ip_camera_url = 'rtsp://admin:123456@192.168.1.33:554/stream1'

# Create a VideoCapture object for the camera
cap = cv2.VideoCapture(ip_camera_url)

# Check if the camera stream has opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

try:
    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Failed to retrieve frame from camera.")
            break

        # Get the frame dimensions
        height, width, _ = frame.shape

        # Calculate the position of the middle vertical line
        middle_x = width // 2

        # Draw a vertical line in the middle of the frame
        color = (0, 255, 0)  # Line color (Green in BGR)
        thickness = 2        # Line thickness
        cv2.line(frame, (middle_x, 0), (middle_x, height), color, thickness)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted, stopping...")

finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
