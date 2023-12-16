import cv2
import numpy as np

def detect_heat_source(frame):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for Ironbow colors in HSV corresponding to the warmer range
    lower_red = np.array([0, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the frame to get only the warmer regions
    heat_source_mask = cv2.inRange(hsv_frame, lower_red, upper_yellow)

    # Apply morphological operations to remove noise and make contours "bigger"
    kernel = np.ones((15, 15), np.uint8)
    heat_source_mask = cv2.morphologyEx(heat_source_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the heat source
    contours, _ = cv2.findContours(heat_source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box around the detected heat source
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Open a video capture object
video_capture = cv2.VideoCapture('../videos/video1.mp4')  # Replace with the path to your thermal video file

# Get video properties (width, height, and frames per second)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust codec as needed
output_video = cv2.VideoWriter('../videos/processed_video.avi', fourcc, fps, (frame_width, frame_height))


while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Unable to read frame.")
        break

    # Detect heat sources in the frame
    result = detect_heat_source(frame)

    # Write the processed frame to the output video
    output_video.write(result)

    # Display the result
    cv2.imshow("Heat Sources", result)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release the video capture and output video objects, and close all windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
