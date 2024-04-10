import cv2
import numpy as np

def detect_heat_source(frame):
    # Convert frame to HSV color space
    # Convert the captured frame to HSV color space for filtering
    filter_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Define the lower and upper bounds for the HSV filter
    lower_bound = np.array([10, 0, 0])
    upper_bound = np.array([60, 255, 255])

    mask = cv2.inRange(filter_frame, lower_bound, upper_bound)

    # Use morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Removes small objects/noise from the foreground
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Closes small holes in the foreground objects
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Set up the blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByColor = True
    params.blobColor =255
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByArea = True
    params.minArea = 500  # Adjust this threshold as needed to filter out small blobs

    # Create the blob detector with the specified parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Perform blob detection
    keypoints = detector.detect(result)

    # Draw detected blobs on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255, 0),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
    return result

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
