import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

recent_x_positions = []
recent_y_positions = []
max_positions = 20

def detect_heat_source(frame):
     # Convert the captured frame to HSV color space for filtering
    filter_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the HSV filter
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([60, 255, 255])

    hsv_mask = cv2.inRange(filter_frame, lower_bound, upper_bound)

    # Use morphological operations to clean up the mask
    gauss_mask = cv2.GaussianBlur(hsv_mask, (5,5), 0)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(gauss_mask, cv2.MORPH_OPEN, kernel)  # Removes small objects/noise from the foreground
    final_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Closes small holes in the foreground objects


    # Calculate moments of the binary image
    M = cv2.moments(final_mask)

    # Calculate x, y coordinate of center
    if M["m00"] != 0: #M["m00"] >= self.min_area and M["m00"] <= self.max_area:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        #Median filter
        if len(recent_x_positions) >= max_positions:
            recent_x_positions.pop(0)
        if len(recent_y_positions) >= max_positions:
            recent_y_positions.pop(0)

        recent_x_positions.append(cX)
        recent_y_positions.append(cY)

        # Apply Gaussian filtering to recent_x_positions and recent_y_positions
        smoothed_x_positions = gaussian_filter1d(recent_x_positions, sigma=5)
        smoothed_y_positions = gaussian_filter1d(recent_y_positions, sigma=5)

        # Extract single elements from the smoothed arrays
        smooth_cX = int(smoothed_x_positions[-1])  # Extract the last smoothed value
        smooth_cY = int(smoothed_y_positions[-1])  # Extract the last smoothed value

        middle_x = int(frame.shape[1] / 2)
        middle_y = int(frame.shape[0] / 2)
        rel_x = smooth_cX - middle_x
        rel_y = smooth_cY - middle_y
    else:
        # set values as -1 if no mass found
        smooth_cX, smooth_cY = -10000, -10000
        rel_x, rel_y = -10000, -10000 



    if smooth_cX != -10000 and smooth_cY != -10000:

        result = cv2.bitwise_and(frame, frame, mask=final_mask)

        cv2.circle(result, (smooth_cX, smooth_cY), 5, (255, 255, 255), -1)

        # Display the coordinates on the frame
        coord_text = f"({rel_x}, {rel_y})"
        cv2.putText(result, coord_text, (smooth_cX, smooth_cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        result = cv2.bitwise_and(frame, frame, mask=final_mask)

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
