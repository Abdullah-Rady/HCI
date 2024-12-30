import cv2

# Open the first camera
cap1 = cv2.VideoCapture(4)

# Check if the camera was opened successfully
if not cap1.isOpened():
    print("Error opening video stream 1")

# Open the second camera
cap2 = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not cap2.isOpened():
    print("Error opening video stream 2")

# Loop through the frames from both cameras
while True:
    # Read the frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # If either camera read fails, break the loop
    if not ret1 or not ret2:
        break

    # Display the frames from both cameras
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Release both cameras and close all windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
