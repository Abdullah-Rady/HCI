import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create VideoCapture objects for each camera
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(4)

# Define the resolution of the cameras
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a window to display the output
cv2.namedWindow('Stereo Vision Hand Tracking', cv2.WINDOW_NORMAL)

# Initialize the hand tracking module
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        # Read frames from each camera
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            break

        # Flip the frames horizontally for correct display
        frame1 = cv2.flip(frame1, 1)
        frame2 = cv2.flip(frame2, 1)

        # Convert the frames to RGB format
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Process each frame using Mediapipe
        results1 = hands.process(frame1)
        results2 = hands.process(frame2)

        # Check if hand landmarks are detected in both frames
        if results1.multi_hand_landmarks and results2.multi_hand_landmarks:
            # Get the landmarks for the first hand in each frame
            hand_landmarks1 = results1.multi_hand_landmarks[0]
            hand_landmarks2 = results2.multi_hand_landmarks[0]

            # Create arrays to store the 2D landmark coordinates
            landmarks_2d_1 = np.zeros((21, 2))
            landmarks_2d_2 = np.zeros((21, 2))

            # Extract the 2D landmark coordinates for each hand
            for i, landmark in enumerate(hand_landmarks1.landmark):
                x1, y1 = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, frame1.shape[1], frame1.shape[0])
                landmarks_2d_1[i] = [x1, y1]

            for i, landmark in enumerate(hand_landmarks2.landmark):
                x2, y2, _ = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, frame2.shape[1], frame2.shape[0])
                landmarks_2d_2[i] = [x2, y2]

            # Perform triangulation to get the 3D coordinates of the hand landmarks
            points_3d = cv2.triangulatePoints(
                np.hstack((np.eye(3), np.zeros((3, 1)))),
                np.hstack((np.eye(3), np.zeros((3, 1)))),
                landmarks_2d_1.T, landmarks_2d_2.T).T[:, :3]

            # Draw the hand landmarks and lines connecting them in each frame
            mp_drawing.draw_landmarks(frame1, hand_landmarks1, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame2, hand_landmarks2, mp_hands.HAND_CONNECTIONS)

            # Draw the 3D points on the left frame
            for i, point in enumerate(points_3d):
                x, y, z = point
                x, y, _ = mp_drawing._normalized_to_pixel_coordinates(
                    x / z, y / z, frame1.shape[1], frame1.shape[0])
                cv2.circle(frame1, (int(x), int(y)), 5, (255, 0, 0), -1)

            # Display the frames in the window
            output_frame = np.concatenate((frame1, frame2), axis=1)
            cv2.imshow('Stereo Vision Hand Tracking', output_frame)

            # Check for user input to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the VideoCapture objects and destroy the window
cap1.release()
cap2.release()
cv2.destroyAllWindows()

