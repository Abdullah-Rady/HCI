import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)
hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)

results0 = hands0.process(frame0)
results1 = hands1.process(frame1)
