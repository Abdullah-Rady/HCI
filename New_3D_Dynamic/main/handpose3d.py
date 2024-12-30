import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk, addArrays, divArrays, subArrays
from Hands_On_Mouse import locate_color


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

actions = np.array(['idle','click1'])

hand_index = {
    'knuckle': [5, 9, 13, 17],
    'index': [8, 7, 6],
    'thumb': [4, 3, 2, 1],
    'pinky': [20, 19, 18],
    'middle': [12, 11, 10],
    'ring': [16, 15, 14],
    'wrist': [0]
}

def calculate_direction(point1, point2, thresholdx, thresholdy, thresholdz = sys.maxsize):


    left, right, upward, downward, upward_z, downward_z   = False, False, False, False, False, False

    #if the two points are in the camera view
    if (point2[0] == -1 and point2[1] == -1 and point2[2] == -1) or (point1[0] == -1 and point1[1] == -1 and point1[2] == -1):
        return left, right, upward, downward, upward_z, downward_z, 0

    diff = point2 - point1
    if diff[0] > thresholdx:
        left = True;
    if diff[0] < -thresholdx:
        right = True;

    if diff[1] > thresholdy:
        upward = True
    if diff[1] < -thresholdy:
        downward = True

    if diff[2] > thresholdz:
        upward_z = True
    if diff[2] < -thresholdz:
        downward_z = True

    return left, right, upward, downward, upward_z, downward_z, diff
        



def watch_finger(finger):

    finger_index = hand_index[finger]
    prev_finger_position = []
    

    def f(frame_p3ds):
        nonlocal prev_finger_position  
        
        finger_position = [frame_p3ds[i] for i in finger_index]
        
        prev_finger_position.append(finger_position)


        if len(prev_finger_position) < 2:
            return 0
        

        # print(prev_finger_position.shape)

        left, right, upward, downward, upward_z, downward_z, diff = calculate_direction(prev_finger_position[-2][0], prev_finger_position[-1][0], 1, 1, 1)

        magnitude = sys.maxsize
        
        if left or right:
            print("left or right")
            magnitude = diff[0]
        
        if upward or downward:
            print("upward or downward")
            magnitude = diff[1]
        
        if upward_z or downward_z:
            print("z-up or z-down")
            magnitude = diff[2]

        return magnitude


    return f


def watch_rotation(finger):

    finger_index = hand_index[finger]
    prev_finger_position = []
    

    def f(frame_p3ds):
        nonlocal prev_finger_position  
        
        finger_position = [frame_p3ds[i] for i in finger_index]
        
        prev_finger_position.append(finger_position)


        if len(prev_finger_position) < 2:
            return 0
        
        diff = prev_finger_position[-1][1], prev_finger_position[-1][2]

        direction = -1

        if diff[0] > diff[1] and diff[0] > diff[2]:
            print("x-rotation")
            # mangitude = prev_finger_position[-1][0] - prev_finger_position[-2][0]
        elif diff[1] > diff[0] and diff[1] > diff[2]:
            print("y-rotation")
            direction = 1
        else:
            print("z-rotation")
            direction = 2

        


        
            

        # print(prev_finger_position.shape)

        

        # return magnitude


    return f





# model = Sequential()
# model.add(Masking(mask_value=-1, input_shape=(15, 63)))
#
# # LSTM layers with dropout for regularization
# model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(64))
#
# # Dense layers with dropout for regularization
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.5))
#
# # Output layer
# model.add(Dense(actions.shape[0], activation='softmax'))


# model = Sequential()
# model.add(Masking(mask_value=-1, input_shape=(15, 63)))

# # LSTM layers with dropout for regularization
# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(128))

# # Dense layers with dropout for regularization
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.5))

# # Output layer
# model.add(Dense(actions.shape[0], activation='softmax'))

model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(15, 63)))

# LSTM layers with dropout for regularization
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))

# Dense layers with dropout for regularization
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
from tensorflow.keras.optimizers.legacy import Adam

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['categorical_accuracy'])

# Compile the model
# WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
# from tensorflow.keras.optimizers import Adam

# Compile the model


# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(learning_rate=0.001),
#               metrics=['categorical_accuracy'])
model.load_weights('model/82new.h5')

#from numba import jit, cuda


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [720, 1280]

#@jit(target_backend='cuda')
def run_mp(input_stream1, input_stream2, P0, P1):
    ges = False
    gestime = []
    gesnum = 0
    action = ''
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    
    #create hand keypoints detector object.
    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)

    #containers for detected keypoints for each camera
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    testseq = []
    
    watch_index = watch_finger('index')
    
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()


        # x0, y0 = locate_color(frame0, 'blue')
        # x1, y1 = locate_color(frame1, 'blue')
        # if (x0 == -1 and x1 == -1):
        #     hands_on_mouse = True
        # else:
        #     hands_on_mouse = False

        if not ret0 or not ret1: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = hands0.process(frame0)
        results1 = hands1.process(frame1)

        #prepare list of hand keypoints of this frame
        #frame0 kpts
        frame0_keypoints = []
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame0.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame0.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame0_keypoints.append(kpts)

        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*21

        #kpts_cam0.append(frame0_keypoints)

        #frame1 kpts
        frame1_keypoints = []
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame1.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame1.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame1_keypoints.append(kpts)

        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]]*21

        #update keypoints container
        #pts_cam1.append(frame1_keypoints)


        #calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        print(watch_index(frame_p3ds))

        # if(hands_on_mouse):
            #  knuckle_pos = addArrays(addArrays(addArrays(frame_p3ds[5],frame_p3ds[9]), frame_p3ds[13]) , frame_p3ds[17])
            #  knuckle_pos1 = divArrays(addArrays(addArrays(addArrays(frame_p3ds[5],frame_p3ds[9]), frame_p3ds[13]) , frame_p3ds[17]), 4)
            #  mid_point = divArrays(addArrays(knuckle_pos, frame_p3ds[0]), 5)
            #  direction = subArrays(knuckle_pos1, frame_p3ds[0])


        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        testframe = frame_p3ds
        frame_p3ds = np.array(frame_p3ds).reshape((21, 3)) #was 21, 3
        #frame_p3ds = np.insert(frame_p3ds, 0, ges, axis=0)
        kpts_3d.append(frame_p3ds)
        testseq.append((np.array(testframe)).reshape(63))
        # Draw the hand annotations on the image.
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
          for hand_landmarks in results0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results1.multi_hand_landmarks:
          for hand_landmarks in results1.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)

        # mouse_pos = np.array([-19.41503611, -2.11918311, 69.2352099 ])
        # thumb_tip = frame_p3ds[hand_index["thumb"][0]]
        # diff  = thumb_tip - mouse_pos 



        

        
            

        #test
        if len(testseq) > 15:
            testres = model.predict(np.expand_dims(testseq[-15:], axis=0), verbose=0)[0]
            newAction = actions[np.argmax(testres)]
            
            if not (action == newAction):
                with open('/Users/decatrox/Documents/TestCommandsFIFO/commandstest.txt', 'w') as f:
                    f.write(newAction+':0.04')

                action = newAction
                print(action)
                print(testres)
            testseq.pop(0)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break #27 is ESC key.
        elif k & 0xFF == ord('q'):
            ges = True
        elif k & 0xFF == ord('w'):
            ges = False

        gesnum += 1
        if ges:
            gestime.append(gesnum)


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d), gestime




if __name__ == '__main__':

    input_stream1 = 'media/cam0_test.mp4'
    input_stream2 = 'media/cam1_test.mp4'

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    #projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    # kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)
    kpts_cam0, kpts_cam1, kpts_3d, geslines = run_mp(0, 1, P0, P1)


    #this will create keypoints file in current working folder
    #write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    #write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)

    write_keypoints_to_disk('kpts_3d.dat', kpts_3d, geslines)
