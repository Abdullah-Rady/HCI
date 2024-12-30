import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)



def locate_color(imageFrame, color):
    # Convert the imageFrame from BGR(RGB color space) to HSV(hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    color_ranges = {
    'red': ([165, 100, 100], [185, 255, 255]),
    'green': ([70, 100, 100], [85, 255, 255]),
    'blue': ([105, 100, 100], [120, 255, 255]),
    'yellow': ([35, 100, 100], [43, 255, 255]),
    }


    color_mask = cv2.inRange(hsvFrame, np.array(color_ranges[color][0]), np.array(color_ranges[color][1]))

    kernal = np.ones((5, 5), "uint8")

    # For red color
    color_mask = cv2.dilate(color_mask, kernal)
    # res_color = cv2.bitwise_and(imageFrame, imageFrame,
    #                             mask=color_mask)


    contours, _ = cv2.findContours(color_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            return x, y

    return -1, -1
    
    #         imageFrame = cv2.rectangle(imageFrame, (x, y),
    #                                     (x + w, y + h),
    #                                     (0, 0, 255), 2)

    #         cv2.putText(imageFrame, "Red Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    #                     (0, 0, 255))
    #         print(x, y, w, h)


    # cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     break