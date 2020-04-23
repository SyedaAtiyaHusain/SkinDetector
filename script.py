def detector():
    import imutils
    import numpy as np
    import argparse
    import cv2
    from google.colab.patches import cv2_imshow
    frame = cv2.imread('28_x2xqwb.jpg')
    lower = np.array([0, 0, 0], dtype = "uint8")
    upper = np.array([20, 255,255 ], dtype = "uint8")

    frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussitanBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    # show the skin in the image along with the mask
    cv2_imshow( np.hstack([frame, skin]))
