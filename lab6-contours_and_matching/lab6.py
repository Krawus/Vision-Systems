import cv2
import numpy as np




def task1():

    def track_cal(val):
        pass

    img = cv2.imread('data/not_bad.jpg')
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    imgOrg = img.copy()
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('ayy', img)
    # cv2.waitKey(0)

    # cv2.namedWindow('trackwin')
    # cv2.createTrackbar('tresh', 'trackwin', 0, 255, track_cal)

    # key = ord('a')

    # while key != ord('x'):
    #     tresh = cv2.getTrackbarPos('tresh', 'trackwin')
    #     ret,th1 = cv2.threshold(img_g ,tresh,255,cv2.THRESH_BINARY)
    #     cv2.imshow('treshwin', th1)

    #     key = cv2.waitKey(10)


    ret, onlySquares = cv2.threshold(imgG, 53, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5),np.uint8)

    onlySquares = cv2.erode(onlySquares ,kernel,iterations = 1)

    # cv2.imshow('aaa', onlySquares)
    # cv2.waitKey(5000)

    contours, hierarchy = cv2.findContours(onlySquares, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0,255,0), 3)


    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]

    pointsVec = []

    for cnt in range(len(contours)):
        M = cv2.moments(contours[cnt])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(img, (cx, cy), 5, colors[cnt], 3)
        pointsVec.append([cx, cy])


    # cv2.imshow('aaa', img)
    # cv2.waitKey(0)

    pointsVec = np.float32(pointsVec)
    y_size, x_size, _ = img.shape

    ptsSource = np.float32([[x_size, y_size], [0, y_size], [x_size, 0], [0, 0]])

    Transform = cv2.getPerspectiveTransform(pointsVec, ptsSource)

    dst = cv2.warpPerspective(imgOrg, Transform, (x_size, y_size))
    cv2.imshow('aaa', dst)
    cv2.waitKey(0)



def task2():
    img = cv2.imread('data/britney.jpg')

    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tmpl = imgG[138:180, 312:417]
    w, h = tmpl.shape[::-1]
    res = cv2.matchTemplate(imgG, tmpl, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img ,top_left, bottom_right, 255, 2)

    cv2.imshow('aaa', img)
    cv2.waitKey(0)


task2()



