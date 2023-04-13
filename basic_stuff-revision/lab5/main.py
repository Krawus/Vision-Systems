import cv2
import numpy as np


def previtt_and_sobel():
    img = cv2.imread('data/britney.jpg', cv2.IMREAD_GRAYSCALE)
    img_float = np.float32(img)
    kernelx_previtt = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]], np.int8)
    
    kernely_previtt = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]], np.int8)
    
    filtered_previtt_x = cv2.filter2D(img_float, ddepth=cv2.CV_32F, kernel=kernelx_previtt)/3
    filtered_previtt_y = cv2.filter2D(img_float, ddepth=cv2.CV_32F, kernel=kernely_previtt)/3
    # result = np.linalg.norm(filtered_previtt)
    
    

    cv2.imshow('filteredx', abs(filtered_previtt_x).astype(np.uint8))
    cv2.imshow('filteredy', abs(filtered_previtt_y).astype(np.uint8))

    img_gradient = cv2.sqrt(cv2.pow(filtered_previtt_x, 2) + cv2.pow(filtered_previtt_y, 2))
    cv2.imshow('both', img_gradient.astype(np.uint8))

    cv2.waitKey(0)

def canny():
    def track_cal1(val):
        tresh1 = val
    def track_cal2(val):
        tresh2 = val
    img = cv2.imread('data/drone_ship.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('canny')
    cv2.createTrackbar('tresh1', 'canny', 0, 255, track_cal1)
    cv2.createTrackbar('tresh2', 'canny', 0, 255, track_cal2)

    key = ord('a')
    while key != ord('x'):
        tresh1 = cv2.getTrackbarPos('tresh1', 'canny')
        tresh2 = cv2.getTrackbarPos('tresh2', 'canny')

        edges = cv2.Canny(img, tresh1, tresh2)

        cv2.imshow('canny', edges)
        cv2.imshow('original', img)
        
        

        key = cv2.waitKey(10)
        
def houghLine():
    img = cv2.imread('data/shapes.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    edges = cv2.Canny(img, 26, 116)
    lines = cv2.HoughLines(edges,1,np.pi/180,80)

    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('lines',img)

    img = cv2.imread('data/shapes.jpg')
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,26,116,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,60,minLineLength=7,maxLineGap=5)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('houghP.jpg',img)
    cv2.waitKey(0)

def houghCir():
    img = cv2.imread('data/shapes.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=116,param2=40,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    print(circles)

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ex1():

    def blah(val):
        pass
    img = cv2.imread('data/drone_ship.jpg', cv2.IMREAD_GRAYSCALE)
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    kernel = np.ones((1,1),np.uint8)
    edges = cv2.Canny(img, 255, 255, apertureSize=3)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    cv2.imshow('canned', opening)
    cv2.namedWindow('ship')
    cv2.createTrackbar('tresh1', 'ship', 0, 255, blah)
    cv2.createTrackbar('tresh2', 'ship', 0, 255, blah)
    cv2.createTrackbar('tresh3', 'ship', 0, 255, blah)

    lines = cv2.HoughLinesP(opening,1,np.pi/250,82,minLineLength=50,maxLineGap=4)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img_c,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('ship',img_c)
    cv2.waitKey(0)

def ex2():
    def empty_callback(val):
        pass

    img_c = cv2.imread('data/fruit.jpg')
    img_c_original = cv2.imread('data/fruit.jpg')
    img_g = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    # cv2.namedWindow('tresholding')
    # cv2.createTrackbar('h_l', 'tresholding', 24, 255, empty_callback)
    # cv2.createTrackbar('h_u', 'tresholding', 79, 255, empty_callback)

    # cv2.createTrackbar('s_l', 'tresholding', 63, 255, empty_callback)
    # cv2.createTrackbar('s_u', 'tresholding', 255, 255, empty_callback)
    
    # cv2.createTrackbar('v_l', 'tresholding', 0, 255, empty_callback)
    # cv2.createTrackbar('v_u', 'tresholding', 255, 255, empty_callback)
    # ret, tresh1 = cv2.threshold(img_g,50,255,cv2.THRESH_BINARY)


    #236 to 255 - all fruits


    #apples: h(24 : 79), s(63 : 255), v(0 : 255)
    #oranges: h(7 : 22), s(65 : 255), v(130 : 255)
    frame_HSV = cv2.cvtColor(img_c, cv2.COLOR_BGR2HSV)
    apples_tresh = cv2.inRange(frame_HSV, (24, 63, 0), (79, 255, 255))
    apples_img = cv2.bitwise_and(img_g, img_g, mask = apples_tresh)
    apples_img = cv2.medianBlur(apples_img,9)
    cv2.imshow('aa', apples_img)

    # apples_canny = cv2.Canny(apples_img, 100, 200)
    # cv2.imshow('canny_app', apples_canny)

    circles = cv2.HoughCircles(apples_img,cv2.HOUGH_GRADIENT,1,30,
                            param1=200,param2=28,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        cv2.circle(img_c_original,(i[0],i[1]),i[2],(0,255,0),2)
    
    #----------oranges
    # frame_HSV = cv2.cvtColor(img_c, cv2.COLOR_BGR2HSV)
    oranges_tresh = cv2.inRange(frame_HSV, (7, 65, 130), (22, 255, 255))
    oranges_img = cv2.bitwise_and(img_g, img_g, mask = oranges_tresh)
    oranges_img = cv2.medianBlur(oranges_img,9)
    cv2.imshow('oo', oranges_tresh)

    # oranges_canny = cv2.Canny(oranges_img, 100, 200)
    # cv2.imshow('canny', oranges_canny)

    circles = cv2.HoughCircles(oranges_img,cv2.HOUGH_GRADIENT,1,20,
                            param1=200,param2=25,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        cv2.circle(img_c_original,(i[0],i[1]),i[2],(0,0,255),2)
    
    cv2.imshow('detected circles',img_c_original)
    cv2.waitKey(0)
    # key = ord('a')
    # while key != ord('x'):
    #     low_H = cv2.getTrackbarPos('h_l', 'tresholding')
    #     low_S = cv2.getTrackbarPos('s_l', 'tresholding')
    #     low_V = cv2.getTrackbarPos('v_l', 'tresholding')
    #     high_H = cv2.getTrackbarPos('h_u', 'tresholding')
    #     high_S = cv2.getTrackbarPos('s_u', 'tresholding')
    #     high_V = cv2.getTrackbarPos('v_u', 'tresholding')

    #     frame_HSV = cv2.cvtColor(img_c, cv2.COLOR_BGR2HSV)
    #     frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    #     # ret, tresh1 = cv2.threshold(img_c[:, :, 2], lo_tresh, up_tresh, cv2.THRESH_BINARY)
    #     img_sum = cv2.bitwise_and(img_c, img_c, mask=frame_threshold)
    #     cv2.imshow('tresholding', frame_threshold)

    #     # cv2.imshow('sum', img_sum)

    #     key = cv2.waitKey(10)


def ex3():
    img_c = cv2.imread('data/coins.jpg')
    img_g = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    
    
    img_g = cv2.medianBlur(img_g, 15)
    canny_coins = cv2.Canny(img_g, 180/2, 180)
    cv2.imshow('canny', canny_coins)

    cv2.imshow("ggg", img_g)

    circles = cv2.HoughCircles(img_g,cv2.HOUGH_GRADIENT,1,20,
                            param1=180,param2=53,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))

    small_coins = 0
    big_coins = 0

    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(img_c,(i[0],i[1]),i[2],(0,255,0),2)
        
        if i[2] < 100:
            small_coins += 1
        else:
            big_coins += 1
        


    cv2.imshow('detected circles',img_c)

    print(f'small coins amount: {small_coins:.2f}, big coins amount: {big_coins:.2f}')
    cv2.waitKey(0)
        

    

if __name__ == '__main__':
    # previtt_and_sobel()
    # canny()
    # houghLine()
    # houghCir()
    ex1()
    # ex2()
    # ex3()
    