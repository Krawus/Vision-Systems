import cv2
import numpy as np
import time


def empty_callback(value):
    print("val: ", value)

def trackbarsTresh():

    modes = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
    imgGray = cv2.imread('data/prison_mike.jpeg', cv2.IMREAD_GRAYSCALE)
    
    cv2.namedWindow('Prison Mike')
    cv2.createTrackbar('thresh1', 'Prison Mike', 0, 255, empty_callback)
    cv2.createTrackbar('thresh2', 'Prison Mike', 0, 255, empty_callback)
    cv2.createTrackbar('switch mode', 'Prison Mike', 0, len(modes)-1, empty_callback)

    while True:
        thresh1 = cv2.getTrackbarPos('thresh1', 'Prison Mike')
        thresh2 = cv2.getTrackbarPos('thresh2', 'Prison Mike')
        mode = cv2.getTrackbarPos('switch mode', 'Prison Mike')

        ret, binary = cv2.threshold(imgGray, thresh1, thresh2, modes[mode])

        cv2.imshow('Prison Mike', binary)

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

    cv2.destroyAllWindows()


def scaling():
    img = cv2.imread('data/qr.jpg')

    tStart = time.perf_counter()
    imgScaled1 = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LINEAR)
    tStop = time.perf_counter()
    print("INTER_LINEAR: ", tStop-tStart)
    cv2.imshow('sINTER_LINEAR', imgScaled1)

    tStart = time.perf_counter()
    imgScaled2 = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_NEAREST)
    tStop = time.perf_counter()
    print("INTER_NEAREST: ", tStop - tStart)
    cv2.imshow('INTER_NEAREST', imgScaled2)

    tStart = time.perf_counter()
    imgScaled3 = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_AREA)
    tStop = time.perf_counter()
    print("INTER_AREA: ", tStop - tStart)
    cv2.imshow('INTER_AREA', imgScaled3)

    tStart = time.perf_counter()
    imgScaled4 = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LANCZOS4)
    tStop = time.perf_counter()
    print("INTER_LANCZOS4: ", tStop-tStart)
    cv2.imshow('INTER_LANCZOS4', imgScaled4)

    cv2.waitKey(0)


def blend():
    cv2.namedWindow("blending")

    putLogo = cv2.imread('data/PUTVISION_LOGO.png')
    mike = cv2.imread('data/prison_mike.jpeg')

    putLogo = cv2.resize(putLogo, dsize=(np.shape(mike)[1], np.shape(mike)[0]))
    

    cv2.createTrackbar('alpha', 'blending', 0, 100, empty_callback)
    cv2.createTrackbar('beta', 'blending', 0 , 100, empty_callback)
    cv2.createTrackbar('gamma', 'blending', 0 , 100, empty_callback)

    while True:
        alpha = cv2.getTrackbarPos('alpha', 'blending')/100
        beta = cv2.getTrackbarPos('beta', 'blending')/100
        gamma = cv2.getTrackbarPos('gamma', 'blending')/100

        blended = cv2.addWeighted(mike, alpha, putLogo, beta, gamma)

        cv2.imshow('blending', blended)

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

    cv2.destroyAllWindows()


def negative():
    img = cv2.imread('data/prison_mike.jpeg')

    negative = 255 - img

    cv2.imshow('wow', negative)
    cv2.waitKey(0)



if __name__ == '__main__':
    # trackbarsTresh()
    # scaling()
    # blend()
    negative()