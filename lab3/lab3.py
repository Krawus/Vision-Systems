import cv2
import numpy as np
import time


def emptyCallback():
    pass

def filters():

    cv2.namedWindow('thresholds')
    cv2.createTrackbar('N', 'thresholds', 0, 10, emptyCallback)

    imgNoise = cv2.imread('data/lenna_noise.bmp')
    imgSaltPepper = cv2.imread('data/lenna_salt_and_pepper.bmp')

    while True:
        n = cv2.getTrackbarPos('N', 'thresholds') * 2 + 1

        noiseBlur = cv2.blur(imgNoise, (n, n))
        noiseMedian = cv2.medianBlur(imgNoise, n)
        noiseGauss = cv2.GaussianBlur(imgNoise, (n, n), 0)

        SaltPepperBlur = cv2.blur(imgSaltPepper, (n, n))
        SaltPepperMedian = cv2.medianBlur(imgSaltPepper, n)
        SaltPepperGauss = cv2.GaussianBlur(imgSaltPepper, (n, n), 0)

        cv2.imshow('noiseBlur', noiseBlur)
        cv2.imshow('noiseMedian', noiseMedian)
        cv2.imshow('noiseGauss', noiseGauss)

        cv2.imshow('SaltPepperBlur', noiseBlur)
        cv2.imshow('SaltPepperMedian', noiseMedian)
        cv2.imshow('SaltPepperGauss', noiseGauss)

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break


def morphCallback(val):
    pass




def morphology():

    cv2.namedWindow('thresholds')
    cv2.createTrackbar('N', 'thresholds', 1, 10, emptyCallback)

    img = cv2.imread('data/lenna_salt_and_pepper.bmp', cv2.IMREAD_GRAYSCALE)
    ret, imgBinary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    
    kernel = np.ones((1,1),np.uint8)

    while True:
        n = cv2.getTrackbarPos('N', 'thresholds')
        kernel = np.ones((n,n),np.uint8)

        erosion = cv2.erode(imgBinary,kernel,iterations = 1)
        dilation = cv2.dilate(imgBinary,kernel,iterations = 1)
        opening = cv2.morphologyEx(imgBinary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(imgBinary, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("dilation", dilation)
        cv2.imshow('erosion', erosion)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break


def scanning():
    imgDefault = cv2.imread('data/prison_mike.jpeg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('data/prison_mike.jpeg', cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape[:2]
    
    for x in range(0, cols, 3):
        for y in range(0, rows):
            img[y, x] = 255
    
    cv2.imshow('white_pixels', img)

    img = cv2.imread('data/prison_mike.jpeg', 0)

    tStart = time.perf_counter()
    imgCv2blur = cv2.blur(img, (3, 3))
    tStop = time.perf_counter()
    print("cv2blur: ", tStop-tStart)

    tStart = time.perf_counter()
    rows, cols = img.shape
    #CUSTOM BLUR
    for x in range (1, rows-1):
        for y in range(1, cols-1):
            avg = (float(img[x-1, y-1]) + float(img[x-1, y]) + float(img[x-1, y+1]) + float(img[x, y-1]) + float(img[x, y]) +
                   float(img[x, y+1]) + float(img[x+1, y-1]) + float(img[x+1, y]) + float(img[x+1, y+1]))/9
            img[x, y] = avg
    tStop = time.perf_counter()
    print("custom: ", tStop-tStart)

    # kernel = np.ones((3,3),np.uint8)
    # filter2d = cv2.filter2D(img, ddepth=1 , kernel=kernel)


    cv2.imshow('default', imgDefault)
    cv2.imshow("customBlur", img)
    cv2.imshow("cv2.blur", imgCv2blur)
    # cv2.imshow("filter2d", filter2d)

    cv2.waitKey(0)


def kuwahara_apply(image: np.ndarray, window_size: int) -> np.ndarray:

    result = np.zeros_like(image)
    padding = window_size // 2
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for y in range(image.shape[0] - window_size+1):
        for x in range(image.shape[1] - window_size+1):
            window = image[y:y + window_size, x:x+window_size]
            region_1 = window[0:(window_size//2+1), 0:(window_size//2+1)]
            region_2 = window[0: window_size//2+1, window_size//2: window_size]
            region_3 = window[window_size//2: window_size, window_size//2: window_size]
            region_4 = window[window_size//2: window_size, 0:window_size//2 + 1]

            best_mean, best_std = cv2.meanStdDev(region_1)

            for region in (region_2, region_3, region_4):
                mean, std = cv2.meanStdDev(region)

                if std < best_std:
                    best_std = std
                    best_mean = mean

            result[y + window_size//2 - padding, x + window_size//2 - padding] = best_mean

    return result



def kuwahara():
    image = cv2.imread("data/prison_mike.jpeg", 0)
    cv2.imshow("pzdr", image)
    result = kuwahara_apply(image, 7)
    cv2.imshow("result", result)
    cv2.waitKey()




if __name__ == '__main__':
    # filters()
    # morphology()
    # scanning()
    kuwahara()
