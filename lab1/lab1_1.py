import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def toROI():
    img = cv2.imread('data/cat_pepege.png')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('aaa', img)
    print(np.shape(img))

    print("Pixel val at [220, 270] (BGR): ", img[220, 270])
    print("Pixel val at [220, 270] (GrayScale): ", imgGray[220, 270])

    eye=img[150:196, 277:340]
    cv2.imshow("eye", eye)

    tripleEyed = img
    tripleEyed[115:115 + eye.shape[0] , 217:217+eye.shape[1]] = eye
    cv2.imshow("wut", tripleEyed)
    cv2.imshow("openCV", img)
    plt.imshow(img)
    plt.show()
    cv2.waitKey(0)
    

def split():
    img = cv2.imread('data/AdditiveColor.png')
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    cv2.imshow('blue', blue)
    cv2.imshow('green', green)
    cv2.imshow('red', red)
    cv2.waitKey(0)

def camera():
    cap = cv2.VideoCapture(0)

    key = ord('a')
    
    ret, frame = cap.read()
    frameToShow = frame
    while key != ord('w'):
        # Capture frame-by-frame
        ret, frame = cap.read()
        

        if key == ord(' '):
            frameToShow = frame
        
        cv2.imshow('camera', frameToShow)
        key = cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()

def slideshow():
    directory = 'data'
    images = []
    files = 0
    for filename in os.scandir(directory):
        if filename.is_file():
            print(filename.path)
            files+=1
            images.append(cv2.imread(filename.path))
    
    key = ord('a')
    index = 0
    while key != ord('x'):
        
        if key == ord('d'):
            index+=1
        if key == ord('a'):
            index-=1
        
        if index > files-1:
            index = 0
        if index < 0:
            index = files-1

        cv2.imshow("slideshow", images[index])
        key = cv2.waitKey(30)
    
    
            

    

if __name__ == '__main__':
    # toROI()
    # split()
    # camera()
    slideshow()