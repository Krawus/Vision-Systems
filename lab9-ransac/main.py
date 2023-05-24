import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    def chooseCorners(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(params) < 2:
                params.append([x, y])

            if len(params) >= 2:
                imgROI = img[params[0][1]:params[1][1], params[0][0]:params[1][0]]

                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(imgROI,None)
                img_sift = cv2.drawKeypoints(imgROI, kp1, None, color=(0, 0, 255))
                # cv2.imshow("hello", img_sift)

                img2 = cv2.imread("data/img2.jpg")
                img2 = cv2.resize(img2, (0, 0), fx=0.2, fy=0.2)

                kp2, des2 = sift.detectAndCompute(img2,None)

                bf = cv2.BFMatcher()
                matches = bf.match(des1,des2)

                matches = sorted(matches, key = lambda x:x.distance)
                imgROIRGB = cv2.cvtColor(imgROI, cv2.COLOR_BGR2RGB)
                img2RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                matchesImg = cv2.drawMatches(imgROIRGB,kp1,img2RGB,kp2,matches[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                plt.imshow(matchesImg),plt.show()
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                h,w,c = imgROI.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                imgEnd = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                cv2.imshow("result", img2)

                # plt.imshow(img3),plt.show()

    cv2.namedWindow("choose object")
    img = cv2.imread('data/img1.jpg')
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

    width = img.shape[1]
    height = img.shape[0]



    corners = []
    key = ord(" ")

    cv2.setMouseCallback("choose object", chooseCorners, corners)

    while key != ord('x'):

        cv2.imshow("choose object", img)

        key = cv2.waitKey(10)