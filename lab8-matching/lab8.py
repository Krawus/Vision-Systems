import cv2
import matplotlib.pyplot as plt

def detection():
    
    img = cv2.imread('data_det_desc/forward-1.bmp', 0)
    
    
    fast = cv2.FastFeatureDetector_create()
    img_cp = img.copy()
    detection_fast = fast.detect(img, None)
    img_fast = cv2.drawKeypoints(img_cp, detection_fast, None, color=(255,0,0))
    print("fast features: {}".format(len(detection_fast)))

    cv2.imshow('FAST', img_fast)
    # cv2.waitKey(0)

    img_cp = img.copy()
    orb = cv2.ORB_create()
    detection_orb = orb.detect(img, None)
    img_orb = cv2.drawKeypoints(img_cp, detection_orb, None, color=(0, 255, 0))
    print("ORB features: {}".format(len(detection_orb)))
    cv2.imshow("ORB", img_orb)
    # cv2.waitKey(0)

    img_cp = img.copy()
    sift = cv2.SIFT_create()
    detection_sift = sift.detect(img, None)
    img_sift = cv2.drawKeypoints(img_cp, detection_sift, None, color=(0, 0, 255))
    print("SIFT features: {}".format(len(detection_sift)))
    cv2.imshow("SIFT", img_sift)

    cv2.waitKey(0)

    print(detection)


def description_and_match():
    img = cv2.imread('data_det_desc/forward-1.bmp', 0)
    img2 = cv2.imread('data_det_desc/rotate-5.bmp', 0)
    

    ########## ORB ###########

    orb = cv2.ORB_create()

    # kp_orb1 = orb.detect(img, None)
    # kp_orb2 = orb.detect(img2, None)   
    # orb1_kp_des, orb1_des = orb.compute(img, kp_orb1)
    # orb2_kp_des, orb2_des = orb.compute(img2, kp_orb2)
 
    orb1_kp, orb1_des = orb.detectAndCompute(img, None)
    orb2_kp, orb2_des = orb.detectAndCompute(img2, None)


    ######### SIFT ###########
    sift = cv2.SIFT_create()

    sift1_kp, sift1_des = sift.detectAndCompute(img, None)
    sift2_kp, sift2_des = sift.detectAndCompute(img2, None)
    

    matcher_orb = cv2.DescriptorMatcher_create(
	cv2.DescriptorMatcher_BRUTEFORCE_HAMMING
    )

    matcher_sift = cv2.DescriptorMatcher_create(
	cv2.DescriptorMatcher_BRUTEFORCE_L1
    )

    # matcher = cv2.BFMatcher(cv2.NORM_L1)

    # Match descriptors.
    # 
    matches_orb = matcher_orb.match(orb1_des, orb2_des)
    matches_orb = sorted(matches_orb, key = lambda x:x.distance)

    matches_sift = matcher_sift.knnMatch(sift1_des, sift2_des, k=2)
    # Sort them in the order of their distance.
    
    good = []
    for m,n in matches_sift:
        if m.distance <  0.7*n.distance:
            good.append([m])
    

    # Draw first 10 matches.
    orb_plot = cv2.drawMatches(img, sift1_kp,img2,sift2_kp, matches_orb[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    sift_plot = cv2.drawMatchesKnn(img, sift1_kp,img2,sift2_kp, good[0:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(orb_plot)
    plt.title("ORB")
    plt.figure()
    plt.imshow(sift_plot)
    plt.title("SIFT")
    plt.show()


    # print(sift_des)




# def matching():


# detection()
description_and_match()