import cv2
import numpy as np

def ex1_cont():
    def track_cal(val):
            pass

    cv2.namedWindow("background_image")
    cv2.namedWindow("current_image")
    cv2.namedWindow("foreground_image")

    cv2.createTrackbar('track', 'foreground_image', 0, 255, track_cal)

    # cap = cv2.VideoCapture(0
    cap = cv2.VideoCapture(0)  # open the default camera
    img_back = []
    img_curr = []

    key = ord(' ')
    while key != ord('q'):
            ret, frame = cap.read()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if key == ord('a'):
                    img_back = img_gray
                    cv2.imshow('background_image', img_back)
            
            if len(img_back) > 0:
            # if key == ord('x'):
                img_curr = img_gray
                cv2.imshow('current_image', img_curr)

                if (len(img_back) > 0) and (len(img_curr)>0):
                        tresh_val = cv2.getTrackbarPos('track', 'foreground_image')
                        
                        diff = cv2.absdiff(img_curr, img_back)
                        ret, thresh = cv2.threshold(diff, tresh_val ,255, cv2.THRESH_BINARY)

                        cv2.imshow('foreground_image', thresh)


            # Display the result of our processing
            img_back = img_gray
            cv2.imshow('camera', img_gray)
            
            
            # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
            key = cv2.waitKey(30)

        # When everything done, release the capture
    cap.release()
        # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()

def ex2():
    def track_cal(val):
            pass

    cv2.namedWindow("background_image")
    cv2.namedWindow("current_image")
    cv2.namedWindow("foreground_image")

    cv2.createTrackbar('track', 'foreground_image', 0, 255, track_cal)

    # cap = cv2.VideoCapture(0
    cap = cv2.VideoCapture(0)  # open the default camera
    img_back = []
    ret, img_curr = cap.read()
    kernel = np.ones((3,3),np.uint8)

    key = ord(' ')
    while key != ord('q'):
            ret, frame = cap.read()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_curr = img_gray

            if len(img_back) == 0:
                  img_back = img_curr
            
            tresh_val = cv2.getTrackbarPos('track', 'foreground_image')
            
            if (img_back != []):
                diff = cv2.absdiff(img_curr, img_back)
                ret, thresh = cv2.threshold(diff, tresh_val ,255, cv2.THRESH_BINARY)
                closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                cv2.imshow('foreground_image', thresh)
                cv2.imshow('background_image', img_back)
                cv2.imshow('current_image', img_curr)
  

            # Display the result of our processing
            cv2.imshow('camera', img_gray)
            
            img_back = img_gray
            
            
            # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
            key = cv2.waitKey(30)

        # When everything done, release the capture
    cap.release()
        # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()

       
def ex2():
    def track_cal(val):
        pass

    cv2.namedWindow("background_image")
    cv2.namedWindow("current_image")
    cv2.namedWindow("foreground_image")

    cv2.createTrackbar('track', 'foreground_image', 0, 255, track_cal)

    # cap = cv2.VideoCapture(0
    cap = cv2.VideoCapture(0)  # open the default camera
    img_back = []
    ret, img_curr = cap.read()

    kernel = np.ones((3,3),np.uint8)

    key = ord(' ')

    while key != ord('q'):
            ret, frame = cap.read()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_curr = img_gray

            if len(img_back) == 0:
                  img_back = img_curr

            tresh_val = cv2.getTrackbarPos('track', 'foreground_image')
            
            B1 = img_back < img_curr
            img_back[B1 == True] = img_back[B1 == True] + 1
            B2 = img_back > img_curr
            img_back[B2 == True] = img_back[B2 == True] - 1

            
            diff = cv2.absdiff(img_curr, img_back)
            ret, thresh = cv2.threshold(diff, tresh_val ,255, cv2.THRESH_BINARY)
            closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cv2.imshow('foreground_image', closed_img)


            # Display the result of our processing
            cv2.imshow('camera', img_gray)
            cv2.imshow('background_image', img_back)

            key = cv2.waitKey(10)

        # When everything done, release the capture
    cap.release()
        # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()
      

def ex3():
    backSub = cv2.createBackgroundSubtractorKNN()

    cap = cv2.VideoCapture(0) 

    key = ord(' ')
      
    while key != ord('q'):
        ret, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgMask = backSub.apply(frame)


        cv2.imshow('Frame', img_gray)
        cv2.imshow('FG Mask', fgMask)

        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()


def ex5_6():
    def track_cal(val):
            pass

    cv2.namedWindow("background_image")
    cv2.namedWindow("current_image")
    cv2.namedWindow("foreground_image")
    
    kernel = np.ones((7, 7),np.uint8)

    cv2.createTrackbar('track', 'foreground_image', 0, 255, track_cal)

    # cap = cv2.VideoCapture(0
    cap = cv2.VideoCapture(0)  # open the default camera
    img_back = []
    img_curr = []

    key = ord(' ')
    while key != ord('q'):
            ret, frame = cap.read()
            img_gray = frame.copy()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_curr = img_gray

            if len(img_back) == 0:
                  img_back = img_curr
            

            if (len(img_back)>0) and (len(img_curr)>0):
                    tresh_val = cv2.getTrackbarPos('track', 'foreground_image')
                        
                    diff = cv2.absdiff(img_curr, img_back)
                    ret, thresh = cv2.threshold(diff,80,255, cv2.THRESH_BINARY)
                    closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                    if np.sum(closed_img == 255) > 100:
                          print("motion!!!")

                          whites_y = np.where(closed_img == 255)[:][0]
                          whites_x = np.where((closed_img == 255))[:][1]
                          

                          
                        #   box = cv2.minAreaRect(closed_img)
                          cv2.rectangle(frame, (np.min(whites_x), np.min(whites_y)), (np.max(whites_x), np.max(whites_y)), (0, 0, 255), 3)
                          
                    cv2.imshow('foreground_image', closed_img)
                    cv2.imshow('colour', frame)


            # Display the result of our processing
            img_back = img_gray

            
            cv2.imshow('camera', img_gray)
            
            
            # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
            key = cv2.waitKey(10)

        # When everything done, release the capture
    cap.release()
        # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()



def hw1():
    def track_cal(val):
        pass

    cv2.namedWindow('diff')
    cv2.createTrackbar('track', 'diff', 0, 255, track_cal)

    cap = cv2.VideoCapture(0) 

    previous_frame= []
    current_frame = []
    next_frame = []
    counter = 0
    kernel = np.ones((3,3),np.uint8)
    deltaI = []

    key = ord(' ')
    while key != ord('q'):
            ret, frame = cap.read()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (key == ord('x')):
                if (counter > 2):
                      counter = 0
                elif (counter == 0):
                      previous_frame = img_gray
                      counter += 1
                      cv2.imshow('prevoious', previous_frame)
                elif (counter == 1):
                      current_frame = img_gray
                      counter += 1
                      cv2.imshow('current', current_frame)
                elif (counter == 2):
                      next_frame = img_gray
                      counter += 1
                      cv2.imshow('next', next_frame)
                      deltaI1 = cv2.absdiff(next_frame, current_frame)
                      deltaI2 = cv2.absdiff(next_frame, previous_frame)
                      deltaI = cv2.bitwise_and(deltaI1, deltaI2)
                      cv2.imshow('deltaI', deltaI)


            if (len(deltaI) > 0):
                tresh_val = cv2.getTrackbarPos('track', 'diff')
                # print(tresh_val)
                ret, thresh = cv2.threshold(deltaI, tresh_val, 255, cv2.THRESH_BINARY)
                closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                cv2.imshow('diff', closed_img)
                      
                      
            cv2.imshow('camera', img_gray)
            
            key = cv2.waitKey(10)


    cap.release()

    cv2.destroyAllWindows()
      

def hw1_cont():
    def track_cal(val):
        pass

    cv2.namedWindow('diff')
    cv2.createTrackbar('track', 'diff', 40, 255, track_cal)

    cap = cv2.VideoCapture(0) 

    previous_frame= []
    current_frame = []
    next_frame = []
    counter = 0
    kernel = np.ones((5,5),np.uint8)
    deltaI = []

    key = ord(' ')
    while key != ord('q'):
            ret, frame = cap.read()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (previous_frame == [] and current_frame == [] and next_frame == []):
                  previous_frame = img_gray
                  current_frame = img_gray
                  next_frame = img_gray
            
            next_frame = img_gray

            deltaI1 = cv2.absdiff(next_frame, current_frame)
            deltaI2 = cv2.absdiff(next_frame, previous_frame)
            deltaI = cv2.bitwise_and(deltaI1, deltaI2)
            cv2.imshow('delta_I', deltaI)
            

            previous_frame = current_frame
            current_frame = next_frame


            if (len(deltaI) > 0):
                tresh_val = cv2.getTrackbarPos('track', 'diff')
                ret, thresh = cv2.threshold(deltaI, tresh_val, 255, cv2.THRESH_BINARY)
                closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                cv2.imshow('diff', closed_img)
                      
                      
            cv2.imshow('camera', img_gray)
            
            key = cv2.waitKey(10)


    cap.release()

    cv2.destroyAllWindows()
# ex1_cont()
# ex2()
# ex3()
# ex5_6()
# hw1()
hw1_cont()

