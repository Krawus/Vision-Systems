import cv2
import numpy as np
import matplotlib.pylab as plt

def drawing_circ_rect():
    def draw_geo(event, x, y, flags, param):
        width = 50
        height = 20
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(img, (int(x-width/2), int(y-height/2)), (int(x+width/2), int(y+height/2)), (0 ,230, 150) )
        elif event == cv2.EVENT_MBUTTONDOWN:
            cv2.circle(img, (x, y), 30, (255, 0, 0))

    img = cv2.imread('data/prison_mike.jpeg')
    cv2.namedWindow('prison_mike')
    cv2.setMouseCallback('prison_mike', draw_geo, img)
    
    key = ord('a')

    while key != ord('x'):
         cv2.imshow('prison_mike',img)
         cv2.waitKey(10)


def road_ex():
    def choose_points(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(params) < 4:
                params.append([x, y])

            if len(params) >= 4:
                params = np.float32(params)
                M = cv2.getPerspectiveTransform(params, np.float32([[0, 560], [0, 0], [896, 0], [896, 560]]))
                print(road.shape)
                dst = cv2.warpPerspective(road, M, (870, 550))
                cv2.imshow('hello', dst)

    
    road = cv2.imread('data/road.jpg')
    road = cv2.resize(road, None, fx = 0.35, fy= 0.35, interpolation = cv2.INTER_AREA)
    cv2.namedWindow('road_image')
    points = []
    cv2.setMouseCallback('road_image', choose_points, points)

    key = ord('a')

    while key != ord('x'):
         cv2.imshow('road_image',road)
        
         key = cv2.waitKey(10)


def histograms():
    img = cv2.imread('data/prison_mike.jpeg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist_gray = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    # hist = cv2.calcHist([img],[0,1,2],None,[256],[0,256])

    plt.hist(img_gray.ravel(),256,[0,256])

    equ = cv2.equalizeHist(img_gray)
    res = np.hstack((img_gray,equ)) #stacking images side-by-side
    cv2.imshow('res.png',res)

    plt.hist(equ.ravel(), 256, [0, 256])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()

def home_ex1():
    def choose_points(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(params) < 2:
                params.append([x, y])

        if len(params) >=2:
            ret, thresh1 = cv2.threshold(img[params[0][1]:params[1][1], params[0][0]:params[1][0], 1], 155, 255, cv2.THRESH_BINARY)
            img[params[0][1]:params[1][1], params[0][0]:params[1][0], 1] = thresh1




    img = cv2.imread('data/prison_mike.jpeg')
    cv2.namedWindow('mike')
    points = []
    cv2.setMouseCallback('mike', choose_points, points)

    key = ord('a')

    while key != ord('x'):
         cv2.imshow('mike', img)
        
         key = cv2.waitKey(10)


img_gallery = None
def art():
    global img_gallery
    def gallery_callback(event, x, y, flags, params):
        global img_gallery

        points = params
        

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))

            if len(points) >= 4:
                points = np.float32(points)
                pug_mask = np.ones_like(img_pug)*255
                rows, cols, bin= pug_mask.shape
                #LU, RU, RD, LD
                pts2 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
                transform = cv2.getPerspectiveTransform(pts2, points)
                dst_mask = cv2.warpPerspective(pug_mask, transform, (img_gallery.shape[1], img_gallery.shape[0]))
                dst_pug = cv2.warpPerspective(img_pug, transform, (img_gallery.shape[1], img_gallery.shape[0]))
                
                dst_mask_inv = cv2.bitwise_not(dst_mask)

                gallery_bg = cv2.bitwise_and(dst_mask_inv, img_gallery)
                # cv2.imshow('bla', gallery_bg)

                img_gallery = cv2.add(gallery_bg, dst_pug)
                
                
    img_pug = cv2.imread('data/pug.png')
    img_gallery = cv2.imread('data/gallery.png')
    pts1 = []

    cv2.namedWindow('gallery')
    cv2.setMouseCallback('gallery', gallery_callback, param = pts1)

    key = ord('a')
    while key != ord('x'):
         
         cv2.imshow('gallery', img_gallery)

        
         key = cv2.waitKey(10)



    
        

if __name__ == '__main__':
    # drawing_circ_rect()
    # road_ex()
    # histograms()
    # home_ex1()
    art()

    