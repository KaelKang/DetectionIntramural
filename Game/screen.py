'''
@Description : Get the cropped screen
@Author : Kael
@Time : 2023/1/16
'''
import numpy as np
import cv2 as cv



class screen_cap:
    def __init__(self, frame):
        # self.img = cv.resize(frame, (901, 901), interpolation=cv.INTER_AREA)
        self.img = frame
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    
    def screen(self):
        #### devide the ranges of blue, green, red
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        rmask1 = cv.inRange(hsv, np.array([156, 60, 60]),  np.array([180, 255, 255]))
        rmask2 = cv.inRange(hsv, np.array([0, 60, 60]),  np.array([10, 255, 255]))
        rmask = rmask1 + rmask2
        # cv.imshow("rmask", rmask)
        bmask = cv.inRange(hsv, np.array([100, 60, 60]), np.array([124, 255, 255]))
        # cv.imshow("bmask", bmask)
        gmask = cv.inRange(hsv, np.array([35, 120, 120]), np.array([77, 255, 255]))
        # cv.imshow("gmask", gmask)
        screen_mask = rmask + bmask + gmask
        # cv.imshow("mask", screen_mask) # test

        #### find contours
        bcontours, hierarchies = cv.findContours(bmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        gcontours, hierarchies = cv.findContours(gmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rcontours, hierarchies = cv.findContours(rmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_all = bcontours + gcontours + rcontours
        contours_selected = list(filter(lambda e:cv.contourArea(e)>100, contours_all))

        #### locate the screen
        points = []
        for cnt in contours_selected:
            for e in cnt:
                points.append(e[0])
        nagative = sorted(points,\
            key=lambda x:x[0] + x[1])
        positive = sorted(points,\
            key=lambda x:x[1] - x[0])
     #### test
        screen_edges = np.array([nagative[0], positive[0], nagative[-1], positive[-1]], np.int32)
        fig = hsv.copy()
        screen = cv.polylines(fig, [screen_edges], isClosed=True, color=[0, 0, 255], thickness=5)
        # cv.imshow('screen', screen)
      ####
        screen_edges = np.array([nagative[0], positive[0], nagative[-1], positive[-1]], np.float32)
        screen_transformed = np.array([[0, 0], [905, 0], [905, 905], [0, 905]], np.float32)
        M = cv.getPerspectiveTransform(screen_edges, screen_transformed)
        transformed = cv.warpPerspective(self.img, M, (905, 905))
        cv.imshow('transformed', transformed)

        return transformed



if __name__=='__main__':
    # img = cv.imread('t4.jpg')
    img = cv.imread('fig91.jpg')
    screen_caping = screen_cap(img)
    screen_caping.screen()


    cv.waitKey(0)
