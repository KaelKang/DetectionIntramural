import numpy as np
import cv2  as cv
from screen import screen_cap
from selector import feature_selector



if __name__=='__main__':
    img = cv.imread('/home/kael/Documents/gxcap/fig91.jpg')
    screen_caping = screen_cap(img)
    img = screen_caping.screen()
    cv.imshow('img', img)
    feature_selectoring = feature_selector(img)
    feature_selectoring.divided_and_colored_and_()



    cv.waitKey(0)
