import numpy as np
import time
import cv2 as cv

########################################################
######################################################
start = time.perf_counter()

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
blank = np.zeros((901,901,3))

#### morphology progcess 
img = cv.imread('/home/kael/Documents/pyfiles/t4.jpg')
# cv.imshow('img', img)
big = cv.resize(img, (901, 901), interpolation=cv.INTER_AREA)
cv.imshow('big', big)
smooth = cv.bilateralFilter(big, d=15, sigmaColor=35, sigmaSpace=25)
# dilated = cv.dilate(big,kernel, iterations=1)
# cv.imshow("dilated",dilated)
eroded = cv.erode(big, kernel, iterations=2)
cv.imshow("eroded", eroded)
dilated = cv.dilate(eroded,kernel, iterations=1)
cv.imshow("dilated",dilated)
eroded = cv.erode(dilated, kernel, iterations=1)
cv.imshow("eroded", eroded)

#### devide the ranges of blue, green, red
hsv = cv.cvtColor(eroded, cv.COLOR_BGR2HSV)
cv.imshow('hsv', hsv)

rmask = cv.inRange(hsv, np.array([156, 120, 120]),  np.array([180, 255, 255]))
cv.imshow('rmask', rmask)
bmask = cv.inRange(hsv, np.array([100, 150, 150]), np.array([124, 255, 255]))
cv.imshow('bmask', bmask)
gmask = cv.inRange(hsv, np.array([35, 120, 120]), np.array([77, 255, 255]))
cv.imshow('gmask', gmask)
screen_mask = rmask + bmask + gmask
cv.imshow('screnn_mask', screen_mask)
screen_eroded = cv.erode(screen_mask, kernel, iterations=2)
cv.imshow('screnn_eroded', screen_eroded)
screen_dilated = cv.dilate(screen_eroded, kernel, iterations=3)
cv.imshow('screnn_dilated', screen_dilated)
# gcanny = cv.Canny(gmask, 125, 175)
# cv.imshow('gcanny', gcanny)

#### find contours
bcontours, hierarchies = cv.findContours(bmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
bcontours_show = big.copy()
cv.drawContours(bcontours_show, bcontours, -1, (0, 0, 255), 2, cv.LINE_AA)
cv.imshow('bcontours', bcontours_show)
gcontours, hierarchies = cv.findContours(gmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
gcontours_show = big.copy()
cv.drawContours(gcontours_show, gcontours, -1, (0, 0, 255), 2, cv.LINE_AA)
cv.imshow('gcontours', gcontours_show)
rcontours, hierarchies = cv.findContours(rmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
rcontours_show = big.copy()
cv.drawContours(rcontours_show, rcontours, -1, (0, 0, 255), 2, cv.LINE_AA)
cv.imshow('rcontours', rcontours_show)

contours_all = bcontours + gcontours + rcontours
contours_selected = list(filter(lambda e:cv.contourArea(e)>10, contours_all))
allcontours_show = big.copy()
cv.drawContours(allcontours_show, contours_selected, -1, (0, 255, 255), -1, cv.LINE_AA)
cv.imshow('allcontours', allcontours_show)

#### locate the screen
points = []

for cnt in contours_selected:
    for e in cnt:
        points.append(e[0])
nagative = sorted(points,\
    key=lambda x:x[0] + x[1])
positive = sorted(points,\
    key=lambda x:x[1] - x[0])

screen_edges = np.array([nagative[0], positive[0], nagative[-1], positive[-1]], np.int32)

screen = cv.polylines(eroded, [screen_edges], isClosed=True, color=[0, 0, 255], thickness=1)
cv.imshow('screen', eroded)
screen_edges = np.array([nagative[0], positive[0], nagative[-1], positive[-1]], np.float32)
screen_transformed = np.array([[0, 0], [900, 0], [900, 900], [0, 900]], np.float32)
M = cv.getPerspectiveTransform(screen_edges, screen_transformed)
transformed = cv.warpPerspective(big,M,(901, 901))
cv.imshow('transformed', transformed)
# rresult = cv.bitwise_and(big, big, mask=rmask)
# gresult = cv.bitwise_and(big, big, mask=gmask)
# bresult = cv.bitwise_and(big, big, mask=bmask)
# cv.imshow('rresoult', rresult)
# cv.imshow('gresoult', gresult)
# cv.imshow('bresoult', bresult)

# smooth = cv.bilateralFilter(eroded, d=15, sigmaColor=35, sigmaSpace=25)
# cv.imshow('bilateral', smooth)
# smooth = cv.GaussianBlur(big, ksize=(3, 3), sigmaX=0)
# cv.imshow('gauss', smooth)
# gray = cv.cvtColor(smooth, cv.COLOR_BGR2GRAY)
# # cv.imshow('gray', gray)
# thresh, thresholded = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)
# cv.imshow('thresholded', thresholded)
# canny = cv.Canny(gray, 125, 175)
# cv.imshow('canny', canny)
print(len(points))
print(screen_edges)
# print(positive)
# print(nagative)

#############################################
######################################################
end = time.perf_counter()
print(end-start, "seconds")


cv.waitKey(0)