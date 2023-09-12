import numpy as np 
import cv2 as cv

img = cv.imread("/home/kael/Documents/pyfiles/nums.png")
# cv.imshow('img', img)
blank = np.zeros((500, 2000, 3), np.uint8)
blank[:] = 255, 255, 255
# cv.imshow('blank', blank)

# cv.putText(blank, "0 1 2 3 4 5 6 7 8 9", (200, 400), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 3)
# cv.putText(blank, "0 1 2 3 4 5 6 7 8 9", (200, 400), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 3)
# cv.putText(blank, "0 1 2 3 4 5 6 7 8 9", (200, 400), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 3)
cv.putText(blank, "0 1 2 3 4 5 6 7 8 9", (200, 400), cv.FONT_HERSHEY_PLAIN, 6, (0, 0, 0), 3)

canny = cv.Canny(blank, 125, 175)
# cv.imshow('canny', canny)
contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# for i in range(len(contours)):
#     cv.drawContours(canny, contours, i, 255, -1, cv.LINE_AA)
cv.drawContours(canny, contours, -1, 255, -1, cv.LINE_AA)
# cv.imshow('canny2', canny)
# canny2 = cv.Canny(blank, 125, 175)
contours2, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# for i in range(len(contours2)):
#     cv.drawContours(canny, contours2, i, 255, -1, cv.LINE_AA)
cv.drawContours(canny, contours2, -1, 255, -1, cv.LINE_AA)
# cv.imshow('canny3', canny)

# for i in range(len(contours2)):
#     x, y, w, h = cv.boundingRect(contours2[i])
#     cv.rectangle(blank, (x, y), (x + w, y + h), (0, 0, 255), 2)
# cv.imshow('blank', blank)
x, y, w, h = cv.boundingRect(contours2[9])
copy = blank.copy()
cv.rectangle(blank, (x, y), (x + w, y + h), 0, 2)
cv.imshow('blank', blank)
fig = copy[y:y + h, x:x + w]
cv.imshow('fig', fig)
cv.imwrite('num2_3.png', fig)

nums = []
for cnt in contours2:
    top_point_x = 10000
    top_point_y = 10000
    bottom_point_x = 0
    bottom_point_y = 0
    for e in cnt:
        top_point_x = min(top_point_x, e[0][0])
        top_point_y = min(top_point_y, e[0][1])
        bottom_point_x = max(bottom_point_x, e[0][0])
        bottom_point_y = max(bottom_point_y, e[0][1])

    
    # cv.rectangle(blank, (top_point_x, top_point_y), (bottom_point_x, bottom_point_y), (0, 0, 255), 2)
    num = img[int(top_point_y):int(bottom_point_y), int(top_point_x):int(bottom_point_x)]
    # num = img[300:400, 500:600]
  
    # cv.imshow('img0', num)
    nums.append(num)
    

# for i in range(10):
#     na = 'num' + str(i) 
#     cv.imshow(na, nums[i])
# cv.imwrite('num7.png', nums[5])







cv.waitKey(0)