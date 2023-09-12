import numpy as np
import cv2 as cv



# blank = np.zeros((900, 1500, 3), np.uint8) 
# # cv.imshow('blank', blank)
# blank[:] = (255, 255, 255)

# cv.circle(blank, (100, 400), 20, (0, 0, 0), 3)
# cv.circle(blank, (200, 400), 30, (0, 0, 0), 3)
# cv.circle(blank, (300, 400), 40, (0, 0, 0), 3)
# cv.rectangle(blank, (500,400), (540, 440), (0,0,0), 3)
# cv.rectangle(blank, (600,400), (660, 460), (0,0,0), 3)
# cv.rectangle(blank, (700,400), (780, 480), (0,0,0), 3)

# cv.line(blank, (900,400-20), (900-20,400+20), (0,0,0), 3)
# cv.line(blank, (900,400-20), (900+20,400+20), (0,0,0), 3)
# cv.line(blank, (900-20,400+20), (900+20,400+20), (0,0,0), 3) 

# cv.line(blank, (1000,400-30), (1000-30,400+30), (0,0,0), 3)
# cv.line(blank, (1000,400-30), (1000+30,400+30), (0,0,0), 3)
# cv.line(blank, (1000-30,400+30), (1000+30,400+30), (0,0,0), 3) 

# cv.line(blank, (1100,400-40), (1100-40,400+40), (0,0,0), 3)
# cv.line(blank, (1100,400-40), (1100+40,400+40), (0,0,0), 3)
# cv.line(blank, (1100-40,400+40), (1100+40,400+40), (0,0,0), 3) 

# # cv.putText(blank, '3 \n 4', (1200-10, 400-10), cv.FONT_HERSHEY_PLAIN, 3,(0,0,0),3)
# cv.imwrite('figues.jpg', blank)
# cv.imshow('blank', blank)

img = cv.imread('/home/kael/Documents/pyfiles/figues.jpg')
# cv.imshow('img', img)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(img2, 125, 175)
# cv.imshow('canny', canny)
contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# cv.drawContours(img, contours, -1, (0, 0, 255), 1, cv.LINE_AA)
# cv.imshow('img', img)
# print(len(contours))
# for cnt in contours:
#     x, y, w, h = cv.boundingRect(cnt)
#     cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
img2 = img.copy()
x, y, w, h = cv.boundingRect(contours[6])
cv.rectangle(img2, (x,y), (x+w, y+h), (0, 0, 255), 2)
fig = img[y:y + h, x:x + w]
cv.imshow('img2', img2)
cv.imshow('fig', fig)
cv.imwrite('cir_1.jpg', fig)



cv.waitKey(0)