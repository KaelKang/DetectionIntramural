# selector.py
import cv2 as cv
import numpy as np



class feature_selector():
    def __init__(self, frame):
        #### resize the frame captured 
        #### 900x900
        self.width = 901
        self.height = 901
        dimensions = (self.width, self.height)
        self.img = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
        self.apex1 = (0, 0)
        self.apex2 = (0, self.height)
        self.apex3 = (self.width, 0)
        self.apex4 = (self.width, self.height)
        self.it = 0

    def divided_and_colored_and_(self):
        ####    throw error
        self.firgue_dec(self.img)

        #### divide the img into 9, and select them by color
        for x in range(3):
            for y in range(3):
                dimg = self.img[x*self.height//3 : (x + 1)*self.height//3,\
                    y*self.width//3 : (y + 1)*self.width//3]
                dhsv = cv.cvtColor(dimg, cv.COLOR_BGR2HSV)
                color_vote = [0, 0, 0]
                for i in range(1, 4):
                    ther_point = dhsv[i*(dhsv.shape[0]//4), i*(dhsv.shape[1]//4)]
                    hue = ther_point[0]
                    if hue<22:
                        color_vote[0] += 1
                    elif hue<78:
                        color_vote[1] += 1
                    elif hue<131:
                        color_vote[2] += 1
                    else:
                        color_vote[0] += 1
                    
                if color_vote[0]>color_vote[1] and color_vote[0]>color_vote[2]:
                    print(f'{x*3 + y + 1}-red', end="")
                elif color_vote[1]>color_vote[0] and color_vote[1]>color_vote[2]:
                    print(f'{x*3 + y + 1}-green', end="")
                else:
                    print(f'{x*3 + y + 1}-blue', end="")

                self.shape_dec(dimg)

                print('  ', end='')
            print('\n')
            
    def shape_dec(self, img):
        #### detect the shape in the img
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh, thresholded = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
        canny = cv.Canny(thresholded, 125, 175)
        contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        blank = np.zeros((900, 900, 3), np.uint8)
        contours_selected = list(filter(lambda e: cv.arcLength(e, True)<1000 and cv.contourArea(e)>1, contours)) 
        for i in range(len(contours_selected)): 
            cv.drawContours(canny, contours_selected, i, (255, 255, 255), -1, cv.LINE_AA)
        contours2, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours_selected2 = list(filter(\
            lambda e: cv.arcLength(e, True)<1000 and cv.contourArea(e)>1,\
            contours2)) 

        minimum = 7
        approx = []
        match_res = []
        for cnt in range(len(contours_selected2)):
            eps = 0.04*cv.arcLength(contours_selected2[cnt], True)
            appr = cv.approxPolyDP(contours_selected2[cnt], eps, True)
            approx.append(appr)
            if len(appr)==3:
                minimum = len(appr)
            
            elif len(appr)<minimum:
                minimum = len(appr)
            
            top_point_x = 12047
            top_point_y = 12047
            bottom_point_x = 0
            bottom_point_y = 0
            for e in contours_selected2[cnt]:
                top_point_x = min(top_point_x, e[0][0])
                top_point_y = min(top_point_y, e[0][1])
                bottom_point_x = max(bottom_point_x, e[0][0])
                bottom_point_y = max(bottom_point_y, e[0][1])
            shape = thresholded[int(top_point_y):int(bottom_point_y),\
                 int(top_point_x):int(bottom_point_x)]

            cv.imwrite('s.png', shape)
            shaped = cv.imread('s.png')
 
            p, q=self.firgue_dec(shaped)
            match_res.append([p, q])
            


        if minimum==3:
            print('-triangle', end='')
        elif minimum==4:
            print('-rectangle', end='')
        elif minimum>5:
            print('-circle', end='')
        match_res.sort(key=lambda x:x[1], reverse=False)
        print(f"-{match_res[0][0]}", end=' ')
    
    def firgue_dec(self, img):
        '''
        find the number
        '''
        minimum = []
        for i in range(10):
            name = 'num' + str(i) +'.png'
            num = cv.imread(name)
            num = cv.resize(num, (img.shape[1], img.shape[0]))
        
            res = cv.matchTemplate(img, num, cv.TM_SQDIFF)
            min, max, minLoc, maxLoc = cv.minMaxLoc(res)
            minimum.append(min)
        order = np.argsort(minimum)

        return order[0], minimum[order[0]]
        
        



if __name__=="__main__":
    
    img = cv.imread('ninty.png')
    cv.imshow('img', img)
    print(img.shape[:2])
    selector = feature_selector(img)
    imgs = selector.divided_and_colored_and_()
    
    
    
    cv.waitKey(0)
