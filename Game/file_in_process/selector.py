# selector.py
import cv2 as cv
import time
import numpy as np

it = 0


class feature_selector():
    def __init__(self, frame):
        '''
        resize the frame captured 
        900x900
        '''
        self.width = 901
        self.height = 901
        dimensions = (self.width, self.height)
        self.img = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
        self.apex1 = (0, 0)
        self.apex2 = (0, self.height)
        self.apex3 = (self.width, 0)
        self.apex4 = (self.width, self.height)
        self.it = 0
        # cv.imshow('resized', self.img)

    def divided_and_colored_and_(self):
        '''
        divide the img into 9, and select them by color
        '''
        ####    throw error
        self.firgue_dec(self.img)
        
        img_test = []
        for x in range(3):
            for y in range(3):
                dimg = self.img[x*self.height//3 : (x + 1)*self.height//3, y*self.width//3 : (y + 1)*self.width//3]
                img_test.append(dimg)
                
                iname = str(x) + "," + str(y) 
                # cv.imshow(iname, dimg)
                dhsv = cv.cvtColor(dimg, cv.COLOR_BGR2HSV)
                # cv.imshow(iname, dhsv)
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

        return img_test
            
    def shape_dec(self, img):
        '''
        detect the shape in the img
        '''
        # cv.imshow('s', self.img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow("gray", gray)
        thresh, thresholded = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
        cv.imshow('thresholded', thresholded)
        canny = cv.Canny(thresholded, 125, 175)
        # cv.imshow("canny", canny)
        contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # blank = np.zeros((img.shape[1], img.shape[0]), 'uint8')
        blank = np.zeros((900, 900, 3), np.uint8)
        # contours_selected = list(filter(lambda e: cv.contourArea(e)>2, contours))
        contours_selected = list(filter(lambda e: cv.arcLength(e, True)<1000 and cv.contourArea(e)>1, contours)) 
        for i in range(len(contours_selected)): 
            cv.drawContours(canny, contours_selected, i, (255, 255, 255), -1, cv.LINE_AA)
        cv.imshow('drawed', canny)
        contours2, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours_selected2 = list(filter(lambda e: cv.arcLength(e, True)<1000 and cv.contourArea(e)>1, contours2)) 
        cv.drawContours(canny, contours_selected2, -1, (255, 255, 255), -1, cv.LINE_AA)
        cv.imshow('drawedd', canny)

        minimum = 7
        approx = []
        match_res = []
        itt=0
        for cnt in range(len(contours_selected2)):
            # print(f"pp{cnt}pp")
            eps = 0.04*cv.arcLength(contours_selected2[cnt], True)
            appr = cv.approxPolyDP(contours_selected2[cnt], eps, True)
            # print(f"-{len(appr)}", end='')
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
            shape = thresholded[int(top_point_y):int(bottom_point_y), int(top_point_x):int(bottom_point_x)]
            # cv.rectangle(img, (top_point_x, top_point_y), (bottom_point_x, bottom_point_y), (0, 0, 255), 2)
            x,y,w,h = cv.boundingRect(contours_selected2[cnt])
            cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
            # cv.drawContours(img, contours_selected2, itt, (0, 0, 255), 1, cv.LINE_AA)
            itt+=1
            na = str(self.it)
            self.it += 1
            # cv.imshow(na,img)
            # # # cv.imshow('s', shape)
            # shape = np.array(shape, np.uint8)
            cv.imwrite('s.png', shape)
            shaped = cv.imread('s.png')
 
            # cv.imshow(na, shaped)
            p, q=self.firgue_dec(shaped)
            # print(p, ' ', q, end=' ')
            match_res.append([p, q])



        if minimum==3:
            print('-triangle', end='')
        elif minimum==4:
            print('-rectangle', end='')
        elif minimum>5:
            print('-circle', end='')
        # print(f":{len(contours_selected2)}:")
        # cv.drawContours(blank, contours_selected2, -1, (0, 0, 255), 1, cv.LINE_AA)
        
        # na = str(self.it)
        # self.it += 1
        # cv.imshow(na, blank)
        # blank = np.zeros((900,900, 3), np.uint8)
        # cv.drawContours(blank, approx, -1, (0, 0, 255), 1, cv.LINE_AA)
        # cv.imshow('na'+na, blank)
        match_res.sort(key=lambda x:x[1], reverse=False)
        # order = np.argsort(match_res)

        # print(order)
        print(f"-{match_res[0][0]}", end=' ')
        # print("\n")
    
    def firgue_dec(self, img):
        '''
        find the number
        '''
        # nums = []
        minimum = []
        for i in range(10):
            name = 'num' + str(i) +'.png'
            num = cv.imread(name)
            num = cv.resize(num, (img.shape[1], img.shape[0]))
            # nums.append(num)
        
            res = cv.matchTemplate(img, num, cv.TM_SQDIFF)
            min, max, minLoc, maxLoc = cv.minMaxLoc(res)
            minimum.append(min)
        order = np.argsort(minimum)

        return order[0], minimum[order[0]]
        
        



if __name__=="__main__":
    start = time.perf_counter()
#######################################################
    img = cv.imread('ninty.png')
    img0 = cv.imread('num8.png')
    img2 = cv.resize(img, (300,300))
    cv.imshow('8',img2)
    
    t = feature_selector(img0)
    t.shape_dec(img0)

    print("\ntest")
    print(t.firgue_dec(img0))
    

    print(img.shape[:2])
    

    selector = feature_selector(img)
    imgs = selector.divided_and_colored_and_()
    print(selector.width)
    
    
######################################################
    end = time.perf_counter()
    print(end-start, "seconds")
    
    cv.waitKey(0)
