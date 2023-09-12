import numpy as np
import cv2 as cv
from screen import screen_cap


class feature_selector:
    def __init__(self, frame):
        self.img = frame
        ## resize the frame captured 
        ## 900x900
        self.width = 905
        self.height = 905
        self.apex1 = (0, 0)
        self.apex2 = (0, self.height)
        self.apex3 = (self.width, 0)
        self.apex4 = (self.width, self.height)
    


    def divide_and_colorDec(self):
        ## divide the img into 9, and select them by color
        for row in range(3):
            for col in range(3):
                dimg = self.img[row*self.height//3 : (row + 1)*self.height//3,\
                    col*self.width//3 : (col + 1)*self.width//3]
                dhsv = cv.cvtColor(dimg, cv.COLOR_BGR2HSV)
                ## test
                # na = str(row + 1) + 'th(st/nd/rd) row, ' + str(col + 1) + 'th(st/nd/rd) col' 
                # cv.imshow(na, dhsv)
                ##
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
                    print(f'{row*3 + col + 1}-red', end="")
                elif color_vote[1]>color_vote[0] and color_vote[1]>color_vote[2]:
                    print(f'{row*3 + col + 1}-green', end="")
                else:
                    print(f'{row*3 + col + 1}-blue', end="")
                
                self.shape_dec(dimg, row, col) # link

                print('  ', end='')
            print('\n')
        


    def shape_dec(self, img, row, col):
        winname = str(row + 1) + 'th(st/nd/rd) row, ' + str(col + 1) + 'th(st/nd/rd) col'
        ## detect the shape in the img, using matchTemplate
        ## process
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        adapt_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 7)
        # th, adapt_thresh = cv.threshold(gray,0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # doesn't work well
        cropped = adapt_thresh[20:-20, 20:-20]
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv.erode(cropped, kernel, iterations=3) # thick
        dilated = cv.dilate(eroded, kernel, iterations=4)  # The image's been eroded once, so it requires dilating one more time 
        cv.imwrite('/home/kael/Documents/pyfiles/s.png', dilated)  # convert 1 channnel images to 3 channels
        img_3 = cv.imread('/home/kael/Documents/pyfiles/s.png')
        ## test
        
        # cv.imshow('gary:' + na, gray) #test
        cv.imshow('adapt_thresh:' + winname, dilated) # test
        ##
        ## match compare
        vote = {'mtri':0, 'mrec':0, 'mcir':0}
        for item in ['mtri', 'mrec', 'mcir']:
            min = [0, 0, 0]
            for j in range(3):                
                na = '/home/kael/Documents/pyfiles/' + item + '_' + str(j) + '.jpg'
                shape = cv.imread(na)
                # num = cv.cvtColor(num, cv.COLOR_BGR2RGB) #??
                res = cv.matchTemplate(img_3, shape, cv.TM_SQDIFF)
                min[j], _, _, _ = cv.minMaxLoc(res)
            order = np.argsort(min)
            # print(min) #test
            vote[item] = min[order[0]]
        order = sorted(vote.items(),\
            key = lambda item:item[1])
        # print(ordered) #test
        print(f'-{order[0][0][1:]}', end="")

        self.figure_dec(cropped, row, col) # link



    def figure_dec(self, img, row, col):
        winname = str(row + 1) + 'th(st/nd/rd) row, ' + str(col + 1) + 'th(st/nd/rd) col'
        ## detect the figure in one part of image, using matchTemplate 
        kernel = np.ones((3, 3), np.uint8)
        img_1 = img
        # img_1 = cv.erode(img_1, kernel, iterations=1)
        img_1 = cv.dilate(img_1, kernel, iterations=1)
        
        cv.imwrite('/home/kael/Documents/pyfiles/s.png', img_1)  # convert 1 channnel images to 3 channels
        img_3 = cv.imread('/home/kael/Documents/pyfiles/s.png')

        vote = {'num0':[0, 0], 'num1':[0, 0], 'num2':[0, 0], 'num3':[0, 0], 'num4':[0, 0], 'num5':[0, 0], 'num6':[0, 0], 'num7':[0, 0], 'num8':[0, 0], 'num9':[0, 0]}
        for item in ['num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9']:
            min = [0, 0, 0, 0]
            minLoc = [0, 0, 0, 0] # test
            for j in range(4):
                na = '/home/kael/Documents/pyfiles/' + item + '_' + str(j) + '.png'
                num = cv.imread(na)
                res = cv.matchTemplate(img_3, num, cv.TM_SQDIFF)
                min[j], _, minLoc[j], _ = cv.minMaxLoc(res)
            order = np.argsort(min)
            vote[item][0] = min[order[0]]
            vote[item][1] = minLoc[order[0]]

        order = sorted(vote.items(),\
                      key=lambda x:x[1][0])
        print(f'-{order[0][0][3:]}', end='')
        # print(f'-{order[1][0][3:]}', end='') # test
        # print(f'-{order[2][0][3:]}', end='') # test
        # for i in range(10):
        #     print(f'-{order[i][0][3:]}:{order[i][1]}', end='')

        marked = cv.rectangle(img, order[0][1][1], (order[0][1][1][0]+50, order[0][1][1][1]+50), 0, 1)
        cv.imshow('adapt_thresh:' + winname, marked) # test

if __name__=="__main__":
    img = cv.imread('/home/kael/Documents/gxcap/fig4.jpg')
    cv.imshow('img', img)
    screen_caping = screen_cap(img)
    img = screen_caping.screen()
    feature_selectoring = feature_selector(img)
    feature_selectoring.divide_and_colorDec()




    cv.waitKey(0)