'''
@Description : Recognize color-shape-number in cropped image
@Author : Kael
@Time : 2023/1/16
'''
import numpy as np
import cv2 as cv
from screen import screen_cap



# a board showing the result
# show = np.zeros((901, 1001, 1), np.uint8)
# cv.putText(show, "THE ZOMBIES ", (150, 75), cv.FONT_HERSHEY_PLAIN, 6, 255, 3)
# cv.putText(show, "ATE YOUR", (230, 175), cv.FONT_HERSHEY_PLAIN, 5, 255, 2)
# cv.putText(show, "BRAIN!", (250, 275), cv.FONT_HERSHEY_PLAIN, 7, 255, 4)



class config:
    width = 905
    height = 905
    iterations = 1
    cropped_x = 20
    cropped_y = 20
    kernel = np.ones((3, 3), np.uint8)



class feature_selector:
    def __init__(self, frame):
        self.show = np.zeros((901, 1001, 1), np.uint8)
        self.img = frame
        ## resize the frame captured 
        ## 900x900
        self.width = config.width
        self.height = config.height
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
                    #show
                    show_str = str(row*3 + col + 1) + '-red'
                    cv.putText(self.show, show_str, (50 + col*300, 400 + row*100), cv.FONT_HERSHEY_PLAIN, 2, 255, 2)
                elif color_vote[1]>color_vote[0] and color_vote[1]>color_vote[2]:
                    print(f'{row*3 + col + 1}-green', end="")
                    #show
                    show_str = str(row*3 + col + 1) + '-green'
                    cv.putText(self.show, show_str, (50 + col*300, 400 + row*100), cv.FONT_HERSHEY_PLAIN, 2, 255, 2)
                else:
                    print(f'{row*3 + col + 1}-blue', end="")
                    #show
                    show_str = str(row*3 + col + 1) + '-blue'
                    cv.putText(self.show, show_str, (50 + col*300, 400 + row*100), cv.FONT_HERSHEY_PLAIN, 2, 255, 2)
                
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
        cropped = adapt_thresh[config.cropped_y:-config.cropped_x, config.cropped_y:-config.cropped_x] # argue
        # kernel = np.ones((3, 3), np.uint8)
        kernel = config.kernel
        eroded = cv.erode(cropped, kernel, iterations=3) # thick # joint agrue
        dilated = cv.dilate(eroded, kernel, iterations=4)  # The image's been eroded once, so it requires dilating one more time 
        cv.imwrite('/home/kael/Documents/pyfiles/s.png', dilated)  # convert 1 channnel images to 3 channels
        img_3 = cv.imread('/home/kael/Documents/pyfiles/s.png')
        ## test
        
        # cv.imshow('gary:' + na, gray) #test
        # cv.imshow('adapt_thresh:' + winname, dilated) # test
        ##
        ## match compare
        vote = {'mtri':0, 'mrec':0, 'mcir':0}
        for item in ['mtri', 'mrec', 'mcir']:
            min = [0, 0, 0]
            for j in range(3):                
                na = '/home/kael/Documents/pyfiles/' + item + '_' + str(j) + '.jpg'
                shape = cv.imread(na)
                shape = cv.GaussianBlur(shape, (3, 3), 0)
                # shape = cv.erode(shape, kernel, iterations=1) # argue
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

        #show
        show_str = '-' + order[0][0][1:]
        cv.putText(self.show, show_str, (180 + col*300, 400 + row*100), cv.FONT_HERSHEY_PLAIN, 2, 255, 2)

        self.figure_dec(cropped, row, col) # link



    def figure_dec(self, img, row, col):
        winname = str(row + 1) + 'th(st/nd/rd) row, ' + str(col + 1) + 'th(st/nd/rd) col'
        blank = np.zeros((301, 301, 1), np.uint8) # argue
        ## detect the figure in one part of image, using matchTemplate 
        # kernel = np.ones((3, 3), np.uint8) # argue
        kernel = config.kernel
        img_1 = img
        img_1 = cv.erode(img_1, kernel, iterations=1) # argue
        # img_1 = cv.dilate(img, kernel, iterations=3)
        canny = cv.Canny(img_1, 50, 175)
        # cv.imshow('canny:' + winname, canny)
        contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours_selected = list(filter(lambda e: cv.arcLength(e, True)>150, contours))
        # cv.drawContours(blank, contours_selected, -1, 255, 3, cv.LINE_AA) # test
        # cv.imshow('adapt_thresh:' + winname, blank) # test
        
        vote = {'num0':0, 'num1':0, 'num2':0, 'num3':0, 'num4':0, 'num5':0, 'num6':0, 'num7':0, 'num8':0, 'num9':0}
        for cnt in contours_selected:
            x, y, w, h = cv.boundingRect(cnt)
            # cv.rectangle(blank, (x, y), (x + w, y + h), 255, 3) # test
            cropped = img[y : y + h, x : x + w]
            dilated = cv.erode(cropped, kernel, iterations=1) # argue
            cv.imshow('crop' + str(x) + winname, cropped) # test
            cv.imwrite('/home/kael/Documents/pyfiles/s.png', dilated)
            img_3 = cv.imread('/home/kael/Documents/pyfiles/s.png')

            subvote = {'num0':0, 'num1':0, 'num2':0, 'num3':0, 'num4':0, 'num5':0, 'num6':0, 'num7':0, 'num8':0, 'num9':0}
            for item in ['num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9']:
                na = '/home/kael/Documents/pyfiles/' + item + '_' + '0' + '.png' #argue
                num = cv.imread(na)
                num = cv.GaussianBlur(num, (3, 3), 0) # argue
                num = cv.erode(num, kernel, iterations=1) # argue
                # num = cv.dilate(num, kernel, iterations=3) # argue
                # num = cv.resize(num, (w, h))                         # joint argue
                img_3 = cv.resize(img_3, (num.shape[1], num.shape[0])) # joint agrue
                img_3 = cv.GaussianBlur(img_3, (3, 3), 0) # argue
                res = cv.matchTemplate(img_3, num, cv.TM_SQDIFF)
                min, max, minLoc, maxLoc = cv.minMaxLoc(res)
                subvote[item] = min
            order = sorted(subvote.items(),\
                          key = lambda x : x[1])
            if vote[order[0][0]]==0 or vote[order[0][0]]>order[0][1]:
                vote[order[0][0]] = order[0][1]

        for item in ['num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9']:
            if vote[item]==0:
                del vote[item]
        
        order = sorted(vote.items(),\
                       key = lambda  x : x[1])
        
        print(f"-{order[0][0][3:]}", end='')
        #show
        show_str = '-' + order[0][0][3:]
        cv.putText(self.show, show_str, (270 + col*300, 400 + row*100), cv.FONT_HERSHEY_PLAIN, 2, 255, 2)


        # show result
        cv.imshow('show', self.show)    
            
        # cv.imshow('adapt_thresh:' + winname, blank) #test


if __name__=="__main__":
    img = cv.imread('/home/kael/Documents/gxcap/fig4.jpg')
    cv.imshow('img', img)
    screen_caping = screen_cap(img)
    img = screen_caping.screen()
    feature_selectoring = feature_selector(img)
    feature_selectoring.divide_and_colorDec()




    cv.waitKey(0)