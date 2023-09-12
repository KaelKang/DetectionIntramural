import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random

class Generator():
    def __init__(self):
        self.img_size = (905,905,3)

    def generate_img(self):
        img = 255 * np.ones(self.img_size,dtype=np.float32)
        for row in range(3):
            for col in range(3):
                row_start_pixel = int(self.img_size[0]//3*row-1)
                row_end_pixel = int(self.img_size[0]//3*(row+1)-1)
                col_start_pixel = int(self.img_size[1]//3*col-1)
                col_end_pixel = int(self.img_size[1]//3*(col+1)-1)

                # -------------------------------------- # 
                # 1. 随即产生背景色
                # 3*3的九宫格中的背景色随机为R,G,B中的一种
                # -------------------------------------- #
                color_idx = random.randint(0,2)
                for i in range(row_start_pixel, row_end_pixel):
                    for j in range(col_start_pixel, col_end_pixel):
                        if color_idx == 0:
                            img[i][j] = (255, 0, 0)
                        elif color_idx == 1:
                            img[i][j] = (0, 255, 0)
                        else:
                            img[i][j] = (0, 0, 255)
                
                # ----------------------------------------------- # 
                # 2. 随即产生数字
                # 在每个九宫格的随机位置中随即产生0-9的随即大小的数字
                # ----------------------------------------------- #
                num_row_pixel = random.choice((100,200))+row_start_pixel
                num_col_pixel = random.choice((100,200))+col_start_pixel
                cv.putText(img, str(random.randint(0,9)), (num_col_pixel-10, num_row_pixel-10), cv.FONT_HERSHEY_PLAIN, random.randint(3,6),(0,0,0),3)
        
                # -------------------------- # 
                # 3.绘制图形
                # 在九宫格随机位置产生图形
                # 0-circle, 1-square, 2-triangle
                # -------------------------- #
                shape = random.randint(0,3)

                # 避免数字和图形重叠
                while True:
                    shape_row_pixel=random.choice((100,200))+row_start_pixel
                    shape_col_pixel=random.choice((100,200))+col_start_pixel
                    if not (shape_col_pixel==num_col_pixel and shape_row_pixel==num_row_pixel):
                        break
                size = random.choice((20,30,40))
                if shape == 0 :
                    cv.circle(img, (shape_col_pixel, shape_row_pixel), size, (0,0,0),3 )
                elif shape == 1:
                    cv.rectangle(img, (shape_col_pixel-size,shape_row_pixel-size), (shape_col_pixel+size,shape_row_pixel+size), (0,0,0), 3)
                else:
                    cv.line(img, (shape_col_pixel,shape_row_pixel-size), (shape_col_pixel-size,shape_row_pixel+size), (0,0,0), 3)
                    cv.line(img, (shape_col_pixel,shape_row_pixel-size), (shape_col_pixel+size,shape_row_pixel+size), (0,0,0), 3)
                    cv.line(img, (shape_col_pixel-size,shape_row_pixel+size), (shape_col_pixel+size,shape_row_pixel+size), (0,0,0), 3)                   
        
        # ------------------------- # 
        # 绘制分界线
        # ------------------------- # 
        cv.line(img,(300,0),(300,900),(0,0,0),2)
        cv.line(img,(600,0),(600,900),(0,0,0),2)
        cv.line(img,(0,300),(900,300),(0,0,0),2)
        cv.line(img,(0,600),(900,600),(0,0,0),2)         
        return img.astype(np.uint8)

if __name__ == "__main__":
    generator = Generator() 
    out_win = "Object"
    cv.namedWindow(out_win, cv.WINDOW_NORMAL)
    cv.setWindowProperty(out_win, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)
    chance = 3
    # while(chance): 
    #     img = generator.generate_img()
    #     cv.imshow(out_win, img)
    #     cv.waitKey(50)
    #     chance -= 1
    img = generator.generate_img()
    cv.imwrite('/home/kael/Documents/pyfiles/ninty.png', img)
    print(img.shape[:2])
    cv.imshow(out_win, img)
    cv.waitKey(0)