import cv2 as cv
import numpy as np
import os
import tensorflow as tf

def Filter(cnt):
    min_w = 150
    min_h = 250
    _,_,w,h = cv.boundingRect(cnt)
    if w >=min_w and h >= min_h:
        return cnt
    else:
        return []

class Window:
    def __init__(self):
        self.SIG_model = tf.keras.models.load_model("Models/model_8.h5")
        self.BG_model = tf.keras.models.load_model("Models/model_bg_2.h5")
        self.classes = sorted(os.listdir("dataset"))
        print(self.classes)

    def get_laplacian_filter(self,img):
        dst = cv.Laplacian(img, cv.CV_16S, ksize= 3)
        abs_dst = cv.convertScaleAbs(dst)
        
        kernel = np.ones(shape=(2,2))
        abs_dst = cv.dilate(abs_dst,kernel)

        return abs_dst

    def get_contour_size(self, filtered_img):
        _, thresh1 = cv.threshold(filtered_img, 127, 255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        contours,_ = cv.findContours(thresh1.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnt = max(contours, key=lambda x:cv.contourArea(x))

        return cnt

    def draw_bbox(self,x,y,w,h,truth,predict,img):
        st = self.classes[predict]+": "+str(truth)[:4]+"%"
        cv.putText(img,st,(0,40), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
            
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),0)
        posx = x
        posy = y

    def try_find(self,img,frame):
        blurred = cv.bilateralFilter(img,9,60,60)
        blurred = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)

        laplacian = self.get_laplacian_filter(blurred)
        cnt = self.get_contour_size(laplacian)

        laplacian = laplacian.reshape(1,250,150,1)
        array = self.BG_model.predict(laplacian)[0]
        predict = np.argmax(array)

        truth = array[predict]
        cv.imshow("Laplaciano",laplacian.reshape((250,150)))
        if truth > 0.9 and predict != 1:
            array = self.SIG_model.predict(laplacian)[0]
            predict = np.argmax(array)
            truth = array[predict]

            x, y, w, h = cv.boundingRect(cnt)
            self.draw_bbox(x,y,w,h,truth,predict,frame)

    #480, 640
    def slide_window(self,frame):
        w = 150
        h = 250
        
        blurred = cv.bilateralFilter(frame,9,75,75)
        gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
        _,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours,_ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        cnts = list(map(Filter,contours))
        #print(len(cnts))
        
        for cnt in cnts:
            if len(cnt) > 0:
                x,y,w,h = cv.boundingRect(cnt)
                crop = frame[y:y+h,x:x+w]
                print(crop.shape)
                if crop.shape[0] != 250 or crop.shape[1] != 150:
                    crop = cv.resize(crop,(150,250),cv.INTER_CUBIC)
                self.try_find(crop,frame)
        #cv.rectangle(frame,(x,y),(x+w+1,y+h+1),(0,255,0),0)
            
        #crop = frame[y+1:y+1+h, x+1:x+1+w]
        #self.try_find(crop,frame)

    def start(self):
        cap = cv.VideoCapture(0)

        while(cap.isOpened()):
            _, img = cap.read()

            self.slide_window(img)
            
            cv.imshow('Gesture', img)
            k = cv.waitKey(10)
            if k == 27:
                break

        cv.destroyAllWindows()
        cap.release()    

window = Window()
window.start()
