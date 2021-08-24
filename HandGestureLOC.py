import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import pickle


class Window:
    def __init__(self):
        self.SIG_model = tf.keras.models.load_model("Models/model_loc1.h5")
        self.BG_model = tf.keras.models.load_model("Models/model_bg_2.h5")
        self.classes = sorted(os.listdir("dataset"))
        self.scaler = None
        with open("Models/scaler.pkl","rb") as f:
            self.scaler = pickle.load(f)
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

    def draw_bbox(self,bbox,truth,predict,img):
        bbox = self.scaler.inverse_transform(bbox).astype(int)[0]
        st = self.classes[predict]+": "+str(truth)[:4]+"%"
        cv.putText(img,st,(0,40), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
            
        cv.rectangle(img,pt1=(bbox[1],bbox[0]),pt2=(bbox[3],bbox[2]),color=(0,255,0),thickness=2)

    def try_find(self,img):
        blurred = cv.bilateralFilter(img,9,60,60)
        blurred = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
        blurred = blurred.reshape((1,480,640,1))

        array,bbox = self.SIG_model.predict(blurred)
        array = array[0]
        predict = np.argmax(array)
        truth = array[predict]

        self.draw_bbox(bbox,truth,predict,img)

    #480, 640
    def slide_window(self,frame):
        w = 150
        h = 250
        x = 0
        y = 0
        
        #for x in range(0,frame.shape[1],w):
        #cv.rectangle(frame,(x,y),(x+w+1,y+h+1),(0,255,0),0)
        
        self.try_find(frame)

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
