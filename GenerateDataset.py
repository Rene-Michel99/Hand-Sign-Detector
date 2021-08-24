import cv2 as cv
import numpy as np
import os

cap = cv.VideoCapture(0)

pastas = sorted(os.listdir("dataset"))
pastas = pastas[::-1]

dirr = pastas.pop()
print(dirr)
count = len(os.listdir("dataset/"+dirr))

ready = False
MODE = True

def draw_left_rect(img):
    cv.rectangle(img,(0,0),(150+2,250+2),(0,255,0),0)
    return 1,1

#480-250, 640-150
def draw_right_rect(img):
    cv.rectangle(img,(490-1,0),(640,250+2),(0,255,0),0)
    return 490,1

def view_laplacian(img):
    img = cv.bilateralFilter(img,9+7,75+7,75+7)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    dst = cv.Laplacian(img, cv.CV_16S, ksize=5)
    abs_dst = cv.convertScaleAbs(dst)
    
    kernel = np.ones(shape=(3,3))
    abs_dst = cv.dilate(abs_dst,kernel)
    abs_dst = cv.medianBlur(abs_dst,5)

    return abs_dst


laplacian_mode = False
while(cap.isOpened()):
    x = y = 0
    ret, img = cap.read()

    if MODE:
        x,y = draw_left_rect(img)
    else:
        x,y = draw_right_rect(img)

    if ready:
        crop = img[y:y+250,x:x+150]
        cv.putText(img,"CAPTURING",(150,80), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        st = "dataset/"+dirr+"/img"+str(count)+".png"
        print("saving in ",st)
        cv.imwrite(st,crop)
        count += 1
        cv.imshow('Gesture', img)
    
    
    k = cv.waitKey(10)
    
    if k == 27:
        break
    elif k == ord("1"):
        count = 0
        dirr = pastas.pop()
        count = len(os.listdir("dataset/"+dirr))
        print("modified to ",dirr,"/ images:",count)
    elif k == ord("e"):
        MODE = not MODE
    elif k == ord("l"):
        laplacian_mode = not laplacian_mode
    elif k == ord("a"):
        ready = not ready

    if laplacian_mode:
        img = view_laplacian(img)

    cv.imshow('Gesture', img)    

cv.destroyAllWindows()
cap.release()    
