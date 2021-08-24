import threading
import cv2 as cv
import numpy as np
import pickle
import time
import os

cap = cv.VideoCapture(0)

pastas = sorted(os.listdir("DatasetBBOX"))
pastas = pastas[::-1]

dirr = pastas.pop()
print(dirr)
count = len(os.listdir("DatasetBBOX/"+dirr))

ready = False
MODE = True



#480-250, 640-150
def draw_right_rect(img):
    cv.rectangle(img,(490-1,0),(640,250+2),(0,255,0),0)
    return 490,1

def timeout():
    global RUNNING,ready,TIMEOUT,x,y,window
    while RUNNING:
        if ready:
            while True:
                time.sleep(1)
                TIMEOUT += 1
                if TIMEOUT >= 7:
                    break
            ready = False
            TIMEOUT = 0
            PASS = True
            x += 50
            if x+window[0] >= 640:
                x = 0
                y += 50
            if y+window[1] >= 480:
                x = 0
                y = 0
            print("Updated coords",(x,y))

RUNNING = True
TIMEOUT = 0
laplacian_mode = False
window = (250,250)
x = y = 0
BBOXS = []
color = (0,0,255)

th = threading.Thread(target=timeout)
th.daemon = True
th.start()
while(cap.isOpened()):
    ret, img = cap.read()

    if ready:
        crop = img.copy()
        cv.putText(img,"CAPTURING time: "+str(TIMEOUT+1),(150,80), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        st = "DatasetBBOX/"+dirr+"/img"+str(count)+".png"
        print("saving in ",st)
        cv.imwrite(st,crop)
        count += 1
        BBOXS.append((x,y,x+window[0],y+window[1]))
    
    k = cv.waitKey(10)
    if ready:
        color = (0,255,0)
    else:
        color = (0,0,255)

    cv.rectangle(img,pt1=(x,y),pt2=(x+window[0],y+window[1]),color=color,thickness=2)
        
    
    if k == 27:
        break
    elif k == ord("1"):
        with open("BBOXS/("+dirr+")bboxs.pkl","wb") as f:
            pickle.dump(BBOXS,f)
        count = 0
        dirr = pastas.pop()
        count = len(os.listdir("DatasetBBOX/"+dirr))
        print("modified to ",dirr,"/ images:",count)
        
        BBOXS.clear()
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
