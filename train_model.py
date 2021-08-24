import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

pastas = os.listdir("gestures")
dataset = []
classes = []
temp = [0 for i in range(len(pastas))]

for i,pasta in enumerate(pastas):
    images = os.listdir("gestures/"+pasta)
    copy = temp.copy()
    copy[i] = 1
    for img in images:
        st = "gestures/"+pasta+"/"+img
        mr = cv2.imread(st)
        dataset.append(mr)
        classes.append(copy.copy())
    copy.clear()