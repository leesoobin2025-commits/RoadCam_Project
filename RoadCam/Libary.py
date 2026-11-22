import numpy as np
import cv2
import matplotlib.pyplot as plt
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)