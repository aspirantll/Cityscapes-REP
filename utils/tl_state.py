import glob

import cv2
import random
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def detectColor(image): # receives an ROI containing a single light
    # convert BGR image to HSV
    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # min and max HSV values
    red_min = np.array([0,5,150])
    red_max = np.array([8,255,255])
    red_min2 = np.array([175,5,150])
    red_max2 = np.array([180,255,255])

    yellow_min = np.array([20,5,150])
    yellow_max = np.array([30,255,255])

    green_min = np.array([35,5,150])
    green_max = np.array([90,255,255])

    # apply red, yellow, green thresh to image
    # 利用cv2.inRange函数设阈值，去除背景部分
    red_thresh = cv2.inRange(hsv_img,red_min,red_max)+cv2.inRange(hsv_img,red_min2,red_max2)
    yellow_thresh = cv2.inRange(hsv_img,yellow_min,yellow_max)
    green_thresh = cv2.inRange(hsv_img,green_min,green_max)

    # apply blur to fix noise in thresh
    # 进行中值滤波
    red_blur = cv2.medianBlur(red_thresh,5)
    yellow_blur = cv2.medianBlur(yellow_thresh,5)
    green_blur = cv2.medianBlur(green_thresh,5)

    # checks which colour thresh has the most white pixels
    red = cv2.countNonZero(red_blur)
    yellow = cv2.countNonZero(yellow_blur)
    green = cv2.countNonZero(green_blur)

    # the state of the light is the one with the greatest number of white pixels
    lightColor = max(red,yellow,green)

    # pixel count must be greater than 60 to be a valid colour state (solid light or arrow)
    # since the ROI is a rectangle that includes a small area around the circle
    # which can be detected as yellow
    if lightColor > 20:
        if lightColor == red:
            return 0
        elif lightColor == yellow:
            return 1
        elif lightColor == green:
            return 2
    else:
        return 3

class TLState(Enum):
    red = 0
    yellow = 1
    green = 2
    off = 3

class TLType(Enum):
    regular = 0
    five_lights = 1
    four_lights = 2

def imgResize(image, height, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # calculate the ratio of the height and construct the dimensions
    r = height / float(h)
    dim = (int(w * r), height)
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def detectState(image, TLType):
    image = imgResize(image, 200)
    (height, width) = image.shape[:2]
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 霍夫圆环检测
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=15,maxRadius=30)
    overallState = 0
    stateArrow = 0
    stateSolid = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            if i[1] < i[2]:
                i[1] = i[2]
            roi = image[(i[1]-i[2]):(i[1]+i[2]),(i[0]-i[2]):(i[0]+i[2])]
            color = detectColor(roi)
            if color > 0:
                if TLType == 1 and i[0] < width/2 and i[1] > height/3:
                    stateArrow = color
                elif TLType == 2:
                    stateArrow = color
                    if i[1] > height/2 and i[1] < height/4*3:
                        stateArrow = color + 2
                else:
                    stateSolid = color

    if TLType == 1:
        overallState = stateArrow + stateSolid + 1
    elif TLType == 2:
        overallState = stateArrow + 7
    else:
        overallState = stateSolid

    return overallState

def plot_light_result(images):

    for i, image in enumerate(images):
        plt.subplot(1, len(images), i+1)
        lena = cv2.imread(image)
        label = TLState(detectState(cv2.imread(image),TLType.regular.value)).name
        cv2.imshow(label, lena)
        cv2.waitKey()


if __name__ == '__main__':
    light_path = list(glob.glob('/media/liulei/Data/dataset/bigdata/traffic_lights/train/0/*.jpg'))
    plot_light_result(light_path)
