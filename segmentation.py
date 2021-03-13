import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/cropped_image.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap='gray')
plt.show()
plt.cla()
_, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# plt.imshow(thresh,cmap='gray')
# plt.show()
# plt.cla()
# plt.cla()

# equ = cv2.equalizeHist(img)
# plt.imshow(equ,cmap='gray')
# plt.show()
# plt.cla()
# plt.cla()

bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cmap in (cv2.COLOR_BGR2HLS,cv2.COLOR_BGR2HSV,cv2.COLOR_BGR2LUV):
    other = cv2.cvtColor(bgr,cmap)
    plt.imshow(other,cmap='jet')
    plt.show()
    plt.cla()
    plt.cla()