import cv2
import numpy as np


img = cv2.imread('./results/191.jpg', 0)
contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
area = []
topk_contours =[]
x, y = img.shape
for i in range(len(contours)):
    # 对每一个连通域使用一个掩码模板计算非0像素(即连通域像素个数)
    single_masks = np.zeros((x, y))
    fill_image = cv2.fillConvexPoly(single_masks, contours[i], 255)
    pixels = cv2.countNonZero(fill_image)
    area.append(pixels)
topk = 1 #取最大面积的个数
for i in range(1):
    top = area.index(max(area))
    area.pop(top)
    topk_contours.append(contours[top])
mask = np.zeros((x,y,3))
mask_img = cv2.drawContours(mask, topk_contours, -1, (255, 255, 255), 1)
cv2.imwrite('mask_img.png', mask_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.imshow('mask_img:', mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()