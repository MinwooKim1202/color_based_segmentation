import cv2
import numpy as np
"""
fname_img1 = 'img/origin1.png'
fname_img2 = 'img/origin2.png'

f0202 = 'C:/Users/kmw26/Downloads/rgb/2022.02.02_0955.png'
f0206 = 'C:/Users/kmw26/Downloads/rgb/2022.02.06_1240.png'
f0210 = 'C:/Users/kmw26/Downloads/rgb/2022.02.10_0805.png'
f0211 = 'C:/Users/kmw26/Downloads/rgb/2022.02.11_1540.png'
f0214 = 'C:/Users/kmw26/Downloads/rgb/2022.02.14_1240.png'
f0214_1 = 'C:/Users/kmw26/Downloads/rgb/2022.02.14_1150.png'
"""

def get_mask(img_dir):
    contour_list = []

    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    img = cv2.resize(img, dsize=(1280, 720), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # hsv convert

    #cv2.imshow("hsv", hsv)

    h, s, v = cv2.split(hsv) # chnnel split

    h = cv2.inRange(h, 30, 100)
    s = cv2.inRange(s, 60, 255)
    v = cv2.inRange(v, 30, 255)

    #cv2.imshow("h", h)
    #cv2.imshow("s", s)
    #cv2.imshow("v", v)

    hsv_mask = cv2.bitwise_and(h, s) # hsv mask making
    hsv_mask = cv2.bitwise_and(hsv_mask, v)

    h,w = img.shape[0:2]
    cv2.rectangle(hsv_mask, (0,0), (w-1, h-1), (0,0,0), 1) # contour 생성시 non-close contour 생성 방지

    #cv2.imshow("hsv_mask", hsv_mask)

    hsv_mask_result = cv2.bitwise_and(hsv, hsv, mask = hsv_mask)

    rgb = cv2.cvtColor(hsv_mask_result, cv2.COLOR_HSV2BGR)
    #cv2.imshow("rgb", rgb)
    gray = cv2.cvtColor(hsv_mask_result, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    thr1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #cv2.imshow("thr1", thr1)
    blur = cv2.medianBlur(thr1, 7)
    #cv2.imshow("blur", blur)
    blur = cv2.bitwise_not(blur)
    #cv2.imshow("not", blur)
    kernel = np.ones((7,7),np.uint8)
    #blur = cv2.dilate(blur,kernel,iterations = 1)

    _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_index = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        #cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)  # blue
        contour_list.append(cnt)
        #cv2.putText(img, str(cnt_index), (cnt[0][0][0], cnt[0][0][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        cnt_index += 1


    #for cnt in label_list:
        #cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    #cv2.imshow('ori', img)
    #cv2.imshow('hsv_mask_result', hsv_mask_result)
    #cv2.imshow('blur', blur)
    #cv2.imshow('thr1', thr1)

    #cv2.waitKey(0)

    return contour_list
