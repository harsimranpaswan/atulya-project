import cv2 as cv
from cv2 import BORDER_WRAP
from cv2 import BORDER_CONSTANT
import cv2.aruco as aruco
import numpy as np
import math

L=['Ha.jpg','HaHa.jpg','LMAO.jpg','XD.jpg']
diction = {}
r = cv.imread('CVtask.jpg')
new_img=cv.resize(r,(1750,1240))

def findAruco(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    key = getattr(aruco,f"DICT_5X5_250")
    arucoDict = aruco.Dictionary_get(key)
    p = aruco.DetectorParameters_create()
    c,i,r= cv.aruco.detectMarkers(gray,arucoDict,parameters=p)
    return (c,i,r)

def coordi(img):
    (c,i,r)=findAruco(img)
    if len(c)>0:
        i=i.flatten()
        for (markercorner,markerid) in zip(c,i):
            corner = markercorner.reshape((4,2))
            (topleft,topright,bottomright,bottomleft)=corner
            topleft = (int(topleft[0]),int(topleft[1]))
            topright = (int(topright[0]),int(topright[1]))
            bottomleft = (int(bottomleft[0]),int(bottomleft[1]))
            bottomright = (int(bottomright[0]),int(bottomright[1]))
        return topleft,topright,bottomright,bottomleft

def orient(img):
    topleft,topright,bottomright,bottomleft=coordi(img)
    cx = int((topleft[0]+bottomright[0])/2)
    cy = int((topleft[1]+bottomright[1])/2)
    px = int((topright[0]+bottomright[0])/2)
    py = int((topright[1]+bottomright[1])/2)
    m=(py-cy)/(px-cx)
    theta = math.atan(m)
    center = (cx,cy)
    cv.circle(img,topright,5,(0,255,0),-1)
    cv.circle(img,bottomright,5,(255,0,0),-1)
    cv.circle(img,(0,0),5,(0,0,255),-1)
    return center,(theta*180)/math.pi

def rotate_image(image, angle,center):
    rot_mat = cv.getRotationMatrix2D(center, angle,0.8)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR,borderMode=BORDER_CONSTANT,borderValue=(255,255,255))
    return result

def crop_img(img):
    topleft,topright,bottomright,bottomleft=coordi(img)
    l=[topleft,topright,bottomright,bottomleft]
    xmax=l[1][0]
    xmin=l[0][0]
    ymax=l[1][1]
    ymin=l[0][1]
    for i in l:
        if i[0]>xmax:
            xmax = i[0]
        if i[0]<xmin:
            xmin = i[0]
        if i[1]>ymax:
            ymax = i[1]
        if i[1]<ymin:
            ymin = i[1]
    print(xmax,xmin,ymax,ymin)
    t = img[ymin:ymax,xmin:xmax]
    return t


for i in L:
    x = cv.imread(i)
    (c,ids,r)=findAruco(x)
    diction[i]=ids
print(diction)

img = cv.imread('CVtask.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_,thresh = cv.threshold(gray,230,255,cv.THRESH_BINARY)
color={'green':[79,209,146],'orange':[9,127,240],'white':[210,222,228],'black':[0,0,0]}
color_id = {'green':1,'orange':2,'white':4,'black':3}

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
for c in contours:
    perimeter = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.01 * perimeter, True)

    if len(approx) == 4:
        
        x,y,w,h=cv.boundingRect(approx+50)
        aspectratio = float(w)/h
        if aspectratio >=0.95 and aspectratio<=1.05:
            print(approx)
            cord = [o[0].tolist() for o in approx]
            xmax=cord[0][0]
            xmin=cord[0][0]
            ymax=cord[0][1]
            ymin=cord[0][1]
            for i in cord:
                if i[0]>xmax:
                    xmax = i[0]
                if i[0]<xmin:
                    xmin = i[0]
                if i[1]>ymax:
                    ymax = i[1]
                if i[1]<ymin:
                    ymin = i[1]
            print(xmax,xmin,ymax,ymin)
            tow = img[ymin:ymax,xmin:xmax]
            shape1 = tow.shape
            new_shape = (shape1[1]-50,shape1[0]-50)
        
            m1=(int((cord[0][0]+cord[1][0])/2),int((cord[0][1]+cord[1][1])/2))
            c=(int((cord[0][0]+cord[2][0])/2),int((cord[0][1]+cord[2][1])/2))
            if (m1[0]-c[0]) != 0:
                theta = math.atan((m1[1]-c[1])/(m1[0]-c[0]))
            else :
                theta = math.pi/(-2)
            
            
            for i in color.keys():
                d = np.array(color[i])
                d.reshape((3,))
                if (d==img[c[1],c[0],:]).any():
                    wer = np.array(color_id[i])
                    wer.reshape((1,1))
                    for j in diction.keys():
                        if (wer==diction[j]).any():
                            sr = j
                    ar = cv.imread(sr)
                    c1,theta1=orient(ar)
                    
                    f=rotate_image(ar,theta1-(theta*180/math.pi),c1)
                    s=cv.resize(crop_img(f),new_shape)
                    
                    print(shape1,s.shape)
                    new_img[(ymin+25):(ymax-25),(xmin+25):(xmax-25),:]=s
                    
                    cv.waitKey(1)
                    cv.imshow('output',new_img)
                    cv.waitKey(1)

                    cv.putText(img,sr,c,cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))

            cv.circle(new_img,cord[0],5,(0,0,255),-1)
            #cv.drawContours(new_img,[approx], -1, (255, 0, 0), 3)
            print(cord[0][0]-cord[1][0])
            
cv.imwrite('output.jpg',new_img)
cv.waitKey(0)