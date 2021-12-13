import cv2
from selectpolygon import croppoly, croppolyinverse, selectpoly
import numpy as np
import random

def inArea(dimensions,point):
    imW=dimensions[0]
    imH=dimensions[1]

    img=np.ones((imW,imH,3), np.uint8)*150
    imgdraw=np.zeros((imW,imH,3), np.uint8)#*255

    lPoints=[(0,imH),(int(imW/2-imW*.05),int(imH*.1))]
    rPoints=[(imW,imH),(int(imW/2+imW*.05),int(imH*.1))]
    totalpoints=[lPoints[0],lPoints[1],rPoints[1],rPoints[0]]
    #_,totalpoints=selectpoly(img,False)

    '''
    #draw debug
    img = cv2.line(img, lPoints[0], lPoints[1], (0,0,255), 1)
    img = cv2.line(img, rPoints[0], rPoints[1], (0,0,255), 1)
    img = cv2.circle(img, point, 7, (255,0,0), 2)
    '''

    imgTop=croppoly(imgdraw,totalpoints,False)
    imgBottom=croppolyinverse(imgdraw,totalpoints,False)

    totalimg=cv2.bitwise_and(imgTop,imgBottom)
    #print(totalimg[point[1]][point[0]])
    if list(totalimg[point[1]][point[0]])==[255,255,255]:
        return False
    else:
        return True
    '''
    if totalimg[point[0]][point[1]]==(0,0,0):
        print("black")
    else:
        print("white")
    '''
    '''
    cv2.imshow("afterCrop",totalimg)
    #color=totalimg[y][x]#(0,0,0)
    cv2.imshow("beforeCrop",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

'''
w=300
h=300
point=(int(random.randrange(0,w)),int(random.randrange(0,h)))
print(inArea((w,h),point))
'''
