import cv2 as cv
import numpy as np
kernel  =   np.ones((5, 5),np.uint8)
cornor_of_paper=0



'''def max_area(contours):
    for cont in contours:
        area=cv.contourArea(cont)
    return np.max(area)'''



def getContours(image, draw=True,canny=False,cannyT=[200,200],max_area=70000):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurImage = cv.GaussianBlur(grayImage, (5, 5), 1)

    cannyImage = cv.Canny(blurImage, cannyT[0],cannyT[1])
    imageDial = cv.dilate(cannyImage, kernel, iterations=2)
    erodeImage = cv.erode(imageDial, kernel, iterations=1)
    if canny: cv.imshow("canny_image",erodeImage)


    contours,  heirarchy    =   cv.findContours(erodeImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area    =   cv.contourArea(contour)
        peri = cv.arcLength(contour, True)
        if area>max_area:
            edges   =   cv.approxPolyDP(contour, 0.02*peri, True)
            if len(edges)==4:
                cornor_of_paper=edges
                bbox= cv.boundingRect(edges)
                if draw:
                    cv.rectangle(image, (bbox[0], bbox[1]),
                                         (bbox[0]+bbox[2], bbox[1]+
                                          bbox[3]), (0, 255, 0), 4)

    return image, bbox, cornor_of_paper

def reorder(mypoints):
    new_points  =   np.zeros_like(mypoints)
    np.reshape(mypoints, ( 4, 2))

    add = mypoints.sum(axis=1)
    sub = np.diff(mypoints,axis=1)
    new_points[0]    =   mypoints[np.argmin(add)]
    new_points[3] = mypoints[np.argmax(add)]

    new_points[1] = mypoints[np.argmin(sub)]
    new_points[2] = mypoints[np.argmax(sub)]
    return new_points


def warp(image,mypoints,pad=10):
    arrange_points  =   reorder(mypoints)
    height = 450
    width = 350
    point_1 = np.float32([arrange_points[0], arrange_points[1], arrange_points[2], arrange_points[3]])
    point_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(point_1, point_2)

    outputImage = cv.warpPerspective(image, matrix, (width, height))
    outputImage=outputImage[pad:outputImage.shape[0]-pad,pad:outputImage.shape[1]-pad]

    return outputImage
def findDis(points):

    #print(points.shape)
    points= points.reshape(4,2)
    x   = round(((((points[0][0]-points[1][0])**2) + (points[0][1] - points[1][1])**2)**0.5)/10)
    y = round(((((points[0][0]-points[2][0])**2) + (points[0][1] - points[2][1])**2)**0.5)/10)
    print("width of the card is ",x)
    print("heights of object is ",y)




