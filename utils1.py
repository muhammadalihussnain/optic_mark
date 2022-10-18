import cv2 as cv
import numpy as np

kernel  =   np.ones((5, 5), np.uint8)
cornor=0


def preprocessing_image(image,canyT):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurImage = cv.GaussianBlur(grayImage, (5, 5), 1)
    cannyImage = cv.Canny(blurImage, canyT[0], canyT[1])

    return cannyImage

def get_contours(image):
    rect_contours   = []
    list_areas_contours=[]
    i=0
    contours, heirarchy     =   cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        peri = cv.arcLength(contour,closed=True)
        edge = cv.approxPolyDP(contour, 0.02 * peri, True)
        if len(edge) == 4:
            i = i+1
            rect_contours.append(edge)
    rect_contours.sort( key=cv.contourArea, reverse=True)

    return rect_contours



def reorder(mypoints):
    print(f"shape of edges is {mypoints.shape}")
    new_points  =   np.zeros_like(mypoints)
    mypoints    =   np.reshape(mypoints, (4, 2))

    add = np.sum(mypoints,  axis=1)
    sub = np.diff(mypoints, axis=1)
    new_points[0]    =   mypoints[np.argmin(add)]
    new_points[3] =      mypoints[np.argmax(add)]

    new_points[1] = mypoints[np.argmin(sub)]
    new_points[2] = mypoints[np.argmax(sub)]
    return new_points



def warp(image,mypoints,pad=10):
    #arrange_points  =   reorder(mypoints)
    height = 450
    width = 350
    point_1 = np.float32([mypoints[0], mypoints[1], mypoints[2], mypoints[3]])
    point_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(point_1, point_2)

    outputImage = cv.warpPerspective(image, matrix, (width, height))
    outputImage=outputImage[pad:outputImage.shape[0]-pad,pad:outputImage.shape[1]-pad]

    return outputImage



