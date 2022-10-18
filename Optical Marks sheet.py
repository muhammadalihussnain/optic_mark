import cv2 as cv
import numpy as np
import utils
import utils1

path=   "F:\opencvtest\Resources"
image   = cv.imread("Resources/Optical_marks_sheet.jpg")
image1  =   image.copy()
image1   =   utils1.preprocessing_image(image1, canyT=[90, 90])
list_rec_contours  =  utils1.get_contours(image1)
print(len(list_rec_contours))

order_points    =   utils1.reorder(list_rec_contours[0])
image_warp  = utils1.warp(image, order_points, pad=10)

warp_canny_image   =   utils1.preprocessing_image(image_warp, canyT=[90, 90])
list_rec_contours_solution  =  utils1.get_contours(warp_canny_image)
print(len(list_rec_contours_solution))
big_rectangle=list_rec_contours_solution[0]
big_rectangle   =   utils1.reorder(big_rectangle)
small_rectangle =   list_rec_contours_solution[1]
small_rectangle =   utils1.reorder(small_rectangle)
print(big_rectangle[0][0])

third_rectangle=list_rec_contours_solution[2]
third_rectangle   =   utils1.reorder(third_rectangle)
forth_rectangle =   list_rec_contours_solution[3]
forth_rectangle =   utils1.reorder(forth_rectangle)
print(big_rectangle[0][0])

cv.rectangle(image_warp, big_rectangle[0][0], big_rectangle[3][0], (255, 0, 0), 10)
cv.rectangle(image_warp, small_rectangle[0][0], small_rectangle[3][0], (255, 0, 0), 10)
cv.rectangle(image_warp, third_rectangle[0][0], third_rectangle[3][0], (255, 255, 0), 10)
cv.rectangle(image_warp, forth_rectangle[0][0], forth_rectangle[3][0], (255, 100, 100), 10)


#cv.rectangle(image,order_points[0][0], order_points[3][0], (0, 0, 255), 3)


image_warp  = utils1.warp(image, order_points, pad=10)
image_warp_copy =   image_warp.copy()

image1   =   utils1.preprocessing_image(image_warp_copy, canyT=[110 , 110])
list_rec_contours  =  utils1.get_contours(image1)

#big_rect_points, small_rect_points  =   list_rec_contours[0],list_rec_contours[1]

big_rect_points =   utils1.reorder(list_rec_contours[0])
print(big_rect_points)
cv.rectangle(image_warp,big_rect_points[0][0],big_rect_points[3][0],(0,0,255),3)

solution_copy   =   utils1.warp(image_warp, big_rect_points, pad=10)
cv.imshow("solution_copy", solution_copy)

small_rect_points =   utils1.reorder(list_rec_contours[1])
grad_image  = utils1.warp(image_warp, small_rect_points, pad=10)
cv.imshow("grad_copy", grad_image )

print(small_rect_points)
cv.rectangle(image_warp,small_rect_points[0][0],small_rect_points[3][0],(255, 0, 255),3)
cv.imshow("original_image", image)
cv.waitKey(0)
'''
cv.imshow("image", image)
cv.imshow("Warp_image", image_warp)
cv.waitKey(0)'''


