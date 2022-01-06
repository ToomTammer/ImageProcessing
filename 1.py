import cv2 as cv

img1 = cv.imread('./Pbg.jpg')
img2 = cv.imread('./P1.jpg')
img3 = img2.copy()
img3 = cv.flip(img3, 1)

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
 
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)

img2_fg = cv.bitwise_and (img2, img2, mask = mask)

img3_fg = cv.bitwise_and (img3, img3, mask = mask)

img3_fg = cv.bitwise_and (img3, img3, mask = mask)


dst = cv.add(img1_bg,img2_fg)
dst2 = cv.add(img1_bg,img3_fg)
img1[0:rows, 137:cols+137] = dst
img1[0:rows, 60:cols+60] = dst2

img4 = cv.imread('./tiny.png')

rows, cols, channels = img4.shape
roi = img1[23:rows+23, 10:cols+10]

img4gray = cv.cvtColor(img4, cv.COLOR_BGR2GRAY)
ret4, mask4 = cv.threshold(img4gray, 10, 255, cv.THRESH_BINARY)
mask_inv4 = cv.bitwise_not(mask4)

img1_bg2 = cv.bitwise_and(roi, roi, mask = mask_inv4)

img4_fg = cv.bitwise_and (img4, img4, mask = mask4)

dst4 = cv.add(img1_bg2,img4_fg)
img1[23:rows+23, 10:cols+10] = dst4

cv.imshow('combine images',img1)


cv.waitKey(0)
cv.destroyAllWindows()
