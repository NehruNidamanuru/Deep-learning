import cv2


img_path='C:\\Users\\nnidamanuru\\Pictures\\imt2.jpg'

img = cv2.imread(img_path)
print(img.shape)
scale_percent1 = 88.3
height = int(img.shape[0] * scale_percent1 / 100)
scale_percent2 = 150
width = int(img.shape[1] * scale_percent2 / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path,resized)


img_path1='C:\\Users\\nnidamanuru\\Downloads\\GEA_Task\\New folder\\box_5.jpg'

img1 = cv2.imread(img_path1)
print(img1.shape)

resized = cv2.resize(img1, (411,575), interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.imwrite(img_path1,resized)
