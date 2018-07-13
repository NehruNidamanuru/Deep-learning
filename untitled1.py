from PIL import Image
import cv2
import os



#Train a classifier on the full object in following cropped coordinates for belt 2:

#Scenario 1: 732,168,1405,1053

x1 = 732
y1 = 168
x2 = 1405
y2 = 1053



path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario1\\3'
 
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
   
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 



#Scenario 2: 657,109,1431,1078

x1 = 657
y1 = 109
x2 = 1431
y2 = 1078

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario2\\3'
 
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
   
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 




#Train a classifier on the full object in following cropped coordinates for belt 1:

#Scenario 3: 484,314,1063,1077
x1 = 484
y1 = 314
x2 = 1063
y2 = 1077

path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario3\\3'
 
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
   
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

#Scenario 4: 445,253,1063,1077
x1 = 445
y1 = 253
x2 = 1063
y2 = 1077
path2='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4\\3\\video_1'
path4='C:\\Users\\nnidamanuru\\Downloads\\Ne1\\Scenario4\\3'
 
for i in os.listdir(path2):
    path3=os.path.join(path2,i)
    img=cv2.imread(path3)
    crop_img = img[y1:y2, x1:x2]
    path5=os.path.join(path4,i)
   
    cv2.imwrite(path5,crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 



##actual code
in_path='C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data4'
os.listdir(in_path)
for i in os.listdir(in_path):
    in_path1=os.path.join(in_path,i)
    for i in os.listdir(in_path1):
        in_path1=os.path.join(in_path1,i)
        for i in os.listdir(in_path1):
            path3=os.path.join(in_path1,i)
            img=cv2.imread(path3)
            crop_img = img[y1:y2, x1:x2]
            path5=os.path.join(path4,i)
   
            cv2.imwrite(path5,crop_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    
path1 = 'C:\\Users\\nnidamanuru\\Downloads\\neh\\test\\image_62.jpg'
img=cv2.imread(path1)

# Croping
crop_img = img[y1:y2, x1:x2]
cv2_im = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
pil_im=Image.fromarray(cv2_im)
pil_im.show()


cv2.imshow("cropped", crop_img)
cv2.imwrite('C:\\Users\\nnidamanuru\\Downloads\\New folder\\image_62.jpg',crop_img)
cv2.waitKey(0)
