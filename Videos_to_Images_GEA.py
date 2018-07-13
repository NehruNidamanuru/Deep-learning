import os
import math
import cv2


def get_store_frames(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
    os.makedirs(target_dir)
    video_listings = os.listdir(base_dir)
    print(video_listings)
    count=0
    for file in video_listings[0:len(video_listings)]:
        print(file)
        vidcap = cv2.VideoCapture(base_dir+'/'+file)
        framerate = vidcap.get(cv2.CAP_PROP_FPS)
        while (vidcap.isOpened()):
            frameId = vidcap.get(1)
            success,image = vidcap.read()
            if(success == False):
                break
            if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(target_dir, "image_" + str(count) + ".jpg")
                    count+=1
                    print(filename)
                    cv2.imwrite(filename,image)
        vidcap.release()
    return

get_store_frames(base_dir='C:\\Users\\nnidamanuru\\Downloads\\Bander', 
                  target_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA_3\\Images_fps')
