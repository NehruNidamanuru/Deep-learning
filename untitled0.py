import os
import math
import cv2


n='images'
def get_store_frames(video_file, video_frames_dir):
     print(video_file, video_frames_dir)
     video = cv2.VideoCapture(video_file)
     #print(video.isOpened())
     framerate = video.get(cv2.CAP_PROP_FPS)
     os.makedirs(video_frames_dir)
     while (video.isOpened()):
         frameId = video.get(1)
         success,image = video.read()
         if(success == False):
             break
         if (frameId % math.floor(framerate) == 0):
                filename = os.path.join(video_frames_dir, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                print(filename)
                cv2.imwrite(filename,image)
     video.release()

def preapare_full_dataset_for_flow(train_dir_original,target_base_dir):
    train_dir = os.path.join(target_base_dir, 'train')
    
    categories = os.listdir(train_dir_original)
    print(categories)

    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
       
       
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            
            
            for t in train_files:
                get_store_frames(t, os.path.join(target_base_dir, n))

            
    else:
        print('required directory structure already exists. learning continues with existing data')

    nb_train_samples = 0  
    
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training samples:', nb_train_samples)
    
    return train_dir,nb_train_samples



train_dir,nb_train_samples = \
                        preapare_full_dataset_for_flow(train_dir_original='C:\\Users\\nnidamanuru\\Downloads\\GEA', 
                               target_base_dir='C:\\Users\\nnidamanuru\\Downloads\\GEA\\data3')