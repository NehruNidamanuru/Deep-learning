import os
os.chdir('C:\\Users\\nnidamanuru\\.spyder-py3')
from ssd import SSD300
import ssd_utils
import custom_generator as generator
import numpy as np

voc_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)

weights_dir = 'C:/Users/nnidamanuru/Downloads/yolo.weights'
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(weights_dir, by_name=True)

path = 'C:\\Users\\nnidamanuru\\Downloads\\Ge_videos\\data3\\1'
batch_size = 1
target_size = (300, 300)
nb_test_samples = 5
test_generator = generator.DirectoryIterator(directory=path,  class_mode=None,
                target_size=target_size, batch_size=batch_size,  shuffle=False)

preds = model.predict_generator(test_generator, nb_test_samples//batch_size)

print (np.shape(preds))
preds_1 = preds[0,0,:]
print (preds_1)

nms_thresh = 0.4
bbox_util = ssd_utils.BBoxUtility(NUM_CLASSES,nms_thresh = nms_thresh)
results = bbox_util.detection_out(preds)
print (np.shape(results))
print (results[0].shape)
print (results[0][0])

images = next(test_generator)
ssd_utils.display_boxes(images[0],results[4],0.4, voc_classes)

score_thresh = 0.4
for i, img in enumerate(next(test_generator)):
    ssd_utils.display_boxes(img,results[i],score_thresh,voc_classes)