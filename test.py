import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector


config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_1.pth'

model = init_detector(config_file,checkpoint_file)

filePath = 'data/coco/test2017'
out_dir = 'results/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)



results = []

for dirpath, dirnames, filenames in os.walk(filePath):
    # print(dirpath) # 'data/coco/test2017'
    # print(dirnames) # []
    # print(filenames) # ['xxx.jpg','yyy.jpg']
    for filename in filenames:
        # print(filename) # 'xxx.jpg'
        img=dirpath+'/'+filename
        # print(filepath)
        result = inference_detector(model,img)
        print(result)
        results.append(result)

# print('\nwriting results to {}'.format('faster_voc.pkl'))
# mmcv.dump(results,out_dir+'faster_voc.pkl')
