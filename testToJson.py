import json
from mmdet.apis import init_detector, inference_detector
import os
from pycocotools.coco import COCO
from tqdm import tqdm

# 提交要求
# [{
#     "image_id": int,
#     "category_id": int,
#     "bbox": [x_min,y_min,width,height],
#     "score": float,
# }]

filename_id = {}

# 获取文件名与对应的 ID
# 如 {'000000.jpg': 12}

# 目标检测配置文件
config_file = 'configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
# 训练模型
checkpoint_file = 'work_dirs/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/epoch_24.pth'

# 配置模型
model = init_detector(config=config_file,
                      checkpoint=checkpoint_file,
                      device='cuda:0')

# 需要推理的图片的路径
root = 'data/coco/test2017'

# 存储结果，并生成 json
results = []

# 开始推理
for file in tqdm(os.listdir(root)):
    result = inference_detector(model=model, imgs=root +'/'+ file)
    print(file)
    for cate, items in enumerate(result, 1):
        print(cate,' ',items)
        # 同一类别的有很多结果
        for item in items:
            item = item.tolist()
            if item!=[]:
                x, y, w, h, s = round(item[0],2), round(item[1],2), round(item[2],2), round(item[3],2), item[4]
                d = {}
                # if file in filename_id:
                d['image_id'] = file
                d['category_id'] = cate
                d['bbox'] = [x, y, w, h]
                d['score'] = s
                results.append(d)

# 保存
print(results)
with open('mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.json', 'w') as f:
    json.dump(results, f, indent=4)