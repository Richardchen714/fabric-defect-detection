# 布匹瑕疵检测
[TOC]
## 安装
由于最新版本（v2.25.0）存在bug导致安装失败，本次实验使用 v2.24.1。使用 `git clone` 命令后按照官方文档操作即可。

## 准备工作

### 修改源代码
由于本实验中瑕疵共有20个分类，在执行任务前需要修改源代码，将所有 `num_classes=80` 改成 `num_classes=20` 并将 COCO 数据集类改为如下内容

```python
def coco_classes():
    return [
        '1','2','3','4','5','6','7','8','9','10',
        '11','12','13','14','15','16','17','18','19','20'
    ]
```

### 准备数据集
将数据集按如下方式组织，其中 `annotations` 存储测试集与验证集的 `JSON` 标注文件，`test2017`, `train2017`, `val2017` 分别存放测试集、训练集和验证集。
![Dataset](./assets/Dataset.png)

## 训练

训练 `yolox_tiny_8x8_300e` 模型，运行如下代码。如需训练其他模型，则更改第二个文件路径。

```sh
python tools/train.py configs/yolox yolox_tiny_8x8_300e_coco.py
```

训练结束后，`mmdetection` 目录下的 `work_dirs` 文件会出现文件夹（名为训练的模型），内有记录训练时每个 epoch 后的 loss 等信息的 `log` 和 `JSON` 文件，以及每个 epoch 的 `.pth` 神经网络文件。

### 绘制 loss 曲线

MMDetection 可根据训练后生成的 `JSON` 文件，运行如下命令即可。
```bash
python tools/analysis_tools/analyze_logs.py plot_curve \
       work_dirs/yolox_tiny_8x8_300e_coco/20220609_150435.log.json \ 
       --keys loss --out ./plot_result/4.png

```

下图为绘制的 Faster_RCNN 模型的 loss 曲线。
![loss](./assets/1.png)

## 测试

提交的源代码中的 `test.py` 可测试模型，并生成复合提交格式的测试结果的 `JSON` 文件。运行如下命令即可。第二个参数为模型的 Python 源代码，第三个参数为训练得到的 `.pth` 神经网络文件。

```bash
python tools/test.py \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_24.pth --eval bbox --show
```

## 参考内容
https://mmdetection.readthedocs.io/en/stable/index.html

https://blog.csdn.net/qq_33897832/article/details/103995636

https://blog.csdn.net/weixin_45734379/article/details/112725000

https://cloud.tencent.com/developer/article/1771899

Loss Visualization https://blog.csdn.net/kellyroslyn/article/details/110086658