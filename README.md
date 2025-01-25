# LSNet
A Lightweight Segmentation Framework for Real-Time Wildfire Detection
# Highlights
1、We introduce a lightweight segmentation network, LSNet, consisting of an encoder and a decoder for real-time wildfire detection to minimize threats to both personal safety and the environment.\
2、We designed an encoder with a parallel feature multiplier (PFM) and a residual feature extractor (RFE) to reduce computational load and enhance segmentation speed, and a decoder with an attentional feature pyramid (AFP) to improve accuracy.\
3、Outperforming deep learning algorithm SOLOv2, the designed LSNet presents a viable solution for timely wildfire identification and localization, thus minimizing the risk to both personal safety and the environment.
# Contributions
This is Pytorch implementation for LSNet on fire-smoke dataset, which is mainly achieved by Cheng Zhang and Zhuoyue Ding. Some parts also benefit from Lei Huang.
# Abstract
Wildfires pose significant threats to both human safety and the environment, necessitating prompt detection and localization. Deep learning algorithms, particularly those for image segmentation, offer promising solutions but often suffer from computational delays that hinder real-time applications. In this paper, we introduce LSNet, a lightweight segmentation network designed for real-time wildfire detection. LSNet comprises an encoder with a parallel feature multiplier and a residual feature extractor to reduce computational load and enhance segmentation speed, and a decoder with an attentional feature-pyramid structure to improve accuracy. Experimental results demonstrate that LSNet achieves a speed of 31 FPS, an AP50 of 65.1%, and a model size of 274 MB during training, outperforming SOLOv2. Testing results further confirm its efficacy with 30 FPS and an AP50 of 72.3%, meeting the requirements for real-time wildfire detection. Thus, LSNet presents a viable solution for timely wildfire identification and localization.
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/1.png)\
Fig. 1. Schematic diagram of the LSNet for the real-time detection of wildfires.
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/3.png)
Fig. 2. Results of four wildfire images containing (a) large, (b) medium, (c) small, and (d) dense targets.
# Network
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/4.png)
Fig. 3. Schematic diagrams of: (a) encoder; (b) FEB; (c) decoder with AFPS; (d) CBAM.
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/2.png)
Fig. 4. Structure of the LSNet.
# Install(Same with Mask R-CNN)
The Environment：

- PyTorch 1.0 from a nightly release. It will not work with 1.0 nor 1.0.1. 
* cocoapi
* yacs
* matplotlib
* GCC >= 4.9
* OpenCV
+ CUDA >= 9.0

a. Clone the PointRCNN repository.
```
git clone https://github.com/xiaoyjing/LSNet/edit/master/EPNet.git
```
b. Install the dependent python libraries like tqdm, tensorboardX  etc.
# Dataset preparation
Please download the fire-smoke dataset and organize the downloaded files as follows:
```
LSNet
├── fire-segmentation data
│   ├── KITTI
│   │   ├── annotations
│   │   ├── things_train2017
│   │   ├── train2017
│   │   ├── val2017
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── Code
│   │   ├── layers
│   │   │      ├──bezier_align.py
│   │   │      ├──def_roi_align.py
│   │   │      ├──deform_conv.py
│   │   │      ├──gcn.py
│   │   │      ├──iou_loss.py
│   │   │      ├──ml_nms.py
│   │   ├── modeling
│   │   │      ├──backbone
│   │   │      |     ├──bifpn.py
│   │   │      |     ├──dla.py
│   │   │      |     ├──fpn.py
│   │   │      |     ├──lpf.py
│   │   │      |     ├──mobilenet.py
│   │   │      |     ├──resnet_interval.py
│   │   │      |     ├──resnet_lpf.py
│   │   │      |     ├──vovnet.py
│   │   │      ├──Networks
│   │   │      |     ├──blender.py
│   │   │      |     ├──nn.py
│   │   │      ├──roi_heads
│   │   │      |     ├──attn_predictor.py
│   │   ├── predictor.py
│   │   ├── test.py
│   │   ├── train_net.py
```
# Trained model
Table 1 Comparing LSNet with state-of-the-art networks in terms of AP50, speed, and model size.
| Network | Speed (FPS) | AP50 (%) | Model size (MB) |
| --- | --- | --- | --- |
| BoxInst | 20 | 60.7 | 406 |
| Mask R-CNN | 18 | 61.0 | 483 |
| SOLOv2 | 14 | 63.5 | 499 |
| E2EC | 30 | 62.9 | 113 |
| SOLQ | 12 | 60.5 | 476 |
| LSNet | 31 | 65.1 | 274 |

![img](https://github.com/xiaoyjing/LSNet/blob/master/img/5.png)
Fig. 5. (a) total loss and (b) mask loss of the LSNet compared with the other state-of-the-art networks.
# Implementation
## Training
Run LSNet for single gpu:
```
CUDA_VISIBLE_DEVICES=0 python train_netpy
```
Run LSNet for two gpu:
```
CUDA_VISIBLE_DEVICES=0,1 python train_netpy
```
## Testing
```
CUDA_VISIBLE_DEVICES=2 python test.py
```
# Acknowledgement
The code is based on [Mask R-CNN](https://gitcode.com/gh_mirrors/ma/maskrcnn-benchmark/tree/main).
# Citation
If you find this work useful in your research, please consider cite:
```
@article{Huang2025LSNet,
  title={LSNet: A Lightweight Segmentation Framework for Real-Time Wildfire Detection},
  author={Cheng Zhang and Zhuoyue Ding and Lei Huang},
  booktitle ={xxx},
  month = {Jan},
  year={2025}
}
```
