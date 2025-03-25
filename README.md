# LSNet
Lightweight Segmentation Network for Real-Time Wildfire Detection: LSNet's Parallel Feature Multiplication and Attentional Fusion
# Highlights
1、We introduce a lightweight segmentation network, LSNet, consisting of an encoder and a decoder for real-time wildfire detection to minimize threats to both personal safety and the environment.\
2、We designed an encoder with a parallel feature multiplier (PFM) and a residual feature extractor (RFE) to reduce computational load and enhance segmentation speed, and a decoder with an attentional feature pyramid (AFP) to improve accuracy.\
3、Outperforming deep learning algorithm SOLOv2, the designed LSNet presents a viable solution for timely wildfire identification and localization, thus minimizing the risk to both personal safety and the environment.
# Contributions
This is Pytorch implementation for LSNet on fire-smoke dataset, which is mainly achieved by Cheng Zhang and Zhuoyue Ding. Some parts also benefit from Lei Huang.
# Abstract
Wildfires pose significant threats to human safety and the environment, necessitating prompt detection and localization. Deep learning algorithms, particularly those for image segmentation, show promise but often struggle with computational delays, hindering real-time applications. This paper introduces LSNet, a lightweight segmentation network tailored for real-time wildfire detection. LSNet comprises an encoder equipped with a parallel feature multiplier and a residual feature extractor to reduce computational load and expedite segmentation. The decoder incorporates an attentional feature pyramid to enhance segmentation accuracy. Experimental results demonstrate that LSNet achieves a speed of 31 FPS, an AP50 of 65.1%, and a model size of 274 MB during training, outperforming both CNN-based SOLOv2 and transformer-based SegFormer. Testing results further confirm its efficacy with 30 FPS and an AP50 of 72.3%, meeting the demands of real-time wildfire detection. Thus, LSNet presents a viable solution for timely wildfire identification and localization. Source code is available at: https://github.com/xiaoyjing/LSNet.
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/1.png)\
Fig. 1. Schematic diagram of the LSNet for the real-time detection of wildfires.
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/3.png)
Fig. 2. Results of four wildfire images containing (a) large, (b) medium, (c) small, and (d) dense targets.
# Network
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/4.png)
Fig. 3. Schematic diagrams of: (a) encoder; (b) FEB; (c) decoder with AFP; (d) CBAM..
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
* numpy
* scipy
* Pillow
* cython
* scikit-image
* tensorflow>=1.3.0
* keras>=2.0.8
* opencv-python
* h5py
* imgaug
* IPython[all]
+ CUDA >= 9.0

a. Clone the PointRCNN repository.
```
git clone https://github.com/xiaoyjing/LSNet.git
```
b. Install dependencies.
```
pip3 install -r requirements.txt
```
c. Run setup from the repository root directory.
```
python3 setup.py install
```
# Dataset preparation
Please download the fire-smoke dataset and organize the downloaded files as follows:
```
LSNet
├── fire-segmentation data
│   ├── annotations
│   │   │      ├──instances_train2017.json
│   │   │      ├──instances_val2017.json
│   ├── things_train2017
│   │   │      ├──1.npz & 2.npz & ... & 770.npz
│   ├── train2017
│   │   │      ├──1.jpg & 2.jpg & ... & 770.jpg
│   ├── val2017
│   │   │      ├──1.png & 2.png & ... & 330.png
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
│   │   ├── requirements.txt
│   │   ├── setup.py
```
* Directions: **annotations**：xxx.json contains the labeled data information for the training set and validation set.
  **thing_train2017**:xxx.npz is used to store the data information of the training dataset, stored in a numpy array format.
  **train2017**:Contains the training dataset.
  **val2017**:Contains the testing dataset.
# Trained model
Table 1 Comparing LSNet with state-of-the-art networks in terms of AP50, speed, and model size.
| Network | Speed (FPS) | AP50 (%) | APs (%) | mIoU (%) | Model size (MB) |
| --- | --- | --- | --- | --- | --- |
| BoxInst | 20 | 60.7 | 15.1 | 80.1 | 406 |
| Mask R-CNN | 18 | 61.0 | 17.2 | 80.6 | 483 |
| SOLOv2 | 14 | 63.5 | 19.3 | 81.7 | 499 |
| Swin Transformer | 24 | 65.4 | 20.5 | 80.8 | 229 |
| SegFormer | 30 | 64.9 | 20.2 | 83.4 | 157 |
| LSNet | 31 | 65.1 | 20.7 | 84.5 | 274 |

![img](https://github.com/xiaoyjing/LSNet/blob/master/img/7.png)
Fig. 5. (a) total loss and (b) mask loss of the LSNet compared with the other state-of-the-art networks.

Table 2 Training results of ablation experiments.
| Network | Speed (FPS) | AP50 (%) | APs (%) | Model size (MB) |
| --- | --- | --- | --- | --- |
| LSNet_R50 | 20 | 58.3 | 17.3 | 189 |
| LSNet_R101 | 11 | 64.2 | 18.0 | 364 |
| LSNet_RP | 32 | 59.1 | 17.8 | 227 |
| LSNet_RA | 16 | 63.7 | 19.8 | 257 |
| LSNet | 31 | 65.1 | 20.7 | 274 |

Figure 6 shows the influence of the number of slices on speed and AP50, where the number of slices is proportional to the speed and the largest AP50 is 65.1% when the number of slices equals 4. Therefore, the number of slices M was set to 4.\
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/6.png)\
Fig. 6. Influence of the number of slices on speed and AP50.
# Implementation
## Training
Run LSNet for single gpu:
```
CUDA_VISIBLE_DEVICES=0 python train_net.py
```
Run LSNet for two gpu:
```
CUDA_VISIBLE_DEVICES=0,1 python train_net.py
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
@article{
  title={Lightweight Segmentation Network for Real-Time Wildfire Detection: LSNet's Parallel Feature Multiplication and Attentional Fusion},
  author={Cheng Zhang and Zhuoyue Ding and Xiaoying Jing and Lei Huang and Run Ye and Bin Yan and Xiaojia Zhou and Jinhong Guo},
  booktitle ={The Visual Computer},
  month = {Jan},
  year={2025}
}
```
