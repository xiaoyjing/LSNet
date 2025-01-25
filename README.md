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
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/1.png)

Fig. 1. Schematic diagram of the LSNet for the real-time detection of wildfires.
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/3.png)
Fig. 2. Results of four wildfire images containing (a) large, (b) medium, (c) small, and (d) dense targets.
# Network
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/4.png)
Fig. 3. Schematic diagrams of: (a) encoder; (b) FEB; (c) decoder with AFPS; (d) CBAM.
![img](https://github.com/xiaoyjing/LSNet/blob/master/img/2.png)
Fig. 4. Structure of the LSNet.



This repository is a full implementation of LSNet:A Lightweight Segmentation Framework for Real-Time Wildfire Detection, building a new real-time wildfire segmentation framework for wildfire detection.
This repository demonstrates the architecture of the LSNet model and the effectiveness and robustness of the scheme. Our implementation supports both single and multi-GPU training.
