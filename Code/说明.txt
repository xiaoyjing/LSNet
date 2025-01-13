layers folder:
Contains the code for the IOU loss function, NMS (Non-Maximum Suppression) code, and the attention mechanism code.
modeling folder：
Contains the code for the model structure.
1）The backbone folder contains the code for the backbone network, such as Residual Networks (ResNet), Feature Pyramid Networks (FPN), and Bi-directional Feature Pyramid Networks (BiFPN).
2）The roi_heads folder contains the code for the attention mechanism output heads, used for the network's output part.
3）The Networks folder contains the code for the network structure, used to build the entire network.
predictor.py： data prediction.
test.py：data testing.
train_net.py：training the model.
