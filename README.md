# Pytorch model to TensorRT engine

This is a repo fork from Nvidia torch2trt

## Setup
```shell
git clone https://github.com/binzh93/torch2trt.git
cd torch2trt
sudo python setup.py install
```



## Support op

| support trt op | support torch op | Implementation Method |
|:-:|:-:|:-:|
| AdaptiveAvgPool2d | torch.nn.AdaptiveAvgPool2d | tensorrt.INetworkDefinition.add_pooling |
| add | torch.Tensor.__add__ | tensorrt.INetworkDefinition.add_elementwise| 
| AvgPool2d | torch.nn.AvgPool2d | tensorrt.INetworkDefinition.add_pooling |
| BatchNorm2d | torch.nn.BatchNorm2d | tensorrt.INetworkDefinition.add_scale |
| cat | torch.cat | tensorrt.INetworkDefinition.add_concatenation |
| Conv2d | torch.nn.Conv2d | tensorrt.INetworkDefinition.add_convolution	|
| ConvTranspose2d | torch.nn.ConvTranspose2d | tensorrt.INetworkDefinition.add_deconvolution |
| iadd | torch.Tensor.__iadd__ | tensorrt.INetworkDefinition.add_elementwise |
| Identity | torch.nn.Dropout  torch.nn.Dropout2d  torch.nn.Dropout3d| / |
| identity | torch.Tensor.contiguous  torch.nn.functional.dropout  torch.nn.functional.dropout2d  torch.nn.functional.dropout3d | /|
| Linear | torch.nn.Linear | tensorrt.INetworkDefinition.add_fully_connected tensorrt.INetworkDefinition.add_shuffle | 
| LogSoftmax | torch.nn.LogSoftmax | tensorrt.INetworkDefinition.add_softmax  tensorrt.INetworkDefinition.add_unary |
| MaxPool2d	| torch.nn.MaxPool2d | tensorrt.INetworkDefinition.add_pooling	|
| mean | torch.mean  torch.Tensor.mean | tensorrt.INetworkDefinition.add_reduce |
| relu | torch.nn.functional.relu | / |
| relu6	| torch.nn.functional.relu6 | / |
| transpose | torch.transpose | tensorrt.INetworkDefinition.add_shuffle  (tensorrt.IShuffleLayer) | 
| view | torch.Tensor.reshape  torch.Tensor.view | tensorrt.INetworkDefinition.add_shuffle	|
| interpolate | torch.nn.functional.interpolate	| self define plugin |
| multiply | torch.Tensor.__mul__ |  tensorrt.INetworkDefinition.add_elementwise |
| pad | torch.nn.functional.pad | tensorrt.INetworkDefinition.add_padding	|
| sigmoid | torch.sigmoid  torch.nn.functional.sigmoid  torch.nn.Sigmoid | tensorrt.INetworkDefinition.add_activation |
| squeeze | torch.squeeze  torch.Tensor.squeeze | tensorrt.INetworkDefinition.add_shuffle	|










