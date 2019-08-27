# Pytorch model to TensorRT engine

This is a repo fork from Nvidia torch2trt

# Requirements 
```
TensorRT 5.1.5
python3
```

## Setup1
```shell
git clone https://github.com/binzh93/torch2trt.git
cd torch2trt
python3 setup.py install
```

## Setup2(Plugins)
1. To install plugins, call the following
```
apt-get install libprotobuf* protobuf-compiler ninja-build
```  

2. protobuf(3.9)
method(official): https://github.com/protocolbuffers/protobuf/blob/master/src/README.md  

3. gcc 4.8.5  
```Shell
apt-get install -y gcc-4.8
apt-get install -y g++-4.8
```
```
cd /usr/bin
rm gcc
ln -s gcc-4.8 gcc 
rm g++
ln -s g++-4.8 g++
```
4. build plugin    
```Shell
python setup.py install --plugins
```


* Differences with official code: 
    * add protobuf lib path to rule link of build.py   
    * add a -std=c++11 flag to rule cxx  

Check the plugin
```Shell
python3 -m torch2trt.test --name=interpolate
```


## Usage
```python
from torch2trt import torch2trt
from torchvision.models.resnet import resnet18

# create some regular pytorch model...
model = resnet18(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))

```


## Support op

| Support trt op | Support torch op | Implementation method |
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
| mul | torch.Tensor.__mul__ |  tensorrt.INetworkDefinition.add_elementwise |
| pad | torch.nn.functional.pad | tensorrt.INetworkDefinition.add_padding	|
| sigmoid | torch.sigmoid  torch.nn.functional.sigmoid  torch.nn.Sigmoid | tensorrt.INetworkDefinition.add_activation |
| squeeze | torch.squeeze  torch.Tensor.squeeze | tensorrt.INetworkDefinition.add_shuffle	|
| functional_conv2d | torch.nn.functional.conv2d | tensorrt.INetworkDefinition.add_convolution	|










