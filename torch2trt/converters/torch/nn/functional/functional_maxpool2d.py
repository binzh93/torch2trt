from torch2trt.torch2trt import *

@tensorrt_converter('torch.nn.functional.max_pool2d')
def convert_functional_max_pool2d(ctx):
    parameters = {}
    kernel_size = 3
    stride = 1
    padding = 0
    input = ctx.method_args[0]
    if(len(ctx.method_args)==2):
        kernel_size = ctx.method_args[1]
    elif(len(ctx.method_args)==3):
        kernel_size = ctx.method_args[1]
        stride = ctx.method_args[2]
    elif(len(ctx.method_args)==4):
        kernel_size = ctx.method_args[1]
        stride = ctx.method_args[2]
        padding = ctx.method_args[3]
    else:
        parameters = ctx.method_kwargs
    output = ctx.method_return

    if('kernel_size' in parameters.keys()):
        kernel_size = parameters['kernel_size']
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    if('stride' in parameters.keys()):
        stride = parameters['stride']
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    if('padding' in parameters.keys()):
        padding = parameters['padding']
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling(
        input=input._trt, type=trt.PoolingType.MAX, window_size=kernel_size)
    layer.stride = stride
    layer.padding = padding

    # add ceil_mode, ceil_mode=False by default in torch
    if('ceil_mode' in parameters.keys()):
        layer.padding_mode = trt.PaddingMode.SAME_UPPER

    output._trt = layer.get_output(0)