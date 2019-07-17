from torch2trt.torch2trt import *
from .Conv2d import *


@tensorrt_converter('torch.nn.functional.conv2d')
def convert_functional_conv2d(ctx):
    # print(ctx.method_args[0].shape, ctx.method_args[1].shape, ctx.method_args[2], ctx.method_args[3], ctx.method_args[4], ctx.method_args[5], ctx.method_args[6])
    input = ctx.method_args[0]
    kernel = ctx.method_args[1].detach().cpu().numpy()
  
    bias = trt.Weights(torch_dtype_to_trt(ctx.method_args[1].dtype))
    if ctx.method_args[2] is not None:
        bias = ctx.method_args[2].detach().cpu().numpy()

    stride = ctx.method_args[3]
    padding = ctx.method_args[4]
    dilation = ctx.method_args[5]
    groups = ctx.method_args[6]

    output = ctx.method_return

    kernel_size = kernel.shape[-2: ]
    out_channels = kernel.shape[0]

    layer = ctx.network.add_convolution(
        input=input._trt,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation
    
    if groups is not None:
        layer.num_groups = groups

    output._trt = layer.get_output(0)
