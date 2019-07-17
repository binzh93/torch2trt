from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    input = ctx.method_args[0]
    args = ctx.method_args[1]
    output = ctx.method_return

    pre_w, post_w, pre_h, post_h = args
    layer = ctx.network.add_padding(input._trt, [pre_h, pre_w], [post_h, post_w])
    output._trt = layer.get_output(0)

