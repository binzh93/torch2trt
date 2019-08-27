from torch2trt.torch2trt import *
from .Sigmoid import *

@tensorrt_converter('torch.sigmoid')
@tensorrt_converter('torch.nn.functional.sigmoid')
def convert_sigmoid(ctx):
    ctx.method_args = (torch.nn.Sigmoid(),) + ctx.method_args
    convert_Sigmoid(ctx)
