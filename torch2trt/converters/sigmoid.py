from torch2trt.torch2trt import *
  

@tensorrt_converter('torch.nn.Sigmoid.forward')
def convert_Sigmoid(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    
    layer = ctx.network.add_activation(
        input=input._trt, type=trt.ActivationType.SIGMOID)
    output._trt = layer.get_output(0)
