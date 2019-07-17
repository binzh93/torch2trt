from torch2trt.torch2trt import *


@tensorrt_converter('torch.squeeze')
@tensorrt_converter('torch.Tensor.squeeze')
def convert_squeeze(ctx):
    input = ctx.method_args[0]
    args = ctx.method_args[1]
    output = ctx.method_return

    tensor_shape = input.shape
    assert args < len(tensor_shape), "Squeeze dim error!"
    assert isinstance(args, int), "Squeeze dim must be int number"
    if tensor_shape[args] == 1:
        if args == -1:
           tensor_shape = tensor_shape[: -1] 
        elif tensor_shape[args] == 1:
            tensor_shape = tensor_shape[: args] + tensor_shape[args: ]

    layer = ctx.network.add_shuffle(input._trt)
    layer.reshape_dims = tensor_shape[1:]

    output._trt = layer.get_output(0)