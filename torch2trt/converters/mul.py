from torch2trt.torch2trt import *


@tensorrt_converter('torch.Tensor.__mul__')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    # currently support example: (1, 256, 56, 56)*(1, 256, 1, 1) and (1, 256, 56, 56)*(1, 256, 56, 56)
    if(input_a.shape[1] == input_b.shape[1]): 
        layer = ctx.network.add_elementwise(input_a._trt, input_b._trt, trt.ElementWiseOperation.PROD)
    elif (input_a.shape[3] == input_b.shape[2]):  # TODO need to check 
        layer = ctx.network.add_matrix_multiply(input_a._trt, trt.MatrixOperation.TRANSPOSE,
                                                input_b._trt, trt.MatrixOperation.TRANSPOSE)
    
    output._trt = layer.get_output(0)
