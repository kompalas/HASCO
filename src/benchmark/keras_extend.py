import tensorflow.keras.applications as keras_models
import csv
import os
from codesign.config import supported_models
from benchmark.computations import conv2d_compute, mm_compute, dwconv_compute
from benchmark import benchmark_dir


def get_model(model_name, input_shape):
    if model_name not in supported_models:
        raise NotImplementedError('unsupported model')
    keras_model = getattr(keras_models, model_name)(weights=None, include_top=True, input_shape=input_shape)
    print('Get the keras model: ' + model_name)

    return keras_model


def get_workloads(model, dtype, layout):
    from benchmark.benchmark import Workload
    workloads = []
    UNIQUE_WORKLOADS = set()

    for layer in model.layers: 
        config = layer.get_config()
        ltype = layer.__class__.__name__

        is_conv2d = ltype == 'Conv2D'
        is_dwconv2d = ltype == 'DepthwiseConv2D'
        is_sepconv2d = ltype == 'SeparableConv2D'
        is_gemm = ltype == 'Dense'

        if (is_conv2d or is_dwconv2d or is_sepconv2d):  
            
            c = layer.input_shape[3]
            y = layer.input_shape[1]
            x = layer.input_shape[2]
            k = layer.output_shape[3]
            r = layer.kernel_size[0]
            s = layer.kernel_size[1]
            stride = config["strides"][0]

            if is_conv2d:
                args = (1, c, y, x, k, r, s, stride, dtype, layout) 
                conv = Workload(config["name"], "CONV", conv2d_compute, args) 
                workloads.append(conv)
                UNIQUE_WORKLOADS.add(conv.tag)  

            elif is_dwconv2d:
                args = (1, c, y, x, k // c, r, s, stride, dtype, layout)
                dwconv = Workload(config["name"], "DWCONV", dwconv_compute, args)
                workloads.append(dwconv)
                UNIQUE_WORKLOADS.add(dwconv.tag)    


            elif is_sepconv2d:
                yo = layer.output_shape[1]
                xo = layer.output_shape[2]
                ko = int(config["depth_multiplier"])
                dwargs = (1, c, y, x, ko, r, s, stride, dtype, layout)
                dwconv = Workload(config["name"]+"_dw", "DWCONV", dwconv_compute, dwargs) 
                pwargs = (1, c * ko , yo, xo, k, 1, 1, 1, dtype, layout)
                pwconv = Workload(config["name"]+"_pw", "CONV", conv2d_compute, pwargs)
                workloads.append(dwconv)
                workloads.append(pwconv)
                UNIQUE_WORKLOADS.add(dwconv.tag)
                UNIQUE_WORKLOADS.add(pwconv.tag) 

        elif is_gemm:
            
            m = 1
            n = layer.output_shape[1]
            k = layer.input_shape[1]
            args = (m, n, k, dtype, layout) 
            gemm = Workload(config["name"], "GEMM", mm_compute, args)
            workloads.append(gemm)
            UNIQUE_WORKLOADS.add(gemm.tag)
        
        else: # unsupported layers
            continue
        
    print("Unique workloads: ", len(UNIQUE_WORKLOADS))

    return workloads


def get_cifar_workloads(model_name, dtype, layout):
    from benchmark.benchmark import Workload
    workloads = []
    UNIQUE_WORKLOADS = set()

    with open(os.path.join(benchmark_dir, 'cifar_models', f'{model_name}.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:

            layer_name, layer_type = row[:2]
            if layer_type == 'conv2d':
                c, y, x, k, r, s, stride = row[2:]
                args = (1, int(c), int(y), int(x), int(k), int(r), int(s), int(stride), dtype, layout)
                conv = Workload(layer_name, "CONV", conv2d_compute, args)
                workloads.append(conv)
                UNIQUE_WORKLOADS.add(conv.tag)
            
            elif layer_type == 'dense':
                m = 1
                k, _, _, n, _, _, _ = row[2:]
                args = (m, int(n), int(k), dtype, layout) 
                gemm = Workload(layer_name, "GEMM", mm_compute, args)
                workloads.append(gemm)
                UNIQUE_WORKLOADS.add(gemm.tag)

    print(len(UNIQUE_WORKLOADS), "unique workloads: ", UNIQUE_WORKLOADS)
    return workloads


