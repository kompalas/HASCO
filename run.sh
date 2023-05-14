#!/bin/bash

#python3 testbench/co_resnet_conv.py 2>&1 | tee logs/out.log
python3 testbench/co_resnet_gemm.py 2>&1 | tee logs/out.log
