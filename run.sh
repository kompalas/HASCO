#!/bin/bash

#python3 testbench/co_resnet_conv.py 2>&1 | tee logs/out.log
#python3 testbench/co_resnet_gemm.py 2>&1 | tee logs/out.log

python3 testbench/co_mobile_conv.py 2>&1 | tee logs/out.log
#python3 testbench/co_mobile_gemm.py 2>&1 | tee logs/out.log

#model=${1?}

#python3 testbench/generic_testbench.py \
#	--model $model \
#	--constraint-metrics energy latency \
#	--constraint-values 0 0 \
#	--intrinsic CONV \
#	--method Model \
#	--dtype int8 \
#	--trials 20 \
#	2>&1 | tee logs/${model}.log

