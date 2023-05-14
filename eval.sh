#!/bin/bash
logfile=${1?}

python3 src/evaluation/eval_run.py \
	--logfile $logfile
