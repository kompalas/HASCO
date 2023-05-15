#!/bin/bash
resdir=${1?}
logfile=${2?}

python3 src/evaluation/eval_run.py \
	--resdir $resdir \
	--logfile $logfile
