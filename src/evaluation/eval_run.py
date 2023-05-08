import pandas as pd
import os
import re
import logging
import argparse
import numpy as np
from ast import literal_eval


def cfg_logger(logfile):
    """Configure logging for an evaluation script
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(stdout_formatter)
    logger.addHandler(ch)

    logfile = re.search('/(.*)', logfile).group(1)
    fh = logging.FileHandler(filename=os.path.join('logs', f'evaluate__{logfile}.log'),
                             mode='w', encoding='utf-8',)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(name)s] (%(processName)s) %(levelname)s: %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', required=True,
                        help='Logging file containing the stdout of the experiment')
    args = parser.parse_args()
    logger = cfg_logger(args.logfile)

    # read the logging file
    with open(args.logfile, 'r') as f:
        log = f.read()

    # getting the useful part of the logging file: the evaluation report
    final_results = re.search("Report:(.*?)###", log, re.DOTALL).group(1)

    # these are dicts of N entries, corresponding to M-sized lists
    #   N = number of evaluation metrics (hopefully 5: latency, throughput, power, energy, area)
    #   M = number of optimization trials
    # metrics: saves the evaluation HW metrics
    # tags: saves the configuration of parameters used to evaluate
    metrics = re.search('(.*?)\n', final_results).group(1)
    metrics = literal_eval(metrics)
    tags = re.search('\n(.*?)\n', final_results).group(1)
    tags = literal_eval(tags)
    tag_keys = ['PEx', 'PEy', 'sp_banks', 'L2_size(kB)', 'L1_size(kB)', 'dma_buswidth', 'dma_maxbytes', 'dataflow', 'dtype']

    for metric_name in tags:
        logger.info(f"Optimal results for {metric_name.capitalize()}")
        for idx, tag in enumerate(tags[metric_name]):
            logger.info(f'{metric_name.capitalize()}={metrics[metric_name][idx]:.3e}: ' +\
                        ', '.join(f'{param}={value}' for param, value in zip(tag_keys, tag.split('_'))))
        logger.info(f"Max {max(metrics[metric_name]):.3e} ({tags[metric_name][np.argmax(metrics[metric_name])]})")
        logger.info(f"Min {min(metrics[metric_name]):.3e} ({tags[metric_name][np.argmin(metrics[metric_name])]})")
        logger.info("*" + "-" * 50 + "*")

    
if __name__ == "__main__":
    main()

