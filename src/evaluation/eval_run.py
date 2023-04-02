import pandas as pd
import re
import logging
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', required=True,
                        help='Logging file containing the stdout of the experiment')
    args = parser.parse_args()

    with open(args.logfile, 'r') as f:
        log = f.read()

    print(log)



if __name__ == "__main__":
    main()

