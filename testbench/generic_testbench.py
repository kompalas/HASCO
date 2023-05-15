import sys
sys.path.append("./src/")
sys.path.append("./src/Ax/")
import argparse

from hw_generator.generator_conv import CONVGenerator
from hw_generator.generator_gemm import GEMMGenerator

from benchmark.benchmark import BenchmarkCNN
from codesign.hw_evaluation import gen_software
from codesign.explorer import codesign
from codesign import config as cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=cfg.supported_models + cfg.additional_models,
                        default='ResNet50', help='DNN model')
    parser.add_argument('--constraint-metrics', dest='metrics', nargs='+', choices=cfg.all_metrics,
                        default=['energy', 'latency'], help='Constraint metrics')
    parser.add_argument('--constraint-values', dest='constraints', nargs='+', type=float,
                        default=[0, 0], help='Constraint values')
    parser.add_argument('--intrinsic', dest='generator_type', choices=cfg.intrinsic_lib,
                        default='CONV', help='Type of intrinsic')
    parser.add_argument('--method', choices=cfg.eval_methods, default='Model', help='Eval method')
    parser.add_argument('--dtype', choices=cfg.bits_map.keys(), default='int8', help='Data type')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials')
    args = parser.parse_args()
    print("Command line arguments:", vars(args))

    assert len(args.metrics) == len(args.constraints), f"Mimatch between constraints ({len(args.metrics)}) "\
                                                       f"and their values ({len(args.constraints)})"
    constraints = {metric: value for metric, value in zip(args.metrics, args.constraints)}

    print(f"Testing accelerators with {args.generator_type} intrinsic ...")
    if args.generator_type == 'CONV':
        generator = CONVGenerator() 
    elif args.generator_type == 'GEMM':
        generator = GEMMGenerator() 

    benchmark = BenchmarkCNN(args.model, args.dtype, layout=generator.type) 
    codesign(benchmark, generator, args.method, constraints, init_size=5, trial_num=args.trials)
    print("Passed.")
