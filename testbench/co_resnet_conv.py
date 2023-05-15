import sys
sys.path.append("./src/")
sys.path.append("./src/Ax/")

from hw_generator.generator_conv import CONVGenerator

from benchmark.benchmark import BenchmarkCNN
from codesign.hw_evaluation import gen_software
from codesign.explorer import codesign



if __name__ == '__main__':
    dtype = "int8"
    method = "Model"
    constraints = {"energy": 0, "latency": 0}  # TODO

    print("Testing accelerators with CONV intrinsic ...")
    generator = CONVGenerator() 
    benchmark = BenchmarkCNN("ResNet50", dtype, layout=generator.type) 
    codesign(benchmark, generator, method, constraints, init_size=5, trial_num=10)
    print("Passed.")
