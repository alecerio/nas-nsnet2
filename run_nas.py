import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
#from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from nas.genetic3D import Genetic3D
from objective_function.objf.compute_objective_function import compute_objective_function
import csv
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def objective_function(x, out, *args, **kwargs):
    x = x.astype(np.int32)
    print(f"x: {x}")
    mapped_x = np.choose(x, [8, 16, 32])
    print(f"mapped x: {mapped_x}")
    numpy_weights_path = '/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/numpy_weights/'
    ref_path = '/home/alessandro/Desktop/nas-nsnet2/examples/pesq/reference2.wav'
    deg_path = '/home/alessandro/Desktop/nas-nsnet2/examples/pesq/degradated2.wav'
    root_path = '/home/alessandro/Desktop/nas-nsnet2/'
    build_path = '/home/alessandro/Desktop/build_nsnet2_nas/'
    [pesq_metric, inference_metric, memory_metric] = compute_objective_function(mapped_x, numpy_weights_path, ref_path, deg_path, root_path, build_path)
    print(f"normalized pesq: {pesq_metric}")
    print(f"normalized inference time: {inference_metric}")
    print(f"normalized memory footprint: {memory_metric}")
    out["F"] = [pesq_metric, inference_metric, memory_metric]

def callback_function(algorithm):
    pareto_front = algorithm.opt.get("F")
    mpq_config = algorithm.opt.get("X")
    gen_number = len(algorithm.history)
    with open(f'pareto_gen_{gen_number}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        header = [f'Var{i+1}' for i in range(mpq_config.shape[1])] + ["PESQ", "Inference Time", "Memory Footprint"]
        writer.writerow(header)

        for var, metrics in zip(mpq_config, pareto_front):
            writer.writerow(np.concatenate((var, metrics)))

population_size = 25
generations = 50

algorithm = NSGA2(
    pop_size=population_size,
    sampling=IntegerRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=PolynomialMutation(prob=0.2),
    eliminate_duplicates=True
)

genetic = Genetic3D(objective_function, 93, 0, 0, 2, algorithm, callback_function)
genetic.run(generations)
