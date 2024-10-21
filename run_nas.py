import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
#from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from nas.genetic3D import Genetic3D
from objective_function.objf.compute_objective_function import compute_objective_function
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    gen_number = len(algorithm.history)
    with open(f'pareto_gen_{gen_number}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PESQ", "Inference Time", "Memory Footprint"]) 
        writer.writerows(np.column_stack([-pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2]]))

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

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(0, generations):
    pareto_front = np.loadtxt(f'pareto_gen_{i}.csv', delimiter=',', skiprows=1)  # Skip header
    ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], label=f'Gen {i}', alpha=0.3)

ax.set_title("Pareto Front Evolution")
ax.set_xlabel("PESQ normalized")
ax.set_ylabel("Inference Time normalized")
ax.set_zlabel("Memory Footprint normalized")

plt.legend(["Final generation"])

plt.show()


#metric_x = 'pesq'
#metric_y = 'inference time'
#metric_z = 'memory footprint'
#genetic.plot_pareto_optimal_3D()
#genetic.plot_pareto_optimal_2D_12(metric_x, metric_y)
#genetic.plot_pareto_optimal_2D_13(metric_x, metric_z)
#genetic.plot_pareto_optimal_2D_23(metric_y, metric_z)
#genetic.print_pareto_optimal()