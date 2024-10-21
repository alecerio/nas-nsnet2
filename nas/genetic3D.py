from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt

class ObjectiveFunction3D(ElementwiseProblem):
    def __init__(self, objective, n_var, n_constr, xl, xu):
        super().__init__( n_var=n_var, n_obj=3, n_constr=n_constr, xl=xl, xu=xu)
        self.objective = objective

    def _evaluate(self, x, out, *args, **kwargs):
        self.objective(x, out, *args, **kwargs)

class CallbackFunction3D(Callback):
    def __init__(self, callback_function):
        super().__init__()
        self.callback_function = callback_function

    def notify(self, algorithm):
        self.callback_function(algorithm)

class Genetic3D():
    def __init__(self, objective, n_var, n_constr, xl, xu, algorithm, callback) -> None:
        self.problem = ObjectiveFunction3D(objective, n_var, n_constr, xl, xu)
        self.algorithm = algorithm
        self.res = None
        self.callback = CallbackFunction3D(callback)
    
    def run(self, n_gen):
        self.res = minimize(
            self.problem,
            self.algorithm,
            termination=('n_gen', n_gen),
            seed=1,
            save_history=True,
            verbose=True,
            callback = self.callback,
        )
    
    def plot_pareto_optimal_3D(self):
        plot = Scatter()
        plot.add(self.res.F, facecolor="none", edgecolor="blue")
        plot.show()
    
    def plot_pareto_optimal_2D_12(self, xlabel, ylabel):
        pareto_trials = self.res.F
        f1_values = pareto_trials[:, 0]
        f2_values = pareto_trials[:, 1]

        plt.scatter(f1_values, f2_values, color='blue')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()
    
    def plot_pareto_optimal_2D_13(self, xlabel, zlabel):
        pareto_trials = self.res.F
        f1_values = pareto_trials[:, 0]
        f3_values = pareto_trials[:, 2]

        plt.scatter(f1_values, f3_values, color='blue')
        plt.xlabel(xlabel)
        plt.ylabel(zlabel)
        plt.grid(True)
        plt.show()
    
    def plot_pareto_optimal_2D_23(self, ylabel, zlabel):
        pareto_trials = self.res.F
        f2_values = pareto_trials[:, 1]
        f3_values = pareto_trials[:, 2]

        plt.scatter(f2_values, f3_values, color='blue')
        plt.xlabel(ylabel)
        plt.ylabel(zlabel)
        plt.grid(True)
        plt.show()
    
    def print_pareto_optimal(self):
        for i, f in enumerate(self.res.F):
            print(f"Pareto Solution {i+1}: Objective 1 = {f[0]}, Objective 2 = {f[1]}, Objective 3 = {f[2]}")
        

