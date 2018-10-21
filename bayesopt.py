import GPyOpt
import GPy
import numpy as np


bounds = [{'name': 'nodes',           'type': 'discrete',    'domain': (64, 128, 256, 512, 1024, 2048)},
          {'name': 'layers',          'type': 'discrete',    'domain': (4, 8, 12, 16, 20, 24, 28, 32)},
          {'name': 'batch_size',      'type': 'discrete',    'domain': (10, 100, 250, 500)},
          {'name': 'epochs',          'type': 'discrete',    'domain': (40, 80, 120, 160, 200)},
          {'name': 'dropout',         'type': 'continuous',  'domain': (.0, .5)}]

## TODO: Add hyperparameters here (replace existing values)
X_init = np.array([[64, 4, 10, 40, .3],
                   [2048, 20, 500, 200, .08]])
## TODO: Add validation score here (replace existing values)
Y_init = np.array([[89], [32]])

bo = GPyOpt.methods.BayesianOptimization(f=None, domain=bounds, X=X_init, Y=Y_init,
                                         kernel=GPy.kern.Matern32(len(bounds), ARD=True),
                                         acquisition_type='EI', model_type='GP', ARD=True)
bo.model.optimize_restarts = 10
x_next = bo.suggest_next_locations()
print("Use hyperparameters:")
print(x_next)