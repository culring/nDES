#  from pathos.multiprocessing import ProcessingPool
from math import ceil
from torch.distributions import Uniform

import numpy as np

from des_torch import DES
from cec2017.wrapper import cec2017
from tqdm import tqdm


def single_evaluation(i, n):
    recorded_times = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8, 0.9, 1.0]
    #  try:
    uniform = Uniform(-100., 100.)
    initial_value = uniform.sample((n,)).double()
    des_ = DES(initial_value, lambda x: cec2017(i, x.detach().numpy()), -100, 100,
               lamarckism=False, log_best_val=True, history=16)
    _, log = des_.run()
    best = log['best_val']
    result = []
    for r, bb in enumerate(recorded_times):
        ind = int(ceil(bb * len(best))) - 1
        result.append(abs(best[ind] - i * 100))
    #  except Exception as e:
        #  print(e)
        #  return None

    return result


def evaluate_cec_function(i, n):
#  pool = ProcessingPool()
#  results = pool.map(lambda _: single_evaluation(i, n), range(51))
#  results = np.array(results).T
    results = []
    for _ in tqdm(range(51)):
        results.append(single_evaluation(i, n))
    results = np.array(results).T
    np.savetxt("./results/{}_{}.txt".format(i, n), results, delimiter=",")


if __name__ == "__main__":
    #  for i in range(1, 31):
        #  evaluate_cec_function(i, 10)
    evaluate_cec_function(1, 10)
