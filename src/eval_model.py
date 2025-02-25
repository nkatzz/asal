import os
import sys

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal.test_model_multproc import test_model_mproc
from src.asal.logger import *
from src.asal.auxils import f1

target_class = 0
path_scoring = False
mini_batch_size = 1000
# dataset = "ROAD-R"
# fold = "fold_3"
# test_path = '/home/nkatz/Desktop/agent_train.txt'

dataset = "bsc_genes_kidney"
fold = "fold_0"

test_path = '/home/nkatz/Downloads/asal/data/bsc_genes_kidney/folds/fold_0/test.csv'

model = """
  accepting(7).
transition(1,f(1,1),1). transition(1,f(1,7),7). transition(7,f(7,7),7).
holds(f(1,7),S,T) :- holds(at_least(hus1b,4),S,T), holds(at_most(loc100132354,4),S,T).
holds(f(7,7),S,T) :- sequence(S), time(T).
holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,7),S,T).
"""

if __name__ == "__main__":
    result = test_model_mproc(model, test_path, str(target_class), mini_batch_size, path_scoring=path_scoring)
    tps, fps, fns = result.get_tps(), result.get_fps(), result.get_fns()
    logger.info(yellow(f'On testing set: TPs, FPs, FNs: {tps}, {fps}, {fns}, '
                       f'F1-score: {f1(tps, fps, fns)}'))

