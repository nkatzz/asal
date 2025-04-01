import os
import sys
import argparse

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal.test_model_multproc import test_model_mproc
from src.logger import *
from src.asal.auxils import f1

args = argparse.Namespace(
    tclass=1,
    batch_size=10000,
    test="/home/nkatz/dev/asal/data/mnist/folds/fold_0/test.csv",
    domain="/home/nkatz/dev/asal/src/asal/asp/domains/mnist.lp",
    predicates="equals",
    warns_off=True
)

model = """\
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). transition(2,f(2,3),3). 
transition(3,f(3,3),3). transition(3,f(3,4),4). transition(4,f(4,4),4).
holds(f(1,2),S,T) :- holds(equals(even,1),S,T), holds(equals(gt_6,1),S,T).
holds(f(2,3),S,T) :- holds(equals(odd,1),S,T), holds(equals(leq_6,1),S,T).
holds(f(3,4),S,T) :- holds(equals(leq_3,1),S,T).
holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,4),S,T).
holds(f(4,4),S,T) :- sequence(S), time(T).
holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,3),S,T).
holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T).
"""

result = test_model_mproc(model, args, path_scoring=False)

tps, fps, fns = result.get_tps(), result.get_fps(), result.get_fns()
logger.info(yellow(f'On testing set: TPs, FPs, FNs: {tps}, {fps}, {fns}, '
                   f'F1-score: {f1(tps, fps, fns)}'))
