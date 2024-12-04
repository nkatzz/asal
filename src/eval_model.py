import os
import sys

sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal.test_model_multproc import test_model_mproc
from src.asal.logger import *
from src.asal.auxils import f1

target_class = 2
path_scoring = False
mini_batch_size = 1000
dataset = "ROAD-R"
fold = "fold_3"

test_path = '/home/nkatz/Desktop/agent_train.txt'

model = """
        accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(1,f(1,3),3). transition(2,f(2,2),2). transition(2,f(2,3),3). transition(2,f(2,4),4). transition(3,f(3,2),2). transition(3,f(3,3),3). transition(4,f(4,4),4).
holds(f(1,3),S,T) :- holds(equals(action_1,movaway),S,T).
holds(f(2,4),S,T) :- holds(equals(action_2,other),S,T), holds(equals(location_1,vehlane),S,T), holds(equals(location_2,vehlane),S,T).
holds(f(1,2),S,T) :- holds(equals(location_1,incomlane),S,T), not holds(f(1,3),S,T).
holds(f(2,3),S,T) :- holds(equals(location_1,other),S,T), not holds(f(2,4),S,T).
holds(f(3,2),S,T) :- holds(equals(location_2,other),S,T).
holds(f(2,4),S,T) :- holds(equals(action_1,stop),S,T), holds(equals(location_2,incomlane),S,T).
holds(f(3,2),S,T) :- holds(equals(action_1,movaway),S,T).
holds(f(1,3),S,T) :- holds(equals(action_2,stop),S,T).
holds(f(2,3),S,T) :- holds(equals(location_1,incomlane),S,T), not holds(f(2,4),S,T).
holds(f(1,2),S,T) :- holds(equals(location_1,other),S,T), not holds(f(1,3),S,T).
holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T), not holds(f(1,3),S,T).
holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,3),S,T), not holds(f(2,4),S,T).
holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,2),S,T).
holds(f(4,4),S,T) :- sequence(S), time(T)."""

if __name__ == "__main__":
    result = test_model_mproc(model, test_path, str(target_class), mini_batch_size, path_scoring=path_scoring)
    tps, fps, fns = result.get_tps(), result.get_fps(), result.get_fns()
    logger.info(yellow(f'On testing set: TPs, FPs, FNs: {tps}, {fps}, {fns}, '
                       f'F1-score: {f1(tps, fps, fns)}'))

