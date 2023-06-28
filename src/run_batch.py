import os
import sys

# Need to add the project root in the pythonpath.
sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal.template import Template
from src.asal.auxils import get_train_data, f1
from src.asal.learner import Learner
from src.asal.structs import Automaton
from src.asal.test_model_multproc import test_model_mproc
from src.asal.logger import *

"""
Time limit (in seconds) for the solver to find a solution. You might want to adapt this parameter
based on your computer's specs. The given time limit of 2 minutes was deemed adequate for runs 
performed on a computer with 12 3.2GHz processors and 16 GBs of RAM. 
You may set this to 'inf' to disable time limit.
"""
t_lim = 120

"""
Max number of states in the learnt automaton
"""
max_states = 5

"""
The target (positive) class (we are in a binary classification setting).
"""
target_class = 1

"""
The size (number of sequences) in a training sample. The ratio of positive/negative 
sequences in the sample is proportional to that ratio in the entire training set. 
"""
mini_batch_size = 30

"""
Path to the training set file.
"""
train_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep + 'Maritime_TRAIN_SAX_8_ASP.csv')

"""
Path to the testing set file.
"""
test_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep + 'Maritime_TEST_SAX_8_ASP.csv')

if __name__ == '__main__':
    time_limit = float('inf') if t_lim == 'inf' else t_lim
    template = Template(max_states)
    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=False)
    mini_batch = train_data[1]  # Get mini-batch 1

    learner = Learner(template, mini_batch, time_limit)
    result = learner.induce_models()

    model: Automaton = result.models

    gr_time = result.grounding_time
    solving_time = result.solving_time
    logger.info(yellow(f'Learnt model:\n{model.show()}\n'
                       f'Cost: {model.cost}\n'
                       f'Local (in training batch) counts: {model.local_performance}\n'
                       f'Grounding time: {gr_time}\n'
                       f'Solving time: {solving_time}'))

    logger.info(green('Testing on the whole training set...'))
    test_model_mproc(model, train_path, str(target_class), mini_batch_size)
    logger.info(green(f'On training set: TPs, FPs, FNs: {model.global_performance_counts}, '
                      f'F1-score: {model.global_performance}'))

    logger.info(green('Testing on testing set...'))
    test_model_mproc(model, test_path, str(target_class), mini_batch_size)

    logger.info(green(f'On testing set: TPs, FPs, FNs: {model.global_performance_counts}, '
                f'F1-score: {model.global_performance}'))
