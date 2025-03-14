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
from src.args_parser import parse_args

# Time limit (in seconds) for the solver to find a solution. Set this to 'inf' to disable time limit.
t_lim = 120

# Max number of states in the learnt automaton
max_states = 5

# The target class that we are trying to predict
target_class = 1

# The size (number of sequences) in a training sample. The ratio of positive/negative
# sequences in the sample is proportional to that ratio in the entire training set.
mini_batch_size = 30

dataset = "maritime"
fold = "fold_0"

with_jobLib = False

# Paths to the training and testing set files. These may be modified to point to any such pair
# of files by replacing the following with the absolute paths of the files.
train_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
    dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'train.csv')

test_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
    dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'test.csv')

if __name__ == '__main__':
    time_limit = float('inf') if t_lim == 'inf' else t_lim
    template = Template(max_states, target_class)
    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=False)
    mini_batch = train_data[1]  # Get mini-batch 1

    parser = parse_args()
    args = parser.parse_args()

    learner = Learner(template, mini_batch, args, with_joblib=with_jobLib)
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
    test_model_mproc(model, train_path, str(target_class), args.domain, mini_batch_size)
    logger.info(green(f'On training set: TPs, FPs, FNs: {model.global_performance_counts}, '
                      f'F1-score: {model.global_performance}'))

    logger.info(green('Testing on testing set...'))
    test_model_mproc(model, test_path, str(target_class), args.domain, mini_batch_size)

    logger.info(green(f'On testing set: TPs, FPs, FNs: {model.global_performance_counts}, '
                f'F1-score: {model.global_performance}'))
