import os
import sys
from statistics import mean

# Need to add the project root in the pythonpath.
sys.path.insert(0, os.path.normpath(os.getcwd() + os.sep + os.pardir))

from src.asal.template import Template
from src.asal.auxils import get_train_data
from src.asal.mcts import RootNode, MCTSRun
from src.asal.test_model_multproc import test_model_mproc
from src.asal.logger import *


# Time limit (in seconds) for the solver to find a solution. Set this to 'inf' to disable time limit.
t_lim = 120

# Max number of states in the learnt automaton
max_states = 4

# The target class that we are trying to predict.
target_class = 2

# The size (number of sequences) in a training sample. The ratio of positive/negative
# sequences in the sample is proportional to that ratio in the entire training set.
mini_batch_size = 100

# To use avg_robot set a higher priority to minimizing FPs, FNs, due to the small num. of positive exmpls per batch.
# dataset = "avg_robot"
# dataset = "maritime"
# dataset = "bsc_lobular"
# dataset = "weather"
# dataset = "mnist"
dataset = "ROAD-R"
fold = "fold_0"

# Paths to the training and testing set files. These may be modified to point to any such pair
# of files by replacing the following with the absolute paths of the files.
train_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
    dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'train.csv')

test_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
    dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'test.csv')

# If true the training set is shuffled before a MCTS run.
shuffle = False

# A "seed" minibatch to kick-start the MCTS run
selected_mini_batch = 0  # Could be randomized.

# Max number of MCTS iterations.
mcts_iterations = 5

# Exploration rate for MCTS.
expl_rate = 0.005

# Max number of children to add to a node. Each child is a revision of the automaton
# that corresponds to the parent node. This parameter controls the "horizontal" expansion
# of the search tree at each iteration.
max_children = 200  # 100

path_scoring = False

if __name__ == "__main__":
    tmpl = Template(max_states)
    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=shuffle)
    seed_data = train_data[selected_mini_batch]
    root = RootNode()

    # This learns something from the seed mini-batch.
    mcts = MCTSRun(train_data, train_path, seed_data, mini_batch_size, tmpl,
                   t_lim, mcts_iterations, expl_rate, target_class, max_children, models_num='0')

    if mcts_iterations > 1:  # revise from new batches.
        mcts.run_mcts()

    logger.info(green(f'\nBest model found:\n{mcts.best_model.show(mode="""reasoning""")}\n\n'
                      f'F1-score on training set: {mcts.best_model.global_performance} '
                      f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})\n'
                      f'Generated models: {mcts.generated_models_count}\n'
                      f'Average grounding time: {mean(mcts.grounding_times)}\n'
                      f'Average solving time: {mean(mcts.solving_times)}\n'
                      f'Model evaluation time: {sum(mcts.testing_times)}\n'
                      f'Total training time: {mcts.total_training_time}'))

    logger.info(yellow('Evaluating on testing set...'))
    test_model_mproc(mcts.best_model, test_path, str(target_class), mini_batch_size, path_scoring=path_scoring)

    logger.info(yellow(f'On testing set: TPs, FPs, FNs: {mcts.best_model.global_performance_counts}, '
                       f'F1-score: {mcts.best_model.global_performance}'))
