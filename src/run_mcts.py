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


# As in run_batch.py
t_lim = 120

# As in run_batch.py
max_states = 3

# As in run_batch.py
target_class = 0

# As in run_batch.py
mini_batch_size = 100

# dataset = "maritime"
# dataset = "avg_robot"  # To use this you need to set a higher priority to minimizing FPs, FNs, due to the small num. of positive exmpls per batch.
dataset = "bsc_ductal"
fold = "fold_0"

# Path to the training set file.
train_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
    dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'train.csv')

# Path to the testing set file.
test_path = os.path.normpath(
    os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
    dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'test.csv')

# If true the training set is shuffled before a MCTS run.
shuffle = False

# A "seed" minibatch to kick-start the MCTS run
selected_mini_batch = 1  # Randomize this.

# Max number of MCTS iterations.
mcts_iterations = 10

# Exploration rate for MCTS.
expl_rate = 0.005

# Max number of children to add to a node. Each child is a revision of the automaton
# that corresponds to the parent node. This parameter controls the "horizontal" expansion
# of the search tree at each iteration.
max_children = 10  # 100

if __name__ == "__main__":
    tmpl = Template(max_states)
    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=shuffle)
    seed_data = train_data[selected_mini_batch]
    root = RootNode()

    mcts = MCTSRun(train_data, train_path, seed_data, mini_batch_size, tmpl,
                   t_lim, mcts_iterations, expl_rate, target_class, max_children, models_num='0')

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
    test_model_mproc(mcts.best_model, test_path, str(target_class), mini_batch_size)

    logger.info(yellow(f'On testing set: TPs, FPs, FNs: {mcts.best_model.global_performance_counts}, '
                       f'F1-score: {mcts.best_model.global_performance}'))
