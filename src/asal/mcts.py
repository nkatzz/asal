from src.asal.template import Template
from src.asal.structs import Automaton, ScoredModel, SolveResult, SolveResultContainer, ffi
from src.asal.learner import Learner
from src.asal.auxils import get_train_data, f1_aux, split_by_n, get_seqs_by_id
from src.asal.test_model_multproc import test_model_mproc
from src.asal.logger import *
import math
import time
import abc
from multiprocessing import Process, Queue
import random
from statistics import mean
import pickle
from joblib import Parallel, delayed

"""
-----------------------------------------------------------------------
Incremental SFA learning with Monte Carlo Tree Search and SFA revision.
-----------------------------------------------------------------------
"""


class TreeNode:

    def __init__(self):
        self.visits: int = 0
        self.rewards: list[float] = []
        self.children = []  # These are InnerNode instances
        self.automaton: Automaton()
        self.is_root_node = False
        self.id = ''

    def update_rewards(self, x: float):
        self.rewards.append(x)

    def updateVisits(self, x: int):
        self.visits = x

    def increment_visits(self):
        self.visits = self.visits + 1

    def get_mean_reward(self):
        return sum(self.rewards) / float(self.visits)

    def add_child(self, child):
        self.children.append(child)

    def get_best_child(self, explore_rate: float):
        f = lambda child: child.get_mcts_score(explore_rate)
        return max(self.children, key=f)

    def is_leaf_node(self):
        return self.children == []

    def propagate_reward(self, reward: float):
        ancestors = self.get_ancestors_path()
        for node in ancestors:
            node.update_rewards(reward)
            node.increment_visits()

    @abc.abstractmethod
    def get_mcts_score(self, explore_rate):
        """Get the MCTS score for this node."""

    @abc.abstractmethod
    def get_depth(self):
        """Get the level in the search tree for this node."""

    @abc.abstractmethod
    def get_ancestors_path(self):
        """Get all ancestors for this node. This returns a list of TreeNode instances."""


class InnerNode(TreeNode):

    def __init__(self, _id, automaton, parent_node):
        super().__init__()
        self.id = _id
        self.automaton: Automaton = automaton
        self.parent_node = parent_node

    def get_mcts_score(self, explore_rate):
        return self.get_mean_reward() + explore_rate * math.sqrt(2 * math.log(self.parent_node.visits) / self.visits)

    def get_depth(self):
        reached_root = False
        parent = self.parent_node
        depth = 1
        while not reached_root:
            if isinstance(parent, InnerNode):
                depth += 1
                parent = parent.parent_node
            else:  # the case here is isinstance(parent, RootNode)
                reached_root = True
        return depth

    def get_ancestors_path(self):
        reached_root = False
        parent = self.parent_node
        ancestors = []
        while not reached_root:
            if isinstance(parent, InnerNode):
                ancestors.append(parent)
                parent = parent.parent_node
            else:  # the case here is isinstance(parent, RootNode)
                ancestors.append(parent)
                reached_root = True
        return ancestors


class RootNode(TreeNode):

    def __init__(self):
        super().__init__()
        self.is_root_node = True
        self.id = 'Root'
        self.visits = 1

    def get_mcts_score(self, explore_rate):
        return 0.0

    def get_depth(self):
        return 0

    def get_ancestors_path(self):
        return []

    def descent_to_best_child(self, explore_rate):
        reached_leaf = False
        best_child = self.get_best_child(explore_rate)
        while not reached_leaf:
            if not best_child.is_leaf_node():
                best_child = best_child.get_best_child(explore_rate)
            else:
                reached_leaf = True

        return best_child


class MCTSRun:

    def __init__(self,
                 training_data,
                 training_data_path,
                 seed_data_batch,
                 batch_size,
                 template,
                 solve_time_lim,
                 iters,
                 explore_rate,
                 target_class,
                 max_children,
                 models_num='0',
                 path_scoring=False,
                 with_joblib=True):

        self.training_data_whole = training_data
        self.train_path = training_data_path
        self.seed_data_batch = seed_data_batch
        self.best_score = 0.0
        self.best_model = Automaton()
        self.grounding_times = []
        self.solving_times = []
        self.testing_times = []
        self.generated_models_count = 0
        self.batch_size = batch_size
        self.template = template
        self.t_lim = solve_time_lim
        self.models_num = models_num
        self.path_scoring = path_scoring
        self.mcts_iterations = iters
        self.explore_rate = explore_rate
        self.target_class = target_class
        self.max_children = max_children
        self.total_training_time = None
        self.root_node = RootNode()
        self.tried_models = []
        self.with_joblib = with_joblib

        # Generate a seed automaton
        self.expand_root()

    def run_mcts(self):
        start = time.time()
        for i in range(self.mcts_iterations):
            best_child: InnerNode = self.root_node.descent_to_best_child(self.explore_rate)
            automaton = best_child.automaton

            """
            fp_chunks = split_by_n(automaton.fps_seq_ids, self.batch_size)
            data_chunks = [get_seqs_by_id(chunk, self.train_path) for chunk in fp_chunks]

            print(best_child.rewards)

            if data_chunks:
                logger.debug(green(f'Descent to best leaf node.\nBest is {best_child.id} '
                                                      f'with model (mean reward, mcts score: '
                                                      f'{best_child.get_mean_reward()}, '
                                                      f'{best_child.get_mcts_score(self.explore_rate)}):'
                                                      f'\n{automaton.show(mode='reasoning')}'))
                chunk = data_chunks[0]
                self.expand_node(best_child, chunk, initial_model=automaton)
            else:
                logger.debug(green('Zero error on training set. Stopping.'))
            """

            logger.info(green(f'\nExpanding best leaf node.\nBest node: {best_child.id}\n'
                              f'Visits: {best_child.visits}\n'
                              f'Mean reward: {best_child.get_mean_reward()}\n'
                              f'MCTS score: {best_child.get_mcts_score(self.explore_rate)}\n'
                              f'Model:\n'
                              f'\n{automaton.show(mode="""reasoning""")}'))

            batch_id = self.get_revision_batch(automaton)

            if f1_aux(automaton.counts_per_batch[batch_id]) == 1.0:
                # The selected mini-batch is either the worst w.r.t to F1-score, or a random one
                # selected from those where the automaton is non-perfect. If such a mini-batch
                # does not exist, the first one is returned, which will be perfect. So, if the
                # automaton achieves an F1-score of 1.0 on the selected mini-batch, this means that
                # its perfect throughout the training set, so we should terminate the search.
                logger.info(blue('Found perfect model!'))
                break

            # automaton.counts_per_batch.pop(batch_id)  # Remove it, so that we don't revise on this again.

            logger.info(f'Parent rewards: {best_child.parent_node.rewards}')
            # data = get_train_data(train_path, str(target_class), mini_batch_size)
            data = self.training_data_whole
            self.expand_node(best_child, data[batch_id], initial_model=automaton)

        end = time.time()
        total_time = end - start
        self.total_training_time = total_time
        # self.show_final_msg(total_time)

    def expand_root(self):
        """Generate a pool of initial models from the seed data batch. These are placed
           as children to the root node of the search tree and are used to kick-start the
           search for revisions. The size of the pool is controlled by the models_num
           parameter (the default '0' yields all optimal models from the seed mini batch)."""
        logger.info('Starting new MCTS run. Expanding root node.')
        self.expand_node(self.root_node, self.seed_data_batch)

    def expand_node(self, parent_node, batch_data: str, initial_model=Automaton()):

        learner = Learner(self.template, batch_data, self.t_lim,
                          existing_model=initial_model, with_joblib=self.with_joblib)

        solve_result = None
        if not self.with_joblib:
            solve_result = self.call(learner, self.models_num)
        else:
            sr = self.call_learner_joblib(learner, self.models_num)
            solve_result = SolveResultContainer(sr.deserialize_models(), sr.grounding_time, sr.solving_time)

        models = solve_result.models

        if not models:
            logger.error("No models returned from the solver.")
            sys.exit()

        if isinstance(models, Automaton):  # A single automaton is returned.
            models = [models]

        logger.info(blue(f'Generated {len(models)} new models (max children size: {self.max_children})'))
        self.update_stats(solve_result)

        if (len(models)) > self.max_children:
            models = random.sample(models, self.max_children)

        # Going for optimal models causes problems in case the optimum has not been found within the time limit.
        # We'll then have zero models returned. So, we compare solving time (ST) with time limit (TM).
        # If ST < TM this means that the optimum has been found and we can ask for models where
        # optimality_proven = True. Otherwise, it's best to simply test the generated models within the time
        # limit and work with them.
        if solve_result.solving_time < self.t_lim:
            optimal_models = list(filter(lambda x: x.optimality_proven, models))
            logger.info(f'Optimal models: {len(optimal_models)}')
            if len(optimal_models) > self.max_children:
                models = random.sample(optimal_models, self.max_children)
            else:
                models = optimal_models

        # for m in models:
        #    print(f'{m.show()}, {m.cost}\n{[a.str for a in m.body_atoms_asp]}\n')

        logger.info(f'Testing...')
        start = time.time()
        for m in models:
            test_model_mproc(m, self.train_path, str(self.target_class),
                             self.batch_size, path_scoring=self.path_scoring,
                             data_whole=self.training_data_whole, test_with_clingo=True)
        end = time.time()
        self.testing_times.append(end - start)
        logger.info(blue(f'Testing time: {end - start} sec'))

        logger.info(f'\nNew models: {len(models)}\nParent:  '
                    f'{parent_node.id}\nModels\' F1-scores: {[m.global_performance for m in models]}.')

        for model, m_count in zip(models, range(1, len(models) + 1)):
            if model.global_performance > initial_model.global_performance:
                node_id = f'Node-{parent_node.get_depth() + 1}-{m_count}'
                new_node = InnerNode(node_id, model, parent_node)
                new_node.increment_visits()
                new_node.update_rewards(model.global_performance)
                parent_node.add_child(new_node)

        if parent_node.children:
            best_model_found = max(parent_node.children, key=lambda x: x.automaton.global_performance)
            reward = best_model_found.automaton.global_performance
            best_model_found.propagate_reward(reward)
        else:
            best_model_found = max(models, key=lambda x: x.global_performance)
            reward = best_model_found.global_performance
            # logger.info(red(f'No children added (all worse that {parent_node.id}). Propagated reward: {reward}'))
            as_node = InnerNode('dummy node', best_model_found, parent_node)
            as_node.propagate_reward(reward)

        if self.get_model(best_model_found).global_performance > self.best_score:
            self.best_model = best_model_found.automaton
            self.best_score = best_model_found.automaton.global_performance

        for m in models: self.tried_models.append(m)

        return models

    @staticmethod
    def get_revision_batch(model: Automaton):
        """Get a mini-batch where the current automaton scores poorly, in order to
           use the mini-batch for revising the automaton.
           A mini-batch where the automaton has a minimal F1-score is returned."""

        sm = ScoredModel(model, [], (0, 0, 0), model.counts_per_batch)
        batch_id = sm.get_most_urgent_mini_batch('f1-score')

        # F1-score may be 0.0 if everything's zero.
        # We don't want a batch where the model makes no mistakes.
        mini_batch_found = False
        while not mini_batch_found:
            if model.counts_per_batch[batch_id] == (0, 0, 0):
                model.counts_per_batch.pop(batch_id)
                batch_id = sm.get_most_urgent_mini_batch('f1-score')
            else:
                mini_batch_found = True

        logger.info(yellow(f'Selected mini-batch: {batch_id}, counts: {model.counts_per_batch[batch_id]}'))
        return batch_id

    @staticmethod
    def call_learner(_learner: Learner, q: Queue, _models_num):
        if _models_num == '0':
            # Return all optimal models.
            q.put(_learner.induce_models(sols_num='0'))
        elif _models_num == '1':
            # Return one optimal model.
            q.put(_learner.induce_models(sols_num='1'))
        else:
            # Return up to _models_num models found within the time limit.
            # The implementation here needs work.
            q.put(_learner.induce_multiple(_models_num))

    def call(self, _learner: Learner, _models_num):
        q = Queue()
        process = Process(target=self.call_learner, args=(_learner, q, _models_num))
        process.start()
        results = q.get()
        process.join()
        return results

    @staticmethod
    def serialize_cffi_obj(cffi_obj):
        """Convert a single CFFI object to a picklable form."""
        return bytes(ffi.buffer(cffi_obj))

    @staticmethod
    def serialize_cffi_list(cffi_list):
        """Convert a list of CFFI objects to a list of picklable bytes."""
        return [bytes(ffi.buffer(obj)) for obj in cffi_list]

    def call_learner_joblib(self, _learner: Learner, models_num):
        """Worker function that processes Learner's results."""
        if models_num == '0':
            result = _learner.induce_models(sols_num='0')  # Single CFFI object
            # serialized_result = self.serialize_cffi_obj(result)  # Serialize it
        elif models_num == '1':
            result = _learner.induce_models(sols_num='1')  # Single CFFI object
            # serialized_result = self.serialize_cffi_obj(result)  # Serialize it
        else:
            result = _learner.induce_multiple(models_num)  # List of CFFI objects
            # serialized_result = self.serialize_cffi_list(result)  # Serialize list

        return result  # serialized_result

    def call_joblib(self, _learner: Learner):
        """Runs the learner in a separate process using joblib."""
        results = Parallel(n_jobs=1, backend="loky")(
            [delayed(self.call_learner_joblib)(_learner, self.models_num)]  # ✅ Enclosed in a list
        )

        return results[0]  # joblib returns a list, so get the first result

    def update_stats(self, solve_result):
        self.grounding_times.append(solve_result.grounding_time)
        self.solving_times.append(solve_result.solving_time)
        new_models_count = len(solve_result.models) if isinstance(solve_result.models, list) else 1
        self.generated_models_count = self.generated_models_count + new_models_count

    @staticmethod
    def get_model(obj):
        if isinstance(obj, Automaton):
            return obj
        else:  # InnerNode
            return obj.automaton

    def show_final_msg(self, total_training_time):
        logger.info(yellow(f'\nBest model found:\n{self.best_model.show(mode="""reasoning""")}\n\n'
                           f'F1-score on training set: {self.best_model.global_performance} '
                           f'(TPs, FPs, FNs: {self.best_model.global_performance_counts})\n'
                           f'Generated models: {self.generated_models_count}\n'
                           f'Average grounding time: {mean(self.grounding_times)}\n'
                           f'Average solving time: {mean(self.solving_times)}\n'
                           f'Model evaluation time: {sum(self.testing_times)}\n'
                           f'Total training time: {total_training_time}'))


if __name__ == "__main__":
    t_lim = 120  # 'inf'
    max_states = 4
    target_class = 1
    mini_batch_size = 20
    tmpl = Template(max_states, target_class)
    train_path = '../../data/Maritime_TRAIN_SAX_8_ASP.csv'
    # train_path = '/media/nkatz/storage/seqs/caviar/caviar_data/fold1/train_fold_1_discretized.txt'
    test_path = '../../debug/Maritime_TEST_SAX_8_ASP.csv'
    shuffle = False
    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=shuffle)
    selected_mini_batch = 1  # Randomize this.
    mcts_iterations = 10
    expl_rate = 0.005
    max_children = 5  # 100
    path_scoring = False  # This works if needed, but makes testing a bit  slower.
    with_joblib = True  # If false uses multiprocessing.Process for multi-processing. Causes problems on Windows!

    seed_data = train_data[selected_mini_batch]
    root = RootNode()

    mcts = MCTSRun(train_data, train_path,
                   seed_data, mini_batch_size, tmpl,
                   t_lim, mcts_iterations, expl_rate,
                   target_class, max_children, models_num='0', path_scoring=path_scoring, with_joblib=with_joblib)
    mcts.run_mcts()
