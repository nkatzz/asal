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
import random

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


class Asal:

    def __init__(self,
                 args,
                 training_data,
                 template,
                 path_scoring=False,
                 with_joblib=False,
                 initialize_only=False):

        self.args = args
        self.training_data_whole = training_data
        self.train_path = args.train

        # self.seed_data_batch = self.training_data_whole[0]
        if isinstance(self.training_data_whole, dict):
            self.seed_data_batch = random.choice(self.training_data_whole)
        elif isinstance(self.training_data_whole, str):  # in this case we pass a set of seqs as a string, no batching
            self.seed_data_batch = training_data

        self.best_score = 0.0
        self.best_model = Automaton()
        self.grounding_times = []
        self.solving_times = []
        self.testing_times = []
        self.generated_models_count = 0
        self.batch_size = args.batch_size
        self.template = template
        self.t_lim = args.tlim
        self.path_scoring = path_scoring
        self.mcts_iterations = args.mcts_iters
        self.explore_rate = args.exp_rate
        self.target_class = args.tclass
        self.max_children = args.mcts_children
        self.total_training_time = None
        self.root_node = RootNode()
        self.tried_models = []
        self.with_joblib = with_joblib
        self.show = "reasoning" if args.show == "r" else "simple"

        if not initialize_only:
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
                              f'Mean reward: {best_child.get_mean_reward():.4f}\n'
                              f'MCTS score: {best_child.get_mcts_score(self.explore_rate):.4f}\n'
                              f'Model:\n'
                              f'\n{automaton.show(mode=self.show)}'))

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

            logger.debug(f'Parent rewards: {best_child.parent_node.rewards}')
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

        learner = Learner(self.template, batch_data, self.args,
                          existing_model=initial_model, with_joblib=self.with_joblib)

        if not self.with_joblib:
            solve_result = self.call(learner)
        else:
            # The main reason that I use subprocess was to have Clingo clean its symbols over iterations
            # so that we don't have memory issues. joblib might be OK for that as well (but need to check).
            # joblib is added for compatibility with Windows, but it is not implemented on the Tester side!
            sr = self.call_learner_joblib(learner)
            solve_result = SolveResultContainer(sr.deserialize_models(), sr.grounding_time, sr.solving_time)

        models = solve_result.models

        if not models:
            logger.error("No models returned from the solver.")
            # save the current induction program and data for debugging
            from src.asal.asp import get_induction_program
            p = get_induction_program(self.args, self.template)
            with open('induction_program.lp', 'w') as file:
                file.write(p)
            with open('induction_data.lp', 'w') as file:
                file.write(batch_data)
            logger.error(f"Current induction program and data saved "
                         f"in 'induction_program.lp' and 'induction_data.lp' files.")
            sys.exit()

        if isinstance(models, Automaton):  # A single automaton is returned.
            models = [models]

        logger.info(f'Generated {len(models)} new models (max children size: {self.max_children})')

        self.update_stats(solve_result)

        if solve_result.solving_time < self.t_lim:  # then an optimal model has been found...
            # However, the optimality_proven flag is not correctly set only when
            # ctl.configuration.solve.opt_mode = 'opt'
            # (is is set when ctl.configuration.solve.opt_mode = 'optN' though...).
            # Therefore, we cannot use this flag to filter-out the optimals as in the line below:
            # optimal_models = list(filter(lambda x: x.optimality_proven, models))
            # So we do is the following:
            # (i) When args.all_opt=False, we just get the last model
            # from the solver. Since solve_result.solving_time < self.t_lim, this model will
            # be optimal.
            # (ii) When args.all_opt=False we sample n models, or just take the n last models
            # from the solver, where n is self.max_children.
            if not self.args.all_opt:
                logger.info("Optimal model found!")
                #for m in models: print(m)
                models = [models[-1]]
            else:
                # Here optimality_proven is set correctly.
                optimal_models = list(filter(lambda x: x.optimality_proven, models))
                logger.info(f"Optimal models: {len(optimal_models)}")
                if len(optimal_models) > self.max_children:
                    models = random.sample(optimal_models, self.max_children)
                else:
                    models = optimal_models
        else:
            if (len(models)) > self.max_children:
                """models = random.sample(models, self.max_children)"""
                # Just keep the last instead of sampling randomly in this case
                models = models[-self.max_children:]

        logger.info(f'Testing...')
        start = time.time()
        for m in models:
            test_model_mproc(m, self.args, path_scoring=self.path_scoring,
                             data_whole=self.training_data_whole, test_with_clingo=True)
        end = time.time()
        self.testing_times.append(end - start)
        logger.info(f'Testing time: {(end - start):.4f} sec')

        logger.info(f'\nNew models: {len(models)}\nParent:  '
                    f'{parent_node.id}\nModels\' F1-scores: {[round(m.global_performance, 4) for m in models]}.')

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
            if isinstance(best_model_found, InnerNode):
                self.best_model = best_model_found.automaton
                self.best_score = best_model_found.automaton.global_performance
            else:  # it's an asal.structs.Automaton instance
                self.best_model = best_model_found
                self.best_score = best_model_found.global_performance

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
            tps, fps, fns = model.counts_per_batch[batch_id]
            condition = tps + fns == 0
            # if model.counts_per_batch[batch_id] == (0, 0, 0):
            if condition:
                model.counts_per_batch.pop(batch_id)
                batch_id = sm.get_most_urgent_mini_batch('f1-score')
            else:
                mini_batch_found = True

        logger.info(yellow(f'Selected mini-batch: {batch_id}, counts: {model.counts_per_batch[batch_id]}'))
        return batch_id

    @staticmethod
    def call_learner(_learner: Learner, q: Queue, all_opt):
        q.put(_learner.induce_models(all_opt=all_opt))
        # q.put(_learner.induce_multiple(_models_num))

    def call(self, _learner: Learner):
        q = Queue()
        process = Process(target=self.call_learner, args=(_learner, q, self.args.all_opt))
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

    def call_learner_joblib(self, _learner: Learner):
        """Worker function that processes Learner's results."""
        result = _learner.induce_models(self.args.all_opt)  # Single CFFI object
        # serialized_result = self.serialize_cffi_obj(result)  # Serialize it

        # result = _learner.induce_multiple(models_num)

        return result  # serialized_result

    def call_joblib(self, _learner: Learner):
        """Runs the learner in a separate process using joblib."""
        results = Parallel(n_jobs=1, backend="loky")(
            [delayed(self.call_learner_joblib)(_learner)]  # ✅ Enclosed in a list
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
        logger.info(yellow(f'\nBest model found:\n{self.best_model.show(mode=self.show)}\n\n'
                           f'F1-score on training set: {self.best_model.global_performance:.4f} '
                           f'(TPs, FPs, FNs: {self.best_model.global_performance_counts})\n'
                           f'Generated models: {self.generated_models_count}\n'
                           f'Average grounding time: {mean(self.grounding_times):.4f}\n'
                           f'Average solving time: {mean(self.solving_times):.4f}\n'
                           f'Model evaluation time: {sum(self.testing_times):.4f}\n'
                           f'Total training time: {total_training_time:.4f}'))
