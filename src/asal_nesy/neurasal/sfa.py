from src.asal.template import Template
from src.asal.auxils import get_train_data
from src.asal.mcts import RootNode, MCTSRun
from statistics import mean
from src.asal.logger import *
from src.asal_nesy.cirquits.asp_programs import *
from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.circuit_auxils import model_count_nnf, nnf_map_to_sfa


def induce_sfa(train_path, max_states, target_class,
               time_lim=1000, mini_batch_size=1000,
               shuffle=False, seed_mini_batch=0,
               mcts_iterations=1, expl_rate=0.005, max_children=100):
    template = Template(max_states, target_class)
    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=shuffle)
    seed_data = train_data[seed_mini_batch]
    root = RootNode()

    # This learns something from the seed mini-batch.
    mcts = MCTSRun(train_data, train_path, seed_data, mini_batch_size, template,
                   time_lim, mcts_iterations, expl_rate, target_class, max_children, models_num='0')

    logger.info(green(f'\nBest model found:\n{mcts.best_model.show(mode="""simple""")}\n\n'
                      f'F1-score on training set: {mcts.best_model.global_performance} '
                      f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})\n'
                      f'Generated models: {mcts.generated_models_count}\n'
                      f'Average grounding time: {mean(mcts.grounding_times)}\n'
                      f'Average solving time: {mean(mcts.solving_times)}\n'
                      f'Model evaluation time: {sum(mcts.testing_times)}\n'
                      f'Total training time: {mcts.total_training_time}'))

    # logger.info('Compiling guards into NNF...')

    asp_program = mnist_even_odd_learn
    learnt_sfa = mcts.best_model.show(mode='simple')
    x = [f'query(guard,{rule.head}) :- {rule.head}.' for rule in mcts.best_model.rules]

    # y = [f'query(guard,{head}) :- {head}.' for head in mcts.best_model.self_loop_guards.keys()
    #      if head not in mcts.best_model.get_accepting_state_self_loop_guards()]

    y = [f'query(guard,{head}) :- {head}.' for head in mcts.best_model.self_loop_guards.keys()]
    query_atoms = x + y
    asp_program = asp_program + learnt_sfa + '\n'.join(query_atoms) + '#show value/2.\n' + '#show query/2.'

    sdd_builder = SDDBuilder(asp_program,
                             vars_names=['d'],
                             categorical_vars=['d'],
                             clear_fields=False)

    sdd_builder.build_nnfs()
    circuits = sdd_builder.circuits
    # mc = model_count_nnf(circuits)
    # print(f'Model counts per query (NNFs):\n{mc}')

    # Compile into an SFA:
    sfa = nnf_map_to_sfa(circuits)
    logger.info(f'\nCompiled SFA: {sfa.transitions}\nStates: {sfa.states}\nSymbols: {sfa.symbols}')

    return sfa
