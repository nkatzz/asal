from src.asal.template import Template
from src.asal.auxils import get_train_data
from src.asal.mcts import RootNode, MCTSRun
from statistics import mean
from src.asal.logger import *
from src.asal_nesy.cirquits.asp_programs import *
from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.circuit_auxils import model_count_nnf, nnf_map_to_sfa
from src.asal.asp import get_induction_program
import time


def compile_sfa(sfa, asp_compilation_program):
    start_time = time.perf_counter()
    x = [f'query(guard,{rule.head}) :- {rule.head}.' for rule in sfa.rules]
    y = [f'query(guard,{head}) :- {head}.' for head in sfa.self_loop_guards.keys()]
    query_atoms = x + y
    sfa_to_str = sfa.show(mode='simple')
    asp_program = asp_compilation_program + sfa_to_str + '\n'.join(query_atoms) + '#show value/2.\n' + '#show query/2.'

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
    end_time = time.perf_counter()
    logger.info(green(f'Compilation time: {end_time - start_time} secs'))
    logger.info(f'\nCompiled SFA: {sfa.transitions}\nStates: {sfa.states}\nSymbols: {sfa.symbols}')
    return sfa


def induce_sfa(args):
    shuffle = False

    template = Template(args.states, args.tclass)
    train_data = get_train_data(args.train, str(args.tclass), args.batch_size, shuffle=shuffle)

    logger.debug(f'The induction program is:\n{get_induction_program(args, template)}')

    # This learns something from the seed mini-batch.
    mcts = MCTSRun(args, train_data, template, models_num='0')

    logger.info(blue(f'\nBest model found:\n{mcts.best_model.show(mode="""reasoning""")}\n\n'
                     f'F1-score on training set: {mcts.best_model.global_performance} '
                     f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})\n'
                     f'Generated models: {mcts.generated_models_count}\n'
                     f'Average grounding time: {mean(mcts.grounding_times)}\n'
                     f'Average solving time: {mean(mcts.solving_times)}\n'
                     f'Model evaluation time: {sum(mcts.testing_times)}\n'
                     f'Total training time: {mcts.total_training_time}'))

    # logger.info('Compiling guards into NNF...')
    asp_program = mnist_even_odd_learn

    compiled = compile_sfa(mcts.best_model, asp_program)
    return compiled
