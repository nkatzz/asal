from src.asal.template import Template
from src.asal.auxils import get_train_data
from src.asal.asal import RootNode, Asal
from statistics import mean
from src.asal.logger import *
from src.asal_nesy.cirquits.asp_programs import *
from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.circuit_auxils import model_count_nnf, nnf_map_to_sfa
from src.asal.asp import get_induction_program
from src.asal.auxils import timer
import time


@timer
def compile_sfa(sfa, asp_compilation_program):
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
    logger.debug(f'\nCompiled SFA: {sfa.transitions}\nStates: {sfa.states}\nSymbols: {sfa.symbols}')
    return sfa


def induce_sfa(args, asp_compilation_program, nn_provided_data=None, existing_sfa=None, ):
    shuffle = False
    template = Template(args.states, args.tclass)
    if nn_provided_data is None:
        train_data = get_train_data(args.train, str(args.tclass), args.batch_size, shuffle=shuffle)
    else:
        train_data = nn_provided_data

    logger.debug(f'The induction program is:\n{get_induction_program(args, template)}')

    # This learns something from the seed mini-batch.
    mcts = Asal(args, train_data, template, initialize_only=True)
    if existing_sfa is None:
        mcts.expand_root()
    else:
        mcts.expand_node(RootNode(), train_data[0], existing_sfa)

    logger.info(blue(f'New SFA:\n{mcts.best_model.show(mode="""simple""")}\n'
                     f'training F1-score: {mcts.best_model.global_performance} '
                     f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})'))

    # logger.info('Compiling guards into NNF...')
    sfa = mcts.best_model

    compiled = compile_sfa(mcts.best_model, asp_compilation_program)
    return compiled, sfa
