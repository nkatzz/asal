from src.asal.template import Template
from src.asal.auxils import get_train_data
from src.asal.asal import Asal
from src.logger import *
from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.circuit_auxils import nnf_map_to_sfa
from src.asal.asp import get_induction_program
from src.asal.auxils import timer
import argparse


@timer
def compile_sfa(sfa, asp_compilation_program, vars):  # vars, e.g. ['d1', 'd2', 'd3'] for multivar MNIST
    x = [f'query(guard,{rule.head}) :- {rule.head}.' for rule in sfa.rules]
    y = [f'query(guard,{head}) :- {head}.' for head in sfa.self_loop_guards.keys()]
    query_atoms = x + y
    sfa_to_str = sfa.show(mode='simple')
    asp_program = asp_compilation_program + sfa_to_str + '\n'.join(query_atoms) + '#show value/2.\n' + '#show query/2.'

    sdd_builder = SDDBuilder(asp_program,
                             vars_names=vars,
                             categorical_vars=vars,
                             clear_fields=False)

    sdd_builder.build_nnfs()
    circuits = sdd_builder.circuits
    # mc = model_count_nnf(circuits)
    # print(f'Model counts per query (NNFs):\n{mc}')

    # Compile into an SFA:
    sfa = nnf_map_to_sfa(circuits)
    logger.debug(f'\nCompiled SFA: {sfa.transitions}\nStates: {sfa.states}\nSymbols: {sfa.symbols}')
    return sfa


def induce_sfa(args, asp_compilation_program, vars, data=None, existing_sfa=None):
    shuffle = False
    template = Template(args.states, args.tclass)
    if data is None:  # Read symbolic sequences from a file (args.train)
        train_data = get_train_data(args.train, str(args.tclass), args.batch_size, shuffle=shuffle)
    else:
        train_data = data

    logger.debug(f'The induction program is:\n{get_induction_program(args, template)}')

    if existing_sfa is None:
        mcts = Asal(args, train_data, template, initialize_only=True)
        mcts.expand_root()
    else:
        new_args = argparse.Namespace(
            tclass=args.tclass,
            batch_size=20,
            test=args.test,
            train=args.train,
            domain=args.domain,
            predicates="equals",
            mcts_iters=2,
            all_opt=True,
            tlim=60,
            states=args.states,
            exp_rate=args.exp_rate,
            mcts_children=args.mcts_children,
            show=args.show,
            unsat_weight=10,  # set this to 0 to have uncertainty weights per sequence
            max_alts=args.max_alts,
            coverage_first=args.coverage_first,
            min_attrs=args.min_attrs,
            warns_off=False,
            revise=False
        )
        mcts = Asal(new_args, train_data, template)
        mcts.run_mcts()

    mode = "simple" if args.show == "s" else "reasoning"
    logger.info(blue(f'New SFA:\n{mcts.best_model.show(mode=mode)}\n'
                     f'training F1-score: {mcts.best_model.global_performance} '
                     f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})'))

    # logger.info('Compiling guards into NNF...')
    sfa = mcts.best_model

    compiled = compile_sfa(mcts.best_model, asp_compilation_program, vars)
    return compiled, sfa
