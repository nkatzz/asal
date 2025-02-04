# This is a very simple module for combining finite automata with
# knowledge compilation. It defines a single class representing
# a symbolic automaton and offers utilities for compiling the
# symbolic transitions to tractable forms and performing weighted
# model counting and mpe.

import nnf
import torch
import operator
import graphviz
import itertools
import more_itertools

from typing import Callable
from functools import reduce


def nnf2str(sentence: nnf.NNF) -> str:
    match sentence:
        case nnf.true:
            return "True"
        case nnf.false:
            return "False"
        case nnf.Var(name=name, true=true):
            return ("~" if not true else "") + str(name)
        case nnf.And(children=children):
            return "({})".format(" & ".join(map(nnf2str, children)))
        case nnf.Or(children=children):
            return "({})".format(" | ".join(map(nnf2str, children)))
        case _:
            raise RuntimeError("pattern matching failed in NNF")


class DeepFA:
    def __init__(
        self,
        transitions: dict[int, dict[int, nnf.NNF]],
        initial_state: int,
        accepting_states: set[int],
        dsharp_path: str = "dsharp",
        check_deterministic: bool = True,
    ):
        # Initialize a DeepFA.
        # :param transitions: The entry transitions[source][destination] should
        #    contain the logical expression used to transition from source to destination
        # :param initial_state: The initial state of the automaton
        # :param accepting_states: The accepting states of the automaton

        self.transitions: dict[int, dict[int, nnf.NNF]] = {}
        self.initial_state = initial_state
        self.accepting_states = accepting_states

        # Keep the originally specified transitions before compilation
        # they are much simpler than their compiled form and better
        # for visualization
        self.original_transitions = transitions

        for source in transitions:
            for destination, guard in transitions[source].items():
                # Compile each transition guard with dsharp. This will yield
                # deterministic and decomposable sentences which we also make
                # smooth to be able to compute MPE. We also add a dummy conjunction
                # in case not all transitions are defined over the same variables.
                # This ensures we can compute posterior marginals and MPE correctly.
                self.transitions.setdefault(source, {})[destination] = nnf.And(
                    {
                        nnf.dsharp.compile(guard.to_CNF(), executable=dsharp_path)
                        .forget_aux()
                        .make_smooth(),
                        nnf.And(
                            set(
                                map(
                                    lambda symbol: nnf.Or(
                                        {nnf.Var(symbol), nnf.Var(symbol, False)}
                                    ),
                                    set(self.symbols) - set(map(str, guard.vars())),
                                )
                            )
                        ),
                    }
                )

        if check_deterministic and not self.check_deterministic():
            raise RuntimeError("The automaton specified is not deterministic")

        self.state2id = {state: i for i, state in enumerate(self.states)}

    def dot(
        self, simplify: bool = True, drop_formulae: bool = False
    ) -> graphviz.Digraph:
        import sympy

        automaton = graphviz.Digraph("automaton")
        for state in self.states:
            automaton.node(
                str(state),
                label="q{}".format(state),
                shape=("doublecircle" if state in self.accepting_states else "circle"),
            )

        for source in self.original_transitions:
            for destination, guard in self.original_transitions.get(source, {}).items():
                automaton.edge(
                    str(source),
                    str(destination),
                    (
                        (
                            str(sympy.simplify(nnf2str(guard)))
                            if simplify
                            else nnf2str(guard)
                        )
                        if not drop_formulae
                        else "guard_{}_{}".format(source, destination)
                    ),
                )

        automaton.node("dummy", label="", style="invis")
        automaton.edge("dummy", str(self.initial_state), "start")

        return automaton

    def check_deterministic(self) -> bool:
        # Will check whether the automaton is deterministic. Nothing implemented
        # in this module makes much sense for non-deterministic automata. Determinism
        # necessitates that all outgoing transitions from a are mutually exclusive.
        # Usually we also check for exhaustive guards but because of assumptions
        # regarding categorical variables this is not implemented here.
        # BE CAREFUL!!!! Make sure the outgoing mass from a state is always one.

        for state in self.states:
            guards = self.transitions.get(state, {}).values()
            if not all(
                (g1.negate() | g2.negate()).equivalent(nnf.true)
                for g1, g2 in itertools.combinations(guards, r=2)
            ):
                return False

        return True

    @property
    def symbols(self) -> list[str]:
        # Return all symbols in all transitions guards of the automaton
        return list(
            sorted(
                reduce(
                    set.union,
                    (
                        set(guard.vars())
                        for destandguards in self.original_transitions.values()
                        for guard in destandguards.values()
                    ),
                )
            )
        )

    @property
    def states(self) -> set[int]:
        # Return the states of the automaton
        return set(more_itertools.collapse(self.transitions.items()))

    def forward(
        self,
        labelling: Callable[[nnf.Var], torch.Tensor],
        max_propagation: bool = False,
        return_accepting: bool = True,
        accumulate: bool = False,
    ) -> torch.Tensor:
        # Evaluate the automaton based on a labelling i.e. a weight for each symbol
        # of the automaton in each timestep of execution.
        # :param labelling: A function that gets a nnf.Var and returns a tensor of
        #   shape either (B, S) or (S) with B batch size and S sequence length. An
        #   nnf.Var also includes the field true indicating whether the variable is
        #   negated. A labelling function should therefore also handle this case. It
        #   provides a weight for each literal (each var and its negation) for each
        #   timestep is a sequence (with possible batching) to compute multiple
        #   sequences simultaneously
        # :param max_propagation: Whether to compute both the circuits and the propagation
        #   of the automaton using the max-product semiring. This effectively computes the
        #   most probable path which leads to the acceptance of a sequence.
        # :param return_accepting: Whether to aggregate the final state probabilities over
        #   accepting or rejecting states.
        # :return: The weight of all paths starting from the initial state and ending in
        #   an accepting state if not max_propagation
        #   else the maximum weight path from an initial state to an accepting state

        # Get a random output from the labelling function to infer batch size
        # and sequence length
        sample_weight = labelling(nnf.Var(list(self.symbols)[0]))

        batch_size, sequence_length = (
            sample_weight.shape
            if len(sample_weight.shape) == 2
            else (1, sample_weight.shape[0])
        )

        device = sample_weight.device

        transition_matrices = torch.zeros(
            batch_size, sequence_length, len(self.states), len(self.states)
        ).to(device)

        for source in self.transitions:
            for destination, guard in self.transitions.get(source, {}).items():
                transition_matrices[
                    :, :, self.state2id[source], self.state2id[destination]
                ] = nnf.amc.eval(
                    guard,
                    (
                        operator.add
                        if not max_propagation
                        else (
                            lambda x1, x2: torch.stack(
                                (
                                    x1.reshape(batch_size, sequence_length),
                                    x2.reshape(batch_size, sequence_length),
                                )
                            )
                            .max(0)
                            .values
                        )
                    ),
                    operator.mul,
                    torch.zeros(batch_size, sequence_length).to(device),
                    torch.ones(batch_size, sequence_length).to(device),
                    labelling,
                )

        initial_state_distribution = torch.zeros(batch_size, len(self.states)).to(
            device
        )
        initial_state_distribution[:, self.state2id[self.initial_state]] = 1

        def batch_mm(vector: torch.Tensor, matrix: torch.Tensor):
            assert len(vector.shape) == 2
            assert len(matrix.shape) == 3
            assert vector.shape == matrix.shape[:-1]
            new_vector = torch.zeros_like(vector)
            for row in range(vector.shape[-1]):
                new_vector[:, row] = (
                    (vector * matrix[:, :, row]).max(-1).values
                    if max_propagation
                    else (vector * matrix[:, :, row]).sum(-1)
                )

            return new_vector

        final_state_distribution = reduce(
            batch_mm,
            transition_matrices.permute(1, 0, 2, 3),
            initial_state_distribution,
        )

        states_to_aggregate = (
            [self.state2id[state] for state in self.accepting_states]
            if return_accepting
            else [self.state2id[state] for state in self.states - self.accepting_states]
        )

        if accumulate:
            return (
                torch.stack(
                    list(
                        itertools.accumulate(
                            iterable=transition_matrices.permute(1, 0, 2, 3),
                            func=batch_mm,
                            initial=initial_state_distribution,
                        )
                    )[1:]
                )[:, :, states_to_aggregate]
                .sum(-1)
                .T
            )

        return (
            final_state_distribution[:, states_to_aggregate].sum(-1)
            if not max_propagation
            else final_state_distribution[:, states_to_aggregate].max(-1).values
        )
