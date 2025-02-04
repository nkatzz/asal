# A bunch of functions which may be helpful to someone somewhen.
import re
import nnf
import torch
from typing import Literal
from src.asal_nesy.deepfa.automaton import DeepFA


def generate_random_sequence(
    deepfa: DeepFA, accepting: bool = True, sequence_length: int = 5
) -> dict[str, list[bool]]:
    # Generates a random sequence from the automaton.
    # :param deepfa: The deepFA from whose language
    #   to create the random sequence.
    # :param accepting: Whether to creating a sequence
    #   that is accepted by the automaton or not.
    # :param sequence_length: The length of the generated
    #   sequence.
    # :return a dictionary with values boolean lists specifying
    #   with entry list[i] specifying whether the symbol (key) is
    #   true in timestep i in the generated sequence.

    weights = {symbol: torch.rand(sequence_length) for symbol in deepfa.symbols}
    for weight in weights.values():
        _ = weight.requires_grad_()

    def labelling_function(var: nnf.Var) -> torch.Tensor:
        return (
            weights[str(var.name)] if var.true else 1 - weights[str(var.name)].detach()
        )

    mpe = deepfa.forward(
        labelling_function, max_propagation=True, return_accepting=accepting
    )

    most_probable_assignment: dict[str, list[bool]] = {
        symbol: (
            (torch.autograd.grad(mpe, weight, retain_graph=True)[0] * weight / mpe)
            > 0.5
        ).tolist()
        for symbol, weight in weights.items()
    }
    return most_probable_assignment


def parse_sapienza_to_fa(
    formula: str,
    variant: Literal["ltlf", "pltlf"] = "ltlf",
    drop_rejecting: bool = False,
) -> DeepFA:

    # Parse an LTLf or PLTLf formula pass it through
    # Mona for automaton conversion and then convert
    # the output of mona to sympy expressions
    # and return them

    from ltlf2dfa.ltlf2dfa import to_dfa
    from ltlf2dfa.parser.ltlf import LTLfParser
    from ltlf2dfa.parser.pltlf import PLTLfParser

    parser = LTLfParser() if variant == "ltlf" else PLTLfParser()
    parsed_formula = parser(formula)

    mona_output = to_dfa(parsed_formula, mona_dfa_out=True)

    if not mona_output.startswith("DFA"):
        raise RuntimeError(
            """Something has gone wrong in the translation and the 
            output of LTLf2DFA cannot be parsed"""
        )

    accepting_states = set(
        map(
            lambda value: int(value) - 1,
            (re.findall(r"Accepting states: (.*)\n", mona_output)[0].strip().split()),
        )
    )

    state_dict: dict[int, int] = {}
    transitions: dict[int, dict[int, list[nnf.NNF]]] = {}

    symbols = list(
        map(
            lambda s: nnf.Var(s.lower()),
            re.findall(r"DFA for formula with free variables: (.*)\n", mona_output)[0]
            .strip()
            .split(),
        )
    )

    for line in mona_output.splitlines():
        if not line.startswith("State"):
            continue

        match = re.match(r"State (\d+): ([01X]+) -> state (\d+)", line)

        if match is None:
            raise RuntimeError("unreachable")

        source, guard, destination = (
            int(match.group(1)),
            match.group(2),
            int(match.group(3)),
        )

        if source == 0 or destination == 0:
            continue

        if source not in state_dict:
            state_dict[source] = len(state_dict)

        if destination not in state_dict:
            state_dict[destination] = len(state_dict)

        symbolic_guard = nnf.And(
            {
                symbol if value == "1" else ~symbol
                for symbol, value in zip(symbols, guard)
                if value != "X"
            }
        )
        transitions.setdefault(source, {}).setdefault(destination, []).append(
            symbolic_guard
        )

    absorbing_rejecting_states = []
    for state in transitions:
        all_outgoing_pairs = list(transitions[state].items())
        if (
            len(all_outgoing_pairs) == 1
            and all_outgoing_pairs[0][0] == state
            and not all_outgoing_pairs[0][0] - 1 in accepting_states
            and nnf.Or(all_outgoing_pairs[0][1]).equivalent(nnf.true)
        ):
            absorbing_rejecting_states.append(state)

    absorbing_rejecting_states = absorbing_rejecting_states if drop_rejecting else []

    return DeepFA(
        {
            source
            - 1: {
                destination - 1: nnf.Or(disjuncts).simplify()
                for destination, disjuncts in transitions[source].items()
                if destination not in absorbing_rejecting_states
            }
            for source in transitions
            if source not in absorbing_rejecting_states
        },
        0,
        accepting_states,
    )
