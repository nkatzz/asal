import nnf
import torch
from src.asal_nesy.deepfa.automaton import DeepFA

# Define 4 symbols the first 3 of which are categorical.
# This means exactly one of them will be true in each timestep
a, b, c, d = map(nnf.Var, ("a", "b", "c", "d"))

# We must define a constraint that tells the compiled circuits
# thate a, b and c are indeed a categorical group.
categorical_constraint = (a | b | c) & (~a | ~b) & (~a | ~c) & (~b | ~c)


transitions = {
    0: {
        0: (a | b) & d & categorical_constraint,
        1: ((a | b) & d).negate() & categorical_constraint,
    },
    1: {0: ~c & categorical_constraint, 1: c & categorical_constraint},
}

deepfa = DeepFA(transitions, 0, {1})

# Observe that for each timestep the sum of the probabilities of
# a, b and c must be 1 since the define a categorical distribution.
weights = {
    "a": torch.Tensor([0.2, 0.2]),
    "b": torch.Tensor([0.7, 0.5]),
    "c": torch.Tensor([0.1, 0.3]),
    "d": torch.Tensor([0.4, 0.7]),
}


def labelling_function(var: nnf.Var) -> torch.Tensor:
    # This is very much like the standard labelling function
    # introduced above but we always give the value 1 for
    # negative literals. It's just the way it is.
    if str(var.name) in ("a", "b", "c"):
        return (
            weights[str(var.name)]
            if var.true
            else torch.ones_like(weights[str(var.name)])
        )

    return weights[str(var.name)] if var.true else 1 - weights[str(var.name)]


# Compute the actual probability
acceptance_prob = deepfa.forward(labelling_function)
print(acceptance_prob)
