import nnf
import torch
import itertools
from src.asal_nesy.deepfa.automaton import DeepFA


def get_sfa():
    d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9 = map(
        nnf.Var, ("d_0", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7", "d_8", "d_9")
    )

    vars = d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9

    constraint = nnf.And(
        [nnf.And([~a | ~b for a, b in itertools.combinations(vars, r=2)]), nnf.Or(vars)]
    )

    even = d_0 | d_2 | d_4 | d_6 | d_8
    smaller_than_3 = d_0 | d_1 | d_2 | d_3
    between_3_and_6 = d_4 | d_5 | d_6
    larger_than_6 = d_7 | d_8 | d_9

    transitions = {
        0: {
            0: (even & larger_than_6).negate() & constraint,
            1: even & larger_than_6 & constraint,
        },
        1: {
            1: (even | larger_than_6) & constraint,
            2: (even.negate() & (smaller_than_3 | between_3_and_6)) & constraint,
        },
        2: {2: (smaller_than_3.negate()) & constraint, 3: (smaller_than_3) & constraint},
        3: {3: nnf.true & constraint},
    }

    deepfa = DeepFA(transitions, 0, {3})
    return deepfa


weights = {
    "d_0": torch.tensor(
        [0.1057, 0.1049, 0.1069, 0.1087, 0.1087, 0.1103, 0.1054, 0.1088, 0.1054, 0.1090]
    ),
    "d_1": torch.tensor(
        [0.1138, 0.1103, 0.1120, 0.1144, 0.1148, 0.1197, 0.1107, 0.1140, 0.1054, 0.1174]
    ),
    "d_2": torch.tensor(
        [0.0965, 0.0941, 0.0981, 0.1005, 0.1003, 0.0965, 0.0988, 0.0981, 0.0982, 0.0976]
    ),
    "d_3": torch.tensor(
        [0.0773, 0.0759, 0.0804, 0.0824, 0.0822, 0.0775, 0.0800, 0.0812, 0.0803, 0.0792]
    ),
    "d_4": torch.tensor(
        [0.0774, 0.0779, 0.0779, 0.0788, 0.0785, 0.0767, 0.0782, 0.0785, 0.0820, 0.0763]
    ),
    "d_5": torch.tensor(
        [0.0971, 0.0993, 0.0959, 0.0929, 0.0934, 0.0938, 0.0955, 0.0940, 0.0960, 0.0951]
    ),
    "d_6": torch.tensor(
        [0.0925, 0.0912, 0.0949, 0.0972, 0.0974, 0.0921, 0.0949, 0.0968, 0.0959, 0.0938]
    ),
    "d_7": torch.tensor(
        [0.0879, 0.0874, 0.0902, 0.0922, 0.0921, 0.0868, 0.0903, 0.0904, 0.0941, 0.0882]
    ),
    "d_8": torch.tensor(
        [0.1257, 0.1267, 0.1218, 0.1181, 0.1172, 0.1253, 0.1240, 0.1187, 0.1203, 0.1221]
    ),
    "d_9": torch.tensor(
        [0.1260, 0.1324, 0.1220, 0.1148, 0.1152, 0.1213, 0.1221, 0.1196, 0.1222, 0.1212]
    ),
}


def labelling_function(var: nnf.Var) -> torch.Tensor:
    return (
        weights[str(var.name)] if var.true else torch.ones_like(weights[str(var.name)])
    )


deepfa = get_sfa()
acceptance_prob = deepfa.forward(labelling_function)
print(acceptance_prob)
