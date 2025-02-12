from circuit_auxils import *
from asp_programs import *
from build_sdds import SDDBuilder
import torch
import torch.nn.functional as F


def test_circuits(case, sdds=True):
    """If sdds=False the generated circuits are nnfs"""
    if case == "mnist":
        asp_program = mnist
        sdd_builder = SDDBuilder(asp_program,
                                 vars_names=['d1', 'd2'],
                                 categorical_vars=['d1', 'd2'],
                                 clear_fields=False)  # var names should match the ones in the ASP program!

    elif case == "road_r":
        asp_program = road_r
        sdd_builder = SDDBuilder(asp_program,
                                 vars_names=['a1', 'a2', 'l1', 'l2'],
                                 categorical_vars=['a1', 'a2', 'l1', 'l2'],
                                 clear_fields=False)

    elif case == "mnist_even_odd":
        asp_program = mnist_even_odd
        sdd_builder = SDDBuilder(asp_program,
                                 vars_names=['d'],
                                 categorical_vars=['d'],
                                 clear_fields=False)
    else:
        raise RuntimeError('Unknown test')

    if sdds:
        sdd_builder.build_sdds()
        model_counts_per_query = model_count(sdd_builder)
        print(f'Model counts per query:\n{model_counts_per_query}')
        print(f'Total model count: {sum(model_counts_per_query.values())}')

        print('\nGenerating random probability distributions for the SDDs variables')

        grouped_vars = sdd_builder.grouped_sdd_vars
        probs_mapping_whole = {}
        for _, sdd_vars in grouped_vars.items():
            logits = torch.rand(1, len(sdd_vars))
            random_probs = F.softmax(logits, dim=1)
            vars_probs_mapping = {var: prob for var, prob in zip(sdd_vars, random_probs.squeeze(0))}
            for k, v in vars_probs_mapping.items():
                probs_mapping_whole[k] = v

        probs_per_query = weighted_model_count(sdd_builder, probs_mapping_whole)
        print(probs_per_query)
    else:
        sdd_builder.build_nnfs()
        circuits = sdd_builder.circuits
        mc = model_count_nnf(circuits)
        print(f'Model counts per query (NNFs):\n{mc}')

        # Compile into an SFA:
        sfa = nnf_map_to_sfa(circuits)
        print(sfa.check_deterministic())
        print(sfa.states)
        print(sfa.symbols)
        d = sfa.dot()
        d.view()
        print(sfa)


def test_sdds():
    # print('Generating SDDs for the MNIST example')

    # test_sdd('mnist')

    # print('\nGenerating SDDs for the ROAD-R example')
    # We get 768 models here. This is expected, since, in contrast to the MNIST case, where just one
    # query sum is true in each model, here there are overlaps, since the guards are not mutually
    # exclusive. I have verified that the sum of 768 is correct, by running clingo with constraints
    # of the form :- not f(1,1), :- not f(1,2) etc, to get the model count for each guard.

    # test_sdd('road_r')

    # print('\nGenerating SDDs for the MNIST-even-odd example')
    # Same here, I have verified that 30 is the correct number coming from summing (correct) per-guard model counts.

    test_circuits('mnist_even_odd')


def test_nnfs():
    test_circuits('mnist_even_odd', sdds=False)


if __name__ == "__main__":
    test_sdds()
    test_nnfs()
