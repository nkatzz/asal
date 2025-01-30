from sdds.asp_programs import road_r_debug
from sdds.build_sdds import SDDBuilder
import torch
from functools import reduce
import operator


class DebugProbSFA:
    def __init__(self, sdd_builder, num_states):
        self.sdd_builder = sdd_builder
        self.num_states = num_states
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.states_distribution = torch.zeros(self.num_states).to(self.device)

    def eval_polynomials(self, nn_output):
        sdd_vars = self.sdd_builder.indexed_vars.keys()
        var_names_to_probs = {var: prob for var, prob in zip(sdd_vars, nn_output.squeeze(0))}
        return {guard: polynomial.eval(var_names_to_probs) for
                guard, polynomial in self.sdd_builder.polynomials.items()}

    def get_guards_probabilities(self, nn_output):
        sdds = self.sdd_builder.circuits
        guards_probabilities = {}
        for guard_name, sdd in sdds.items():
            guards_probabilities[guard_name] = self.get_guard_prob(sdd, nn_output)
        return guards_probabilities

    def get_guard_prob(self, guard_sdd, nn_output):
        sdd_vars = self.sdd_builder.indexed_vars.keys()

        # This mapping is of the form {'a1_movaway': p1, 'a1_movtow': p2, ...}
        var_names_to_probs = {var: prob for var, prob in zip(sdd_vars, nn_output.squeeze(0))}

        # This is a mapping with the actual sdd var ids as keys ({1: p1, 2: p2, ...}):
        sdd_vars_to_probs = {self.sdd_builder.indexed_vars[var]: prob for var, prob in var_names_to_probs.items()}

        # This is the reverse indexed_vars mapping ({1: 'a1_movaway', 2: a1_movtow, ...})
        sdd_vars_to_var_names = {v: k for k, v in self.sdd_builder.indexed_vars.items()}

        return self.propagate_guard(guard_sdd, sdd_vars_to_probs, sdd_vars_to_var_names)

    def propagate_guard(self, sdd_node, sdd_vars_to_probs, sdd_vars_to_var_names):
        if sdd_node.is_false():
            return torch.tensor([0])
        elif sdd_node.is_true():
            return torch.tensor([1])
        elif sdd_node.is_literal():
            literal_id = abs(sdd_node.literal)
            if sdd_node.literal > 0:
                return sdd_vars_to_probs[literal_id]
            else:  # Negative literal
                if sdd_vars_to_var_names[literal_id].split('_')[0] in self.sdd_builder.categorical_vars:
                    # Negative literals of categorical variables always get 1
                    return torch.tensor([1])
                else:  # Binary (non-categorical variable)
                    return torch.tensor([1]) - sdd_vars_to_probs[literal_id]
        elif sdd_node.is_decision():
            return reduce(
                operator.add,
                [
                    self.propagate_guard(prime, sdd_vars_to_probs, sdd_vars_to_var_names)
                    * self.propagate_guard(sub, sdd_vars_to_probs, sdd_vars_to_var_names)
                    for prime, sub in sdd_node.elements()
                ],
            )
        else:
            raise RuntimeError("Error at pattern matching during WMC.")

    def get_transition_matrix(self, guards_probabilities):
        transition_matrix = torch.zeros(self.num_states, self.num_states, device=self.device)
        for guard, prob in guards_probabilities.items():
            s = guard.split('_')[1].split(',')
            from_state = int(s[0].split('(')[1])
            to_state = int(s[1].split(')')[0])
            transition_matrix[from_state - 1][to_state - 1] = prob
        # There are no outgoing transitions from the accepting state, except for the self-loop:
        transition_matrix[self.num_states - 1][self.num_states - 1] = torch.tensor(1.0)
        return transition_matrix

    @staticmethod
    def update_state_distribution(state_prob, transition_matrix):
        return torch.matmul(state_prob, transition_matrix)

    def process_sequence(self, sequence):
        self.states_distribution = torch.zeros(self.num_states).to(self.device)
        self.states_distribution[0] = 1.0  # Initially we're in the start state with probability 1.0

        print(self.states_distribution)

        for prediction in sequence:
            guards_probabilities = self.eval_polynomials(prediction)
            # guards_probabilities = self.get_guards_probabilities(prediction)
            transition_matrix = self.get_transition_matrix(guards_probabilities)
            self.states_distribution = self.update_state_distribution(
                self.states_distribution, transition_matrix)

            print(self.states_distribution, sum(self.states_distribution))

        return self.states_distribution


input_1 = torch.tensor([[0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.]])

input_1 = torch.tensor([[1.1672e-01, 7.9234e-02, 8.0405e-01, 8.7101e-03, 8.9478e-01, 9.6506e-02,
                         8.4292e-01, 1.1254e-01, 3.8565e-02, 5.9709e-03, 7.6949e-01, 5.7554e-03,
                         1.2813e-01, 9.6629e-02],
                        [7.3262e-02, 1.2150e-01, 8.0524e-01, 1.0403e-01, 8.2354e-01, 7.2429e-02,
                         9.2590e-01, 1.1525e-02, 2.7436e-02, 3.5134e-02, 8.2543e-01, 1.2900e-01,
                         3.7182e-02, 8.3845e-03],
                        [8.8761e-02, 1.2078e-01, 7.9046e-01, 1.0977e-01, 8.1998e-01, 7.0247e-02,
                         6.8956e-01, 1.1355e-01, 8.7121e-02, 1.0978e-01, 7.4804e-01, 1.1312e-01,
                         1.0283e-01, 3.6010e-02],
                        [5.8338e-02, 7.0180e-03, 9.3464e-01, 6.2333e-02, 9.3736e-01, 3.1194e-04,
                         7.6712e-01, 1.1736e-01, 9.9257e-02, 1.6268e-02, 8.0497e-01, 3.4318e-03,
                         6.8156e-02, 1.2344e-01],
                        [5.3316e-02, 2.3433e-02, 9.2325e-01, 2.4950e-02, 9.5316e-01, 2.1887e-02,
                         8.5941e-01, 6.7551e-02, 6.1925e-02, 1.1119e-02, 7.3061e-01, 1.0264e-01,
                         8.8471e-02, 7.8281e-02],
                        [6.3876e-02, 9.4539e-03, 9.2667e-01, 2.2419e-02, 8.4080e-01, 1.3678e-01,
                         8.1050e-01, 1.2510e-01, 4.9473e-02, 1.4920e-02, 8.3541e-01, 2.7609e-02,
                         9.9281e-03, 1.2706e-01],
                        [1.1887e-02, 1.3571e-01, 8.5241e-01, 2.9187e-02, 9.5593e-01, 1.4878e-02,
                         8.1224e-01, 5.9675e-02, 7.9443e-02, 4.8640e-02, 7.4058e-01, 1.1515e-01,
                         2.8510e-02, 1.1576e-01],
                        [1.2927e-01, 9.4500e-02, 7.7623e-01, 9.2497e-03, 9.2659e-01, 6.4157e-02,
                         8.1746e-01, 8.2855e-03, 8.4142e-02, 9.0117e-02, 7.4810e-01, 9.9968e-02,
                         7.0307e-02, 8.1627e-02],
                        [5.3292e-02, 4.9005e-02, 8.9770e-01, 4.0691e-03, 8.8226e-01, 1.1368e-01,
                         7.9107e-01, 6.1403e-02, 5.7066e-02, 9.0459e-02, 8.0655e-01, 7.1821e-02,
                         3.0141e-02, 9.1487e-02],
                        [6.6541e-03, 6.3011e-02, 9.3033e-01, 3.0595e-02, 8.7552e-01, 9.3884e-02,
                         7.9020e-01, 4.6231e-02, 6.2659e-02, 1.0091e-01, 9.3124e-01, 2.9904e-02,
                         2.3761e-02, 1.5091e-02]])


input_1 = torch.tensor([[1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                        [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.]])

input_1 = torch.tensor([[8.3646e-01, 1.0464e-01, 5.8902e-02, 4.1993e-02, 1.2931e-01, 8.2870e-01,
                         6.9324e-02, 6.9188e-02, 2.7833e-02, 8.3366e-01, 6.6734e-02, 7.9333e-01,
                         9.7090e-03, 1.3023e-01],
                        [7.7871e-01, 1.2382e-01, 9.7475e-02, 1.2017e-01, 5.1564e-03, 8.7468e-01,
                         1.1963e-01, 6.1726e-03, 9.8075e-02, 7.7613e-01, 1.7776e-02, 8.0687e-01,
                         8.6758e-02, 8.8599e-02],
                        [8.4043e-01, 5.1899e-02, 1.0767e-01, 3.5372e-02, 1.2606e-01, 8.3857e-01,
                         3.4635e-02, 7.5963e-02, 9.9306e-02, 7.9009e-01, 1.3013e-01, 7.9170e-01,
                         5.7861e-03, 7.2386e-02],
                        [9.3660e-01, 1.5945e-02, 4.7458e-02, 1.3442e-01, 4.0618e-02, 8.2496e-01,
                         1.1236e-01, 4.5125e-03, 3.7571e-02, 8.4556e-01, 6.2224e-02, 7.4940e-01,
                         1.2213e-01, 6.6247e-02],
                        [8.9703e-01, 2.0460e-02, 8.2513e-02, 7.9903e-02, 5.9483e-02, 8.6061e-01,
                         6.7486e-02, 4.7294e-02, 4.7455e-02, 8.3777e-01, 6.6416e-02, 8.0129e-01,
                         1.2220e-01, 1.0089e-02],
                        [8.3498e-01, 1.0387e-01, 6.1153e-02, 3.5984e-02, 1.0589e-01, 8.5812e-01,
                         2.9285e-02, 4.1851e-02, 5.9890e-02, 8.6897e-01, 8.7336e-02, 7.5558e-01,
                         7.1605e-02, 8.5483e-02],
                        [8.5341e-01, 1.0711e-01, 3.9479e-02, 6.1004e-02, 7.6729e-02, 8.6227e-01,
                         1.0900e-01, 5.2957e-02, 7.6900e-03, 8.3035e-01, 2.6101e-03, 7.9902e-01,
                         6.8686e-02, 1.2969e-01],
                        [9.8647e-01, 1.2762e-02, 7.6929e-04, 2.4073e-02, 7.8261e-02, 8.9767e-01,
                         5.0084e-02, 8.0178e-02, 4.1405e-02, 8.2833e-01, 8.0182e-02, 8.4383e-01,
                         4.0448e-02, 3.5543e-02],
                        [9.7613e-01, 1.1181e-02, 1.2690e-02, 9.4752e-02, 1.0408e-01, 8.0117e-01,
                         1.4325e-02, 1.2930e-01, 1.2534e-01, 7.3104e-01, 4.8358e-02, 7.8903e-01,
                         4.9328e-02, 1.1329e-01],
                        [7.9875e-01, 8.4445e-02, 1.1680e-01, 2.7575e-02, 1.1326e-01, 8.5917e-01,
                         9.5806e-02, 7.4003e-02, 5.7826e-03, 8.2441e-01, 7.1927e-02, 8.4006e-01,
                         2.8153e-02, 5.9855e-02]])


if __name__ == '__main__':
    print('Building SDDs...')

    sequence = input_1

    asp_program = road_r_debug
    var_names = ['a1', 'a2', 'l1', 'l2']
    sdd_builder = SDDBuilder(asp_program,
                             vars_names=var_names,
                             categorical_vars=var_names,
                             clear_fields=False)
    sdd_builder.build_sdds()
    num_states = 4
    sfa = DebugProbSFA(sdd_builder, num_states)

    final_states_distribution = sfa.process_sequence(sequence)

    print(final_states_distribution)
