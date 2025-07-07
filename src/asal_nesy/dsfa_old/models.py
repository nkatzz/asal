import torch
import torch.nn as nn
import torch.nn.functional as F
from src.asal_nesy.dsfa_old.utils import digit_probs_to_rule_probs, digit_probs_to_rule_probs_no_nesy
from functools import reduce
import operator
import time
from src.asal_nesy.globals import device

'''
import importlib.util

# Define the path to the module
module_path = '../../neural_component/model.py'

# Load the module
spec = importlib.util.spec_from_file_location("road_r", module_path)
road_r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(road_r)
'''


class DigitCNN(nn.Module):
    def __init__(self, dropout_rate=0.4, out_features=10, log_softmax: bool = False):
        super().__init__()
        self.out_features = out_features
        self.conv1 = nn.Conv2d(1, 8, (3, 3))
        self.conv2 = nn.Conv2d(8, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.dense = nn.Linear(in_features=32, out_features=out_features)
        self.log_softmax = log_softmax
        if log_softmax:
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = nn.Softmax(dim=1)
        self.last_logits = None

    def forward(self, input_image, apply_softmax=True, store_output=False):
        x = self.avg_pool(self.relu(self.conv1(input_image)))
        x = self.avg_pool(self.relu(self.conv2(x)))
        x = self.avg_pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        logits = self.dense(x)

        if store_output:
            self.last_logits = logits

        # magnitude_outputs = self.softmax(x[:, :3])
        # parity_outputs = self.softmax(x[:, 3:])
        # return torch.cat((magnitude_outputs, parity_outputs), dim=1)

        if apply_softmax:
            return self.softmax(logits)
        return logits  # return the logits in the pre-training setting from individual images.


def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return probabilities


class NeuralSFA(nn.Module):
    def __init__(self,
                 num_states,
                 num_guards,
                 all_trainable=False,
                 nesy_mode=True,
                 with_backward_transitions=True,
                 cnn_out_features=10,
                 with_prior_model=False):

        super(NeuralSFA, self).__init__()
        self.num_states = num_states
        self.num_rules = num_guards
        self.all_trainable = all_trainable
        self.nesy_mode = nesy_mode
        self.with_backward_transitions = with_backward_transitions
        self.cnn_out_features = cnn_out_features
        self.with_prior_model = with_prior_model
        self.device = device
        self.states_distribution = torch.zeros(self.num_states).to(self.device)
        self.cnn = DigitCNN(out_features=self.cnn_out_features)

        # Initialize the transition matrices as learnable parameters. These are the 'per-guard' transition matrices,
        # i.e. the entries in each such matrix P_r are the probabilities of moving from state to state with guard r.
        # Note that these are (num_states - 1) x num_states matrices. We omit the last rows, corresponding to outgoing
        # transitions from the accepting state, since we assume that the accepting state is absorbing.
        if not self.all_trainable:
            self.transition_matrices = nn.ParameterList([nn.Parameter(
                torch.softmax(torch.randn(num_states - 1, num_states), dim=1)) for _ in range(num_guards)])
        else:
            self.transition_matrices = nn.ParameterList([nn.Parameter(
                torch.softmax(torch.randn(num_states, num_states), dim=1)) for _ in range(num_guards)])

        # ========================================
        # Disable learning of the automaton
        if self.with_prior_model:
            for param in self.transition_matrices:
                param.requires_grad = False
        # ========================================

        if not self.with_backward_transitions:
            self.disable_backward_transitions()

        # Initialize a fixed non-trainable tensor for the last row of each matrix, corresponding to the outgoing
        # transitions from the accepting state. This is an all-zeroes vector, with 1.0 in its final entry,
        # representing the fact that a self-loop on the accepting state can be taken by any one guard.
        # This tensor is not part of the ParameterList, so it won't be updated during training. It is
        # concatenated to each trainable per-guard transition matrix in the forward method to correctly perform
        # matmuls for computing the effective transition matrix.
        if not self.all_trainable:
            self.accepting_state_row = torch.zeros(num_states).to(self.device)
            self.accepting_state_row[-1] = 1.0

    def disable_backward_transitions(self):
        for matrix in self.transition_matrices:
            for i in range(self.num_states - 1):
                for j in range(i):
                    matrix.data[i, j] = 0.0

    def adjust_backward_transition_gradients(self):
        for matrix in self.transition_matrices:
            with torch.no_grad():  # Operate on the data without tracking history
                for i in range(self.num_states - 1):
                    for j in range(i):
                        if matrix.grad is not None:
                            matrix.grad[i, j] = 0.0  # Zero out gradients for backward transitions

    def get_effective_transition_matrix(self, rule_probs, softmax_temp):
        effective_transition_matrix = torch.zeros(self.num_states, self.num_states, device=rule_probs.device)
        for i, prob in enumerate(rule_probs):
            # Apply softmax to each matrix to ensure it's stochastic
            stochastic_matrix = softmax_with_temperature(self.transition_matrices[i], temperature=softmax_temp)

            if not self.all_trainable:
                # Create a complete matrix for each guard by appending the fixed last row to the trainable part.
                complete_matrix = torch.cat((stochastic_matrix, self.accepting_state_row.unsqueeze(0)), dim=0)
            else:
                complete_matrix = stochastic_matrix

            effective_transition_matrix += prob * complete_matrix
        return effective_transition_matrix

    """
    def get_effective_transition_matrix(self, rule_probs, softmax_temp):
        effective_transition_matrix = torch.zeros(self.num_states, self.num_states, device=rule_probs.device)
        for i, prob in enumerate(rule_probs):
            matrix = self.transition_matrices[i]

            if not self.all_trainable:
                # Create a complete matrix for each guard by appending the fixed last row to the trainable part.
                complete_matrix = torch.cat((matrix, self.accepting_state_row.unsqueeze(0)), dim=0)
            else:
                complete_matrix = matrix

            effective_transition_matrix += prob * complete_matrix
        # return effective_transition_matrix
        return softmax_with_temperature(effective_transition_matrix, temperature=softmax_temp)
    """

    @staticmethod
    def update_state_distribution(state_prob, effective_transition_matrix):
        return torch.matmul(state_prob, effective_transition_matrix)

    def forward(self, sequence, softmax_temp=0.1):
        self.states_distribution = torch.zeros(self.num_states).to(self.device)
        self.states_distribution[0] = 1.0  # Initially we're in the start state with probability 1.0
        cnn_predictions, guards_predictions = [], []
        sequence = sequence if self.nesy_mode else sequence.squeeze(0)
        for image in sequence:
            if self.nesy_mode:
                # digit_distribution, _ = self.cnn(image)
                digit_distribution = self.cnn(image)
                guards_distribution = digit_probs_to_rule_probs(self.num_rules, digit_distribution)
                cnn_predictions.append(digit_distribution)
                guards_predictions.append(guards_distribution)
                effective_trans_matrix = self.get_effective_transition_matrix(guards_distribution, softmax_temp)
                self.states_distribution = self.update_state_distribution(self.states_distribution,
                                                                          effective_trans_matrix)
            else:
                digit = image.item()
                # Create an one-hot vector with 1.0 in the position of the digit
                # one_hot_vector = torch.zeros(10)
                # one_hot_vector[digit] = 1.0

                one_hot_vector = digit_probs_to_rule_probs_no_nesy(digit)
                effective_trans_matrix = self.get_effective_transition_matrix(one_hot_vector, softmax_temp)
                self.states_distribution = self.update_state_distribution(self.states_distribution,
                                                                          effective_trans_matrix)
        return cnn_predictions, guards_predictions, self.states_distribution


class NonTrainableNeuralSFA(nn.Module):

    def __init__(self,
                 sdd_builder,
                 num_states,
                 cnn_out_features=10):
        super().__init__()
        self.sdd_builder = sdd_builder
        self.num_states = num_states
        self.cnn_out_features = cnn_out_features
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.states_distribution = torch.zeros(self.num_states).to(self.device)
        self.cnn = DigitCNN(out_features=self.cnn_out_features)

    def eval_polynomials(self, nn_output):
        sdd_vars = self.sdd_builder.indexed_vars.keys()
        var_names_to_probs = {var: prob for var, prob in zip(sdd_vars, nn_output.squeeze(0))}
        return {guard: polynomial.eval(var_names_to_probs) for
                guard, polynomial in self.sdd_builder.polynomials.items()}

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

    def get_guards_probabilities(self, nn_output):
        sdds = self.sdd_builder.circuits
        guards_probabilities = {}
        for guard_name, sdd in sdds.items():
            guards_probabilities[guard_name] = self.get_guard_prob(sdd, nn_output)
        return guards_probabilities

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

    def forward(self, sequence):
        self.states_distribution = torch.zeros(self.num_states).to(self.device)
        self.states_distribution[0] = 1.0  # Initially we're in the start state with probability 1.0
        cnn_predictions, guards_predictions, state_dist = [], [], []

        seq_size = sequence.shape[1]  # Keep Sequence length for later
        # Make the sequence of size (batch_size * seq_len, 1, 28, 28)
        sequence = sequence.view(-1, sequence.shape[2], sequence.shape[3], sequence.shape[4])

        start_time = time.time()
        nn_outputs = self.cnn(sequence)

        # This uses the propagate_guard method in this class, which computes the value of the polynomial
        # function that corresponds to the SDD when its input variables are replaced by the specific
        # probabilities returned by the CNN, by converting the SDD into the polynomial on the fly.
        # To use this the sdd_builder that is passed to the NonTrainableNeuralSFA constructor needs to be
        # called with clear_fields=false, as in
        # sdd_builder = SDDBuilder(asp_program,
        #                          vars_names=['d'],
        #                          categorical_vars=['d'],
        #                          clear_fields=false)
        # so that the actual SDDs are retained in the sdd_builder instance.

        # guards_probabilities = self.get_guards_probabilities(nn_output)

        # This uses the actual polynomials instead of the SDDs and evaluates the polynomials
        # with the input vars probabilities. It is slightly faster that the method above.

        # The nn_outputs is of size (batch_size * seq_len, 10)
        for i, nn_output in enumerate(nn_outputs, start=1):
            guards_probabilities = self.eval_polynomials(nn_output.cpu())
            transition_matrix = self.get_transition_matrix(guards_probabilities)
            transition_matrix = transition_matrix.to(self.device)
            cnn_predictions.append(nn_output)
            guards_predictions.append(guards_probabilities)
            self.states_distribution = self.update_state_distribution(
                self.states_distribution, transition_matrix)
            # Keep distribution for every batch
            if i % seq_size == 0:
                state_dist.append(self.states_distribution)

                # clean the states probability vector (nkatz - 13/12/2024)
                # self.states_distribution = torch.zeros(self.num_states).to(self.device)
                # self.states_distribution[0] = 1.0
        end_time = time.time()
        # print(f'{end_time - start_time} sec')
        state_dist = torch.stack(state_dist, dim=0)
        return cnn_predictions, guards_predictions, state_dist


class NonTrainableNeuralSFA_roadR(nn.Module):
    def __init__(self,
                 sdd_builder,
                 num_states,
                 road_r_model,
                 out_features):
        super().__init__()
        self.sdd_builder = sdd_builder
        self.num_states = num_states
        self.out_features = out_features
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.states_distribution = torch.zeros(self.num_states).to(self.device)
        self.nn = road_r_model

    def eval_polynomials(self, nn_output):
        sdd_vars = self.sdd_builder.indexed_vars.keys()
        var_names_to_probs = {var: prob for var, prob in zip(sdd_vars, nn_output.squeeze(0))}
        return {guard: polynomial.eval(var_names_to_probs) for
                guard, polynomial in self.sdd_builder.polynomials.items()}

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

    def get_guards_probabilities(self, nn_output):
        sdds = self.sdd_builder.circuits
        guards_probabilities = {}
        for guard_name, sdd in sdds.items():
            guards_probabilities[guard_name] = self.get_guard_prob(sdd, nn_output)
        return guards_probabilities

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

    def forward(self, sequence, bboxes):
        nn_predictions, guards_predictions, state_dist = [], [], []

        seq_size = sequence.shape[2]  # Keep Sequence length for later
        # Make the sequence of size (batch_size * seq_len, rgb-channels, width, height)
        # sequence = sequence.view(-1, sequence.shape[2], sequence.shape[3], sequence.shape[4])

        start_time = time.time()
        nn_outputs = self.nn(sequence, bboxes)
        nn_outputs = nn_outputs.reshape(-1, seq_size, nn_outputs.shape[1])
        # The nn_outputs is of size (batch_size, seq_len, prediction_size)

        for nn_output in nn_outputs:

            self.states_distribution = torch.zeros(self.num_states).to(self.device)
            self.states_distribution[0] = 1.0  # Initially we're in the start state with probability 1.0

            nn_output = nn_output.squeeze()

            for frame in nn_output:
                guards_probabilities = self.eval_polynomials(frame.cpu())
                transition_matrix = self.get_transition_matrix(guards_probabilities)
                transition_matrix = transition_matrix.to(self.device)
                nn_predictions.append(frame)
                guards_predictions.append(guards_probabilities)
                self.states_distribution = self.update_state_distribution(
                    self.states_distribution, transition_matrix)
            # Keep distribution for every batch
            state_dist.append(self.states_distribution)
        end_time = time.time()
        # print(f'{end_time - start_time} sec')
        state_dist = torch.stack(state_dist, dim=0)
        return nn_predictions, guards_predictions, state_dist
