import sys
import clingo
from clingo import Number
from clingo.script import enable_python
import clingo.ast
from clingox.reify import reify_program
import multiprocessing
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os
from collections import Counter

diverse_models_meta_encoding = \
    """
model(1..m).

conjunction(B,M) :- model(M), literal_tuple(B),
        hold(L,M) : literal_tuple(B, L), L > 0;
    not hold(L,M) : literal_tuple(B,-L), L > 0.

body(normal(B),M) :- rule(_,normal(B)), conjunction(B,M).
body(sum(B,G),M)  :- model(M), rule(_,sum(B,G)),
    #sum { W,L :     hold(L,M), weighted_literal_tuple(B, L,W), L > 0 ;
           W,L : not hold(L,M), weighted_literal_tuple(B,-L,W), L > 0 } >= G.

  hold(A,M) : atom_tuple(H,A)   :- rule(disjunction(H),B), body(B,M).
{ hold(A,M) : atom_tuple(H,A) } :- rule(     choice(H),B), body(B,M).

show(T,M) :- output(T,B), conjunction(B,M).
#show.
#show (T,M) : show(T,M).

#const k=1.

:- model(M), model(N), M<N, option = 1,
    #sum{ 1,T: show(T,M), not show(T,N) ;
          1,T: not show(T,M), show(T,N) } < k.

#maximize{ 1,T,M,N: show(T,M), not show(T,N), model(N), option = 2 }.
"""

program = \
    """
% Automata interpreter
inState(1,T) :- seqStart(T).
inState(S2,T+1) :- inState(S1,T), transition(S1,f(S1,S2),S2), holds(f(S1,S2),T).
accepted :- inState(S,T), accepting(S), seqEnd(T).
reach_accept_at(T) :- inState(S,T), accepting(S); #false: inState(S,T1), T1 < T.

seqStart(1).
seqEnd(T+1) :- time(T), not time(T+1).
% time(1..10).  % increase the end point to get longer seqs
time(1..50).
digit(0..9).

% :- reach_accept_at(T), not seqEnd(T).  % reach acceptance only at the end of the sequence

% Automaton
%*
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). 
transition(2,f(2,3),3). transition(3,f(3,3),3). transition(3,f(3,4),4). 
transition(4,f(4,4),4).

holds(f(1,2),T) :- holds(equals(d1,even),T), holds(equals(d2,gt_6),T).
holds(f(2,3),T) :- holds(equals(d2,odd),T), holds(equals(d1,leq_6),T).
holds(f(3,4),T) :- holds(equals(d1,leq_3),T), holds(equals(d2,gt_5),T).
holds(f(3,4),T) :- holds(equals(d3,leq_3),T).
holds(f(1,1),T) :- time(T), not holds(f(1,2),T).
holds(f(2,2),T) :- time(T), not holds(f(2,3),T).
holds(f(4,4),T) :- time(T).
holds(f(3,3),T) :- time(T), not holds(f(3,4),T).
*%


accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). 
transition(2,f(2,3),3). transition(3,f(3,3),3). transition(3,f(3,4),4). 
transition(4,f(4,4),4).

holds(f(1,2),T) :- holds(equals(d1,even),T), holds(equals(d1,gt_6),T).
holds(f(2,3),T) :- holds(equals(d1,odd),T), holds(equals(d1,leq_6),T).
holds(f(3,4),T) :- holds(equals(d1,leq_3),T).
holds(f(1,1),T) :- time(T), not holds(f(1,2),T).
holds(f(2,2),T) :- time(T), not holds(f(2,3),T).
holds(f(3,3),T) :- time(T), not holds(f(3,4),T).

holds(f(4,4),T) :- time(T).  % No over-general self loop to avoid having acceptance prob. increase indefinitely in the experiments.
% holds(f(4,4),T) :- holds(equals(d1,leq_3),T).
% holds(f(4,1),T) :- time(T), not holds(f(4,4),T).


%*
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). 
transition(2,f(2,3),3). transition(3,f(3,3),3). transition(3,f(3,4),4). 
transition(3,f(3,1),1). transition(2,f(2,1),1). 
transition(4,f(4,4),4).

holds(f(1,2),T) :- holds(equals(d1,even),T), holds(equals(d2,gt_6),T).
holds(f(2,3),T) :- holds(equals(d2,odd),T), holds(equals(d1,leq_6),T).
holds(f(3,4),T) :- holds(equals(d1,leq_3),T), holds(equals(d2,gt_5),T).
holds(f(3,4),T) :- holds(equals(d2,odd),T).
holds(f(3,1),T) :- holds(equals(d1,odd),T), holds(equals(d2,even),T).
holds(f(2,1),T) :- holds(equals(d1,odd),T), holds(equals(d2,even),T).
holds(f(1,1),T) :- time(T), not holds(f(1,2),T).
holds(f(2,2),T) :- time(T), not holds(f(2,3),T).
holds(f(4,4),T) :- time(T).
holds(f(3,3),T) :- time(T), not holds(f(3,4),T).
*%

% Predicate definitions
holds(equals(D,even),T) :- seq(obs(D,X),T), X \ 2 = 0.
holds(equals(D,odd),T) :- seq(obs(D,X),T), X \ 2 != 0.
holds(equals(D,gt_6),T) :- seq(obs(D,X),T), X > 6.
holds(equals(D,leq_6),T) :- seq(obs(D,X),T), X <= 6.
holds(equals(D,gt_3),T) :- seq(obs(D,X),T), X > 3.
holds(equals(D,leq_3),T) :- seq(obs(D,X),T), X <= 3.
holds(equals(D,gt_5),T) :- seq(obs(D,X),T), X > 5.

1 {seq(obs(d1,X),T): digit(X)} 1 :- time(T).
% 1 {seq(obs(d2,X),T): digit(X)} 1 :- time(T).
% 1 {seq(obs(d3,X),T): digit(X)} 1 :- time(T).

#show.
#show seq/2.
"""

positive_constraint = ":- not accepted."
negative_constraint = ":- accepted."


class SeqAtom:
    def __init__(self, digit_id, digit_value, time, model_num):
        self.digit_id = digit_id
        self.digit_value = digit_value
        self.time = time
        self.model_num = model_num


class DigitSequence:
    def __init__(self, seq_id, seq_dict: dict[str, list[str]], seq_label):
        self.seq_id = seq_id
        self.sequence = seq_dict
        self.seq_attributes = list(seq_dict.keys())
        self.seq_label = seq_label

    def length(self):
        return len(list(self.sequence.values())[0])

    def get_sequence(self):
        """used for calculating hamming distances"""
        if len(self.seq_attributes) == 1:  # univariate sequence
            return self.sequence[list(self.seq_attributes)[0]]
        else:
            return list(zip(*self.sequence.values()))

    def to_string(self):
        result = []
        for k in self.seq_attributes:
            seq = self.sequence[k]
            to_str = f"""{k} {" ".join([str(x) for x in seq])}"""
            result.append(to_str)
        return '\n'.join(result)

    def to_asp(self):
        result = []
        for k in self.seq_attributes:
            seq = self.sequence[k]
            sub_seq = [f'seq({self.seq_id},obs({k},{digit}),{time}).' for time, digit in enumerate(seq)]
            sub_seq.append(f'class({self.seq_id},{self.seq_label}).')
            result.append(' '.join(sub_seq))
        return '\n'.join(result)


def hamming(seq1: DigitSequence, seq2: DigitSequence):
    assert len(seq1.seq_attributes) == len(seq2.seq_attributes)
    return sum(x != y for x, y in zip(seq1.get_sequence(), seq2.get_sequence()))


class MNISTSeqGenerator:
    def __init__(self, args):
        self.seq_count = args.sequence_counter
        self.asp_program = args.asp_program
        self.wait = args.async_wait
        self.generate_positives = args.generate_positives
        self.label = 1 if self.generate_positives else 0
        self.show = args.show
        self.generate_constraint = positive_constraint if self.generate_positives else negative_constraint
        cores_num = multiprocessing.cpu_count()
        self.cores = f'-t{cores_num if cores_num <= 64 else 64}'
        self.seq_target_num = args.seq_num
        self.generated_sequences = []
        self.handle = None

    @staticmethod
    def group_seq_atoms(seq_atoms):
        # Step 1: Group by model_num
        model_groups = defaultdict(list)
        for atom in seq_atoms:
            model_groups[atom.model_num].append(atom)

        # Step 2 & 3: For each model, group by digit_id and sort by time
        result = {}
        for model_num, atoms in model_groups.items():
            digit_groups = defaultdict(list)
            for atom in atoms:
                digit_groups[atom.digit_id].append(atom)

            # Sort by time and collect digit_value
            digit_values = {
                digit_id: [a.digit_value for a in sorted(atoms, key=lambda a: a.time)]
                for digit_id, atoms in digit_groups.items()
            }

            result[model_num] = digit_values

        return result

    def keep_sequence(self, s: DigitSequence):
        s.seq_id = self.seq_count
        self.generated_sequences.append(s)
        self.seq_count += 1

    def parse_solver_results(self, model: list[clingo.Symbol]):
        seq_atoms = []
        for atom in model:
            seq_atom = atom.arguments[0]  # e.g. seq(d(1),5)
            model_num = int(str(atom.arguments[1]))
            time_atom = int(str(seq_atom.arguments[1]))
            digit_atom = seq_atom.arguments[0]
            digit_name, digit_value = str(digit_atom.arguments[0]), int(str(digit_atom.arguments[1]))
            seq_atom = SeqAtom(digit_name, digit_value, time_atom, model_num)
            seq_atoms.append(seq_atom)

        group_seq_atoms = self.group_seq_atoms(seq_atoms)

        # set a dummy (0) id here, the actual one (from self.seq_count) will be set if the sequence will be
        # added to the set of the generated sequences, based on its mean Hamming Distance from existing seqs.
        sequences = map(lambda s: DigitSequence(0, s, self.label), list(group_seq_atoms.values()))

        if not self.generated_sequences:
            for s in sequences:
                self.keep_sequence(s)
        else:
            for seq in sequences:
                mean_hamming = sum(hamming(seq, s) for s in self.generated_sequences) / len(self.generated_sequences)
                # if mean_hamming > seq.length() - (1 / 5 * seq.length()):
                if True:
                    self.keep_sequence(seq)
                    if self.show != 'no':
                        if self.show != "asp":
                            print(f'Sequence {seq.seq_id}: '
                                  f'{seq.to_string()} Avg Hamming: {mean_hamming:.3f}')
                        else:
                            print(f'Sequence:\n{seq.to_asp()} Avg Hamming: {mean_hamming:.3f}')

    def on_model(self, model: clingo.solving.Model):
        atoms = [atom for atom in model.symbols(shown=True)]
        self.parse_solver_results(atoms)
        if len(self.generated_sequences) > self.seq_target_num:
            print(f'Generated {len(self.generated_sequences)} sequences.')
            # sys.exit()
            self.handle.cancel()  # this hangs and the program does not terminate...

    def get_models(self):
        enable_python()
        # Step 1: Reify the SFA program
        symbols = reify_program('\n'.join([self.asp_program, self.generate_constraint]))
        reified_program = '\n'.join([str(sym) + '.' for sym in symbols])

        ctl = clingo.Control(["-Wno-atom-undefined", "-c", "m=3", "-c", "option=1", "-c", "k=20",
                              self.cores, '0'])  # up to (e.g.) 20 models: '-n20'
        ctl.add("base", [], reified_program)
        ctl.add("base", [], diverse_models_meta_encoding)
        ctl.ground([("base", [])])

        with ctl.solve(on_model=self.on_model, async_=True) as handle:
            handle.wait(self.wait)
            self.handle = handle

        # ctl.solve(on_model=self.on_model)


def count_digit_frequencies(digit_sequences: list[DigitSequence]):
    counter = Counter()
    for ds in digit_sequences:
        for dim_seq in ds.sequence.values():
            counter.update(dim_seq)
    return counter


def count_digit_frequencies_per_label(digit_sequences):
    counters = {
        'positive': Counter(),
        'negative': Counter()
    }

    for seq in digit_sequences:
        label = 'positive' if seq.seq_label == 1 else 'negative'
        for values in seq.sequence.values():
            counters[label].update(values)

    return counters

def print_label_counters(counts):
    for label in ['positive', 'negative']:
        print(f"\n{label.capitalize()} counts:")
        for digit in sorted(counts[label].keys()):
            print(f"Digit {digit}: {counts[label][digit]}")


def generate_seqs():
    """
    To generate multivariate versions: Just add extra generate directives in the program, i.e.
    1 {seq(obs(d1,X),T): digit(X)} 1 :- time(T).
    1 {seq(obs(d2,X),T): digit(X)} 1 :- time(T).
    etc.

    Modify the SFA pattern to generate datasets of positive (satisfy the pattern) and negative sequences.

    To generate sequences of different lengths, just increase the end point in the time/1 atom. For instance,
    with time(1..50). in the ASP program, sequences of length 50 will be generated.

    To verify the results, write the automaton used to generate the data in a file and run ASAL with --eval file
    and --test clingo_generated_train.lp, or --test clingo_generated_test.lp. These should return an F1-score of 1.0.

    NOTE: The automaton needs to be written in the format that ASAL expects. For example, this is how the two SFAs
    that exists in the ASP program above should be written, in order to run ASAL:

    SFA for univariate seqs:
    ------------------------
    holds(f(1,2),SeqId,T) :- holds(equals(even(d1),1),SeqId,T), holds(equals(gt_6(d1),1),SeqId,T).
    holds(f(2,3),SeqId,T) :- holds(equals(odd(d1),1),SeqId,T), holds(equals(leq_6(d1),1),SeqId,T).
    holds(f(3,4),SeqId,T) :- holds(equals(leq_3(d1),1),SeqId,T).
    holds(f(1,1),SeqId,T) :- time(T), sequence(SeqId), not holds(f(1,2),SeqId,T).
    holds(f(2,2),SeqId,T) :- time(T), sequence(SeqId), not holds(f(2,3),SeqId,T).
    holds(f(4,4),SeqId,T) :- time(T), sequence(SeqId).
    holds(f(3,3),SeqId,T) :- time(T), sequence(SeqId), not holds(f(3,4),SeqId,T).

    SFA for multivariate seqs:
    --------------------------
    holds(f(1,2),SeqId,T) :- holds(equals(even(d1),1),SeqId,T), holds(equals(gt_6(d2),1),SeqId,T).
    holds(f(2,3),SeqId,T) :- holds(equals(odd(d2),1),SeqId,T), holds(equals(leq_6(d1),1),SeqId,T).
    holds(f(3,4),SeqId,T) :- holds(equals(leq_3(d1),1),SeqId,T), holds(equals(gt_5(d2),1),SeqId,T).
    holds(f(3,4),SeqId,T) :- holds(equals(leq_3(d3),1),SeqId,T).
    holds(f(1,1),SeqId,T) :- sequence(SeqId), time(T), not holds(f(1,2),SeqId,T).
    holds(f(2,2),SeqId,T) :- sequence(SeqId), time(T), not holds(f(2,3),SeqId,T).
    holds(f(4,4),SeqId,T) :- sequence(SeqId), time(T).
    holds(f(3,3),SeqId,T) :- sequence(SeqId), time(T), not holds(f(3,4),SeqId,T).
    """
    args = argparse.Namespace(
        sequence_counter=0,
        asp_program=program,
        generate_positives=True,  # Generate positive or negative sequences
        seq_num=1000,  # How many sequences to generate
        show='no',  # The format to print the generated sequences in (options: no, asp, lists)
        async_wait=10,  # increase this for intensive tasks, e.g. large hamming distance
    )
    print('Generating positives')
    generator = MNISTSeqGenerator(args)
    generator.get_models()
    positives = generator.generated_sequences

    print(len(positives), generator.seq_count)
    assert len(positives) == generator.seq_count

    sequence_counter = generator.seq_count + 1

    print('Generating negatives')
    args.generate_positives = False
    args.sequence_counter = sequence_counter
    generator = MNISTSeqGenerator(args)
    generator.get_models()
    negatives = generator.generated_sequences

    # train/test split
    all_seqs = positives + negatives
    labels = [s.seq_label for s in all_seqs]
    train, test = train_test_split(all_seqs, stratify=labels, train_size=0.7, random_state=42)
    print('train pos:', len([s for s in train if s.seq_label == 1]), 'train negs:',
          len([s for s in train if s.seq_label == 0]))
    print('test pos:', len([s for s in test if s.seq_label == 1]), 'test negs:',
          len([s for s in test if s.seq_label == 0]))

    print('Distribution in training set:')
    counts_train = count_digit_frequencies(train)
    print(counts_train)
    print('Distribution in testing set:')
    counts_test = count_digit_frequencies(test)
    print(counts_test)
    print('Distribution per label in whole dataset:')
    counts_per_label_pos = count_digit_frequencies_per_label(positives)
    counts_per_label_neg = count_digit_frequencies_per_label(negatives)
    print(counts_per_label_pos)
    print(counts_per_label_neg)

    train_file = '../../../../data/mnist_nesy/double_digit/clingo_generated_train.lp'
    test_file = '../../../../data/mnist_nesy/double_digit/clingo_generated_test.lp'
    train_csv_file = '../../../../data/mnist_nesy/double_digit/clingo_generated_train.csv'
    test_csv_file = '../../../../data/mnist_nesy/double_digit/clingo_generated_test.csv'

    with open(train_file, 'w') as f1:
        for seq in train:
            f1.write(seq.to_asp())
            f1.write('\n\n')

    with open(test_file, 'w') as f2:
        for seq in test:
            f2.write(seq.to_asp())
            f2.write('\n\n')

    with open(train_csv_file, 'w') as f3:
        for seq in train:
            f3.write(seq.to_string())
            f3.write('\n')

    with open(test_csv_file, 'w') as f4:
        for seq in train:
            f4.write(seq.to_string())
            f4.write('\n')

    return train, test


if __name__ == '__main__':
    generate_seqs()


