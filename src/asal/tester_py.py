import itertools
from src.asal.structs import Automaton, SeqAtom, ClassAtom, MultiVarSeq
import clingo


class TesterPy:
    """Tests e model in an imperative fashion (no Clingo reasoning)"""

    def __init__(self, batch_data, target_class, automaton, path_scoring):
        self.batch_data_asp = batch_data
        self.target_class = target_class
        self.automaton: Automaton = automaton
        self.path_scoring = path_scoring
        self.mvar_seqs: dict[int, MultiVarSeq] = {}
        self.ctl = clingo.Control()

    @staticmethod
    def parse_atoms(model: list[clingo.Symbol]):
        seq_atoms, class_atoms = [], []
        for atom in model:
            if atom.match("seq", 3):
                seq_atoms.append(SeqAtom(atom))
            elif atom.match("class", 2):
                class_atoms.append(ClassAtom(atom))
            else:
                pass
        return seq_atoms, class_atoms

    def get_mvar_seqs(self, model):
        atoms = [atom for atom in model.symbols(shown=True)]
        seq_atoms, class_atoms = self.parse_atoms(atoms)
        class_dict = dict([(a.seq_id, a.class_id) for a in class_atoms])
        multi_var_seqs: dict[int, MultiVarSeq] = {}
        for atom in seq_atoms:
            if atom.seq_id in multi_var_seqs.keys():
                seq_obj = multi_var_seqs[atom.seq_id]
                if atom.attribute in seq_obj.sequences.keys():
                    seq_obj.sequences[atom.attribute].append((atom.time, atom.att_value))
                else:
                    seq_obj.sequences[atom.attribute] = [(atom.time, atom.att_value)]
            else:
                seq_obj = MultiVarSeq(atom.seq_id)
                seq_obj.sequences[atom.attribute] = [(atom.time, atom.att_value)]
                multi_var_seqs[atom.seq_id] = seq_obj
        for seq_id, seq in multi_var_seqs.items():
            seq.sort_seqs()
            seq.set_class(class_dict[seq_id])
        self.mvar_seqs = multi_var_seqs

    def convert_seqs(self):
        """Converts the sequences in batch_data into simple symbol seqs. For instance the multivariate seq:

           seq(s1, alive(a), 0), seq(s1, alive(f), 1),...,seq(s1, alive(d), 50)
           seq(s1, necrotic(b), 0), seq(s1, necrotic(c), 1),...,seq(s1, necrotic(a), 50)
           seq(s1, apoptotic(c), 0), seq(s1, apoptotic(d), 1),...,seq(s1, apoptotic(e), 50)
           class(s1, 1)

           is converted into:

           alive,a,f,...,d
           necrotic,b,c,...,a
           apoptotic,c,d,...,c
           """
        self.ctl.add("base", [], self.batch_data_asp)
        self.ctl.add("base", [], '#show seq/3. #show class/2.')
        self.ctl.ground([("base", [])])
        self.ctl.solve(on_model=self.get_mvar_seqs)
        return self.mvar_seqs
