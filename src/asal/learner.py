from src.asal.structs import Automaton, SolveResult, GuardBodyAtom, TransitionAtom, AcceptingStateAtom, CountsAtom, \
    AnyAtom, ScoredModel
from src.asal.test_model_multproc import test_model_mproc
from src.asal.template import Template
from src.asal.tester import rewrite_automaton
from src.asal.constraints import generate_constraints
from src.asal.logger import *
import multiprocessing
import clingo
from clingo import Number
import time
import os
from clingo.script import enable_python
from clingo.symbol import parse_term
from src.asal.asp import get_induction_program
from src.asal.auxils import timer


class Learner:
    def __init__(self,
                 template: Template,
                 mini_batch: str,
                 args,
                 existing_model=Automaton(),
                 constraints=None,
                 debug=False,
                 mode='reasoning',
                 with_joblib=False):

        self.args = args
        self.constraints = constraints  # extra constraints (passed as a list of strings), to be added to the program.
        self.template = template
        self.induction_program = get_induction_program(args, self.template)
        self.target_class = template.target_class
        self.max_states = template.max_states
        self.mini_batch = mini_batch
        self.time_limit = float('inf') if args.tlim == 0 else args.tlim
        self.start_time = time.time()
        self.existing_model = existing_model
        self.mode = mode
        self.debug = debug
        self.induced_models = []
        self.generated_automata = []

        cores_num = multiprocessing.cpu_count()
        cores = f'-t{cores_num if cores_num <= 64 else 64}'  # 64 threads is the implementation-defined limit for Clingo

        if args.warns_off:
            self.ctl = clingo.Control([cores, "--warn=none"])
        else:
            self.ctl = clingo.Control([cores])  # clingo.Control([cores, "--warn=none"])  "--warn=no-atom-undefined"

        self.grounding_time = 0.0
        self.solving_time = 0.0
        self.with_joblib = with_joblib

    @staticmethod
    def parse_solver_results(model: list[clingo.Symbol]):
        output = []
        for atom in model:
            if atom.match("transition", 3):
                output.append(TransitionAtom(atom))
            elif atom.match('accepting', 1):
                output.append(AcceptingStateAtom(atom))
            elif atom.match("body", 3):
                output.append(GuardBodyAtom(atom))
            elif atom.match("tps", 1) or atom.match("fps", 1) or atom.match("fns", 1):
                output.append(CountsAtom(atom))
            else:
                output.append(AnyAtom(atom))
        return output

    @timer
    def on_model(self, model):
        atoms = [atom for atom in model.symbols(shown=True)]
        parsed_atoms = self.parse_solver_results(atoms)

        fsm = Automaton(parsed_atoms, self.template)
        fsm.cost = model.cost
        fsm.optimality_proven = model.optimality_proven

        if len(fsm.asp_model) >= 1 and len(fsm.transitions) > 0:
            self.induced_models.append(fsm)

        if True:  # show all generated models
            logger.debug('Found model with cost {0} ({1}):\n{2}\nReturned model is:\n{3}'.
                         format(fsm.cost, ' '.join(fsm.local_performance),
                                fsm.show(mode=self.mode), list(map(lambda x: x.str, fsm.asp_model))))

        """
        # Save them as objects for debugging.
        pickle_file_name = f'rules-size-{len(fsm.rules)}'
        file = open(f'debug/pickled_fsms/{pickle_file_name}', 'wb')
        pickle.dump(fsm, file)
        file.close()
        """

    @timer
    def ground(self, file=None):
        logger.info('Grounding...')
        start = time.time()
        if file is None:
            self.ctl.ground([("base", [])])
        else:
            self.ctl.ground(file)
        end = time.time()
        grounding_time = end - start
        self.grounding_time = grounding_time

    @timer
    def solve(self):
        logger.info('Solving with time limit {0}'.format(self.time_limit))
        start = time.time()
        if self.time_limit == float('inf'):
            self.ctl.solve(on_model=self.on_model)
        else:
            with self.ctl.solve(on_model=self.on_model, async_=True) as handle:
                handle.wait(self.time_limit)
                handle.cancel()
        end = time.time()
        solving_time = end - start
        self.solving_time = solving_time

    def show_model(self, model):
        if not self.debug:
            logger.debug('\nFound model with cost {0} ({1}):\n{2}\nThe corresponding answer set '
                         'is:\n{3}'.
                         format(model.cost, ' '.join(model.local_performance),
                                model.show(mode=self.mode),
                                ' '.join(list(map(lambda x: x.str + '.', model.asp_model)))))

    @staticmethod
    def __get_existing(existing_body_atoms: list[GuardBodyAtom]):
        """Adds an existing model into the base program."""
        existing = []
        for atom in existing_body_atoms:
            # Example atom signature: body(f(1,3),1,at_most(latitude,d))
            rule_head_id = atom.guard_head
            conjunction_id = atom.body_id
            from_state = rule_head_id.split(',')[0].split('(')[1]
            to_state = rule_head_id.split(',')[1].split(')')[0]

            rule_atom = f'rule({rule_head_id}).'
            conj_atom = f'conjunction({conjunction_id}).'
            transition_atom = f'transition({from_state},{rule_head_id},{to_state}).'
            whole_atom = atom.str + '.'

            # self.ctl.add("base", [], whole_atom)
            # self.ctl.add("base", [], rule_atom)
            # self.ctl.add("base", [], conj_atom)
            # self.ctl.add("base", [], transition_atom)

            # existing.extend([whole_atom, rule_atom, conj_atom, transition_atom])

            choice_atom = f"existing({atom.str})."
            existing.append(choice_atom)

        rule_constraint = f":- body(Guard,_,_), not rule(Guard)."
        conj_constraint = f":- body(_,Conj,_), not conjunction(Conj)."
        transition_constraint = f":- body(f(I,J),_,_), not transition(I,f(I,J),J)."
        minimize_statement = "#minimize{1@0,F: existing(body(G,C,F)), not body(G,C,F)}."
        existing.extend([rule_constraint, conj_constraint, transition_constraint, minimize_statement])
        return existing

    def induce_models(self, all_opt=False):
        enable_python()
        """
        if 'src' in os.getcwd():
            induction_file = os.path.normpath(os.getcwd() + os.sep + 'asp' + os.sep + 'induce.lp')
        else:
            induction_file = os.path.normpath(os.getcwd() + os.sep + 'src' + os.sep + 'asp' + os.sep + 'induce.lp')

        self.ctl.load(induction_file)
        self.ctl.add("base", [], self.mini_batch)

        if self.constraints is not None:
            self.ctl.add("base", [], '\n'.join(self.constraints))

        if not self.existing_model.is_empty:
            # body_3_facts = ' '.join(list(map(lambda x: f':~ not {x.str}. [5@5]', self.existing_model.body_atoms_asp)))
            self.add_existing(self.existing_model.body_atoms_asp)
            # body_3_facts = ' '.join(list(map(lambda x: f'{x.str}.', self.existing_model.body_atoms_asp)))
            # self.ctl.add("base", [], body_3_facts)

        induction_program = [("base", []), ("class", [Number(int(self.target_class))])]
        """
        self.ctl.add("base", [], self.induction_program)
        self.ctl.add("base", [], self.mini_batch)

        if not self.existing_model.is_empty:
            # body_3_facts = ' '.join(list(map(lambda x: f':~ not {x.str}. [5@5]', self.existing_model.body_atoms_asp)))
            # body_3_facts = ' '.join(list(map(lambda x: f'{x.str}.', self.existing_model.body_atoms_asp)))
            # self.ctl.add("base", [], body_3_facts)
            existing = self.__get_existing(self.existing_model.body_atoms_asp)
            logger.debug(f'Induction program (with prior model) is:'
                         f'\n{get_induction_program(self.args, self.template, existing)}')

            if self.args.revise:
                for atom in existing:
                    self.ctl.add("base", [], atom)

        models = None

        if all_opt:
            self.ctl.configuration.solve.opt_mode = 'optN'  # optimality_proven is set to true only in this case.
        else:
            self.ctl.configuration.solve.opt_mode = 'opt'  # default in Clingo.

        self.ground()
        # self.ground(file=induction_program)  # use this if you're loading from a file (as in the commented code above)
        self.solve()

        """
        if sols_num == '1':  # Return one, the best found within the time limit.
            models = self.induced_models[0]
            self.show_model(models)
        elif sols_num == '0':  # Return All optimal (found within the time limit).
            models = self.induced_models
        """

        models = self.induced_models
        return SolveResult(models, self.grounding_time, self.solving_time, with_joblib=self.with_joblib)

    def induce_multiple(self, sols_num):
        """Generate a hypothesis A within the time limit. Then ask for up to sols_num
        hypotheses with cost = A.cost. This is currently used by MCTS. """

        model = self.induce_models()

        cost_str = ','.join(list(map(lambda x: str(x), model.cost)))
        self.ctl.configuration.solve.opt_mode = f'enum,{cost_str}'
        self.ctl.configuration.solve.models = str(sols_num)
        self.solve()
        models = self.induced_models
        for m in models:
            print(m.cost)
            print(m.show(mode='reasoning'))

        return models

    """
    def induce_iteratively(self, train_path, target_class,
                           mini_batch_size, min_precision,
                           current_global_model, local_iterations):

        # Generates automata, scores them on the training set and generates path constraints based
        # on min_precision. This needs work for correctly passing the constraints. One way is to
        # define body/3 atoms as external and add them as constraints via e.g.
        #
        # ctl.assign_external(body(f(1,3),1,at_most(latitude,d)), False).
        #
        # There are three caveats for this: first the atom cannot be passed as a string, but as clingo atom.
        # This can be achieved via
        # 
        # from clingo.symbol import parse_term
        # clingo_atom = parse_term('body(f(1,3),1,at_most(latitude,d))')
        # ctl.assign_external(Function(clingo_atom.name, clingo_atom.arguments), False)
        #
        # The second caveat is that I need to deal with constraints that have conjunctions in their body
        # (these are generated by low quality transition guards with more than one atom in their body).
        # The approach above with external atoms does not suffice for that.
        #
        # The third caveat is that this method calls the testing routine from here, which means that during
        # testing the memory allocated by the solver instance is not freed. This can cause memory issues
        # with large mini-batch sizes.
        

        flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]
        generated_models: list[ScoredModel] = []
        prev_constraints = self.constraints if self.constraints is not None else []

        def score_model(m: Automaton):
            transformed_model, guards_map = rewrite_automaton(m)
            test_result = self._test_model(transformed_model, train_path,
                                           str(target_class), mini_batch_size, self.template)
            scored_paths = test_result.scored_paths
            global_counts = (test_result.get_tps(), test_result.get_fps(), test_result.get_fns())
            counts_per_batch = test_result.scores_per_batch
            m.update_seqs_store((test_result.tp_seqs, test_result.fp_seqs, test_result.fn_seqs))
            con = generate_constraints(scored_paths, guards_map, min_precision, current_global_model)
            scored_model = ScoredModel(m, con, global_counts, counts_per_batch)
            generated_models.append(scored_model)
            return con

        # Generate an initial model
        model = self.induce_models()
        constraints = score_model(model)
        if constraints:
            for i in range(local_iterations - 1):
                constraints = prev_constraints + flat_map(lambda x: x.constraints, generated_models)

                self.ctl.add("constraints", [], '\n'.join(constraints))
                self.ctl.add("constraints", [], 'p :- 0 = 1. :- not p.')
                self.ctl.ground([("constraints", [])])

                logger.info(f' Re-solving with constraints:\n{constraints}')
                self.solve()
                model = self.induced_models[0]
                self.show_model(model)
                new_constraints = score_model(model)
                if not new_constraints:  # then the last model scores above min_precision.
                    break

        best_model = max(generated_models, key=lambda x: x.global_precision())
        generated_models.remove(best_model)  # Remove to avoid adding constraints related to best_model (see next line).
        constraints = list(set(prev_constraints + flat_map(lambda x: x.constraints, generated_models)))

        # constraints should be returned too, if we need to accumulate them over time.
        return best_model
        """
