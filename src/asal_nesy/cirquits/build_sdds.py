import clingo
from clingo import Number
from clingo.script import enable_python
from pysdd.sdd import Vtree, SddManager, WmcManager, Fnf, SddNode
import nnf
import sympy
from sympy import *
from functools import reduce
import operator
from graphviz import Source
import multiprocessing
from itertools import combinations
from abc import ABC, abstractmethod
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class ArithmeticExpression(ABC):
    @abstractmethod
    def eval(self, env):
        """
        Evaluate the arithmetic expression based on the variable instantiation in `env`.
        :param env: A dictionary mapping variable names to their instantiated values.
        """
        pass

    def to_str(self):
        pass


class Const(ArithmeticExpression):
    def __init__(self, value):
        self.value = value

    def eval(self, env):
        return self.value

    def to_str(self):
        return self.value


class Var(ArithmeticExpression):
    def __init__(self, name):
        self.name = name

    def eval(self, env):
        return env[self.name]

    def to_str(self):
        return self.name


class Sum(ArithmeticExpression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, env):
        return self.left.eval(env) + self.right.eval(env)

    def to_str(self):
        return f'({self.left.to_str()} + {self.right.to_str()})'


class Prod(ArithmeticExpression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, env):
        return self.left.eval(env) * self.right.eval(env)

    def to_str(self):
        return f'{self.left.to_str()} * {self.right.to_str()}'


class SDDBuilder:
    def __init__(self, asp_program, vars_names, categorical_vars, clear_fields=True):
        """
        Builds an SDD from the models of an ASP program that represents the domain.

        Examples:
        --------------------------------------------------------------------------------------------------------
        --------------------------------------------------------------------------------------------------------
        1. In the NeSy MNIST addition experiment the domain may be represented by the following program:

        nn_input(d1). nn_input(d2).
        1 {value(D,0..9)} 1 :- nn_input(D).
        sum(Z) :- value(D1,X), value(D2,Y), D1 != D2, Z = X + Y.
        query(sum,X) :- sum(X).
        #show value/2.
        #show query/2.

        Two input variable names will be extracted, d1 and d2. Each variable has a range of possible values, which
        for the MNIST example are 0..9. Note that the constructed SDD will have one variable for each possible
        value of the d1 and d2 variables, therefore, it will have 20 variables, d1_0,...,d1_0,d2_0,...,d2_9.
        The number of the SDD variables is again dictated by the ASP program that models the domain and these
        variables are constructed from the value/2 atoms in the models of the program. query/2 instances
        represent probabilistic queries that we'll need to compute with the SDD, so for each such query an SDD
        decision node is constructed. As an example, from the following two models of the ASP program above:

        value(d1,0) value(d2,1) query(sum,1)
        value(d1,1) value(d2,0) query(sum,1)

        the SDD builder generates the query "sum_1" (corresponding to a decision node) and the SDD variables
        d1_0, d2_1, d1_1, d2_0. The formula that will be used to compute the SDD node for sum_1 will be:

        sum_1 = (d1_0 & d2_1) | (d1_1 & d2_0).

        Note that we need to add extra constraints in this case to make sure that the SDD variables are mutually
        exclusive and exhaustive. In the sum_1 example, and assuming for simplicity that d1_0, d2_1, d1_1, d2_0
        are the only SDD variables, these extra constraints are:

        c1 = (~d1_0 | ~d1_1) & (d1_0 & | d1_1)
        c2 = (~d2_0 | ~d2_1) & (d2_0 & | d2_1)

        and so the formula to build the SDD node from is

        sum_1 = ((d1_0 & d2_1) | (d1_1, d2_0)) & c1 & c2
        --------------------------------------------------------------------------------------------------------
        --------------------------------------------------------------------------------------------------------
        2. Consider the following simplified example from the ROAD_R domain:

        1 {value(a1,stop) ; value(a1,movaway) ; value(a1,movtow) ; value(a1,other)} 1.
        1 {value(a2,stop) ; value(a2,movaway) ; value(a2,movtow) ; value(a2,other)} 1.
        1 {value(l1,incomlane) ; value(l1,jun) ; value(l1,vehlane) ; value(l1,other)} 1.
        1 {value(l2,incomlane) ; value(l2,jun) ; value(l2,vehlane) ; value(l2,other)} 1.

        f(1,2) :- value(a1,movaway), not f(1,3).
        f(1,2) :- value(a2,movtow), not f(1,3).
        f(1,3) :- value(a1,stop), value(l2,vehlane).
        f(1,3) :- value(a1,stop), value(l1,vehlane).
        f(1,1) :- not f(1,2), not f(1,3).

        query(guard,f(1,2)) :- f(1,2).
        query(guard,f(1,3)) :- f(1,3).
        query(guard,f(1,1)) :- f(3,4).

        #show value/2.
        #show query/2.

        The choice rules pick one action and one location for each vehicle. The f/2-defining rules are a fragment
        of the transition guards for an automaton that accepts overtake sequences, in particular, this fragment
        is the guards for the outgoing transitions from state 1 (the start state).

        There are 4 random variables here:

        a1, a2, l1, l2, all categorical

        Each takes 4 possible values, resulting in the following SDD variables:

        a1_stop, a1_movaway, a1_movtow, a1_other
        a2_stop, a2_movaway, a2_movtow, a2_other
        l1_incomlane, l1_jun, l1_vehlane, l1_other
        l2_incomlane, l2_jun, l2_vehlane, l2_other

        Finally, we have the following queries, corresponding to SDD decision nodes:

        guard_f(1,2), guard_f(1,3), guard_f(1,1)

        The formulas to build the SDDs for each query are generated from the models as in the MNIST example.

        :param asp_program: the ASP program that represents the domain. This is passed as a string, see examples.
        :param vars_names: the names of the random variables that are input to the SDD. These are passed
         as a list of strings and must agree with the names used in the ASP program, see the examples above.
         For example, in the MNIST case the vars_names should be:

         ['d1', 'd2']

         and in the ROAD-R case they should be

         ['a1', 'a2', 'l1', 'l2']

        :param categorical_vars: The names of the input vars that are categorical (thus require extra constraints).
        Again, these should be passed as a list of strings. In the examples, since all variables are categorical,
        the input for vars_names is identical with the input for categorical_vars.
        """
        self.asp_program = asp_program
        self.vars_names = vars_names
        self.categorical_vars = categorical_vars
        self.grouped_sdd_vars = {}
        self.sdd_vars_num = None
        self.vtree = None
        self.sdd_manager = None
        self.sdd_queries = set()
        self.sdd_variables = set()
        self.queries_vars_map = {}
        self.indexed_vars = {}
        self.query_formulas_map = {}
        self.circuits = {}
        self.polynomials = {}
        cores_num = multiprocessing.cpu_count()
        cores = f'-t{cores_num if cores_num <= 64 else 64}'  # 64 threads is the limit for Clingo
        self.ctl = clingo.Control([cores])
        self.sum = lambda x, y: Sum(x, y)
        self.clear_fields = clear_fields

    def __parse_solver_results(self, model: list[clingo.Symbol]):
        model_variables, model_queries = [], []
        for atom in model:
            if atom.match("value", 2):
                variable = f'{str(atom.arguments[0])}_{str(atom.arguments[1])}'
                model_variables.append(variable)
                self.sdd_variables.add(variable)
            elif atom.match('query', 2):
                query = f'{str(atom.arguments[0])}_{str(atom.arguments[1])}'
                model_queries.append(query)
                self.sdd_queries.add(query)
            else:
                raise RuntimeError("Unexpected atom.")
        for q in model_queries:
            if q in self.queries_vars_map.keys():
                self.queries_vars_map[q].append(model_variables)
            else:
                self.queries_vars_map[q] = [model_variables]

    def on_model(self, model: clingo.solving.Model):
        atoms = [atom for atom in model.symbols(shown=True)]
        self.__parse_solver_results(atoms)

    def __get_models(self):
        enable_python()
        self.ctl.configuration.solve.models = '0'  # all models
        self.ctl.add("base", [], self.asp_program)
        self.ctl.ground([("base", [])])
        self.ctl.solve(on_model=self.on_model)
        # self.ctl.solve(on_model=print)

    def __get_var_index(self, variable):
        return self.indexed_vars[variable]

    def __get_var_index_str(self, variable):
        return str(self.indexed_vars[variable])

    def __group_and_index_sdd_vars(self):
        grouped_vars, indexed_vars = {}, {}
        for sdd_var_name in self.sdd_variables:
            variable_name = sdd_var_name.split('_')[0]
            if variable_name not in grouped_vars:
                grouped_vars[variable_name] = []
            grouped_vars[variable_name].append(sdd_var_name)
        for key in grouped_vars:
            grouped_vars[key].sort()
        grouped_vars = dict(sorted(grouped_vars.items()))
        # Index the sdd variables to get a reference to each by its name
        flattened_vars = [item for sublist in grouped_vars.values() for item in sublist]
        indexed_vars = dict(zip(flattened_vars, range(1, len(flattened_vars) + 1)))
        return grouped_vars, indexed_vars

    def __get_mutex_constraints(self):
        formulas = []
        for var_name in self.categorical_vars:
            related_sdd_vars = self.grouped_sdd_vars[var_name]
            mutex_formula = ('(' + ' & '.join([f'(~{x} | ~{y})'
                                               for x, y in combinations(related_sdd_vars, 2)]) +
                             ') & ' + f"""({' | '.join(related_sdd_vars)})""")
            formulas.append(mutex_formula)
        return formulas

    def get_sdd_var(self, name):
        return self.sdd_manager.vars[int(self.indexed_vars[name])]

    def __build_formulas_from_models(self):
        self.sdd_vars_num = len(self.sdd_variables)
        self.vtree = Vtree(var_count=self.sdd_vars_num, vtree_type="balanced")
        self.sdd_manager = SddManager.from_vtree(self.vtree)

        self.grouped_sdd_vars, self.indexed_vars = self.__group_and_index_sdd_vars()

        conjunction = lambda var_list: f"""({' & '.join(var_list)})"""
        disjunction = lambda conjunctions: f"""{' | '.join(conjunctions)}"""
        dnf_formula = lambda conjunctions_list: disjunction(list(map(conjunction, conjunctions_list)))

        mutex_formulas = self.__get_mutex_constraints() if self.categorical_vars else []
        mutex_formulas = ' & '.join(mutex_formulas)

        self.query_formulas_map = (
            dict([(query, f'({dnf_formula(dnf)}) & {mutex_formulas}')
                  for query, dnf in self.queries_vars_map.items()])
        )

    def __build_sdd(self, expr):
        match expr:
            case Or(args=args):
                return reduce(operator.or_, map(self.__build_sdd, args))
            case And(args=args):
                return reduce(operator.and_, map(self.__build_sdd, args))
            case Not(args=args):
                return ~self.__build_sdd(args[0])
            case Symbol(name=name):
                return self.sdd_manager.vars[int(self.indexed_vars[name])]
            case _:
                raise RuntimeError("Error in circuit building.")

    def __build_nnf(self, expr):
        """If the sdd param is False then the method compiles the expression into a nnf circuit."""
        match expr:
            case Or(args=args):
                return reduce(operator.or_, map(self.__build_nnf, args))
            case And(args=args):
                return reduce(operator.and_, map(self.__build_nnf, args))
            case Not(args=args):
                return ~self.__build_nnf(args[0])
            case Symbol(name=name):
                return nnf.Var(name)
            case _:
                raise RuntimeError("Error in circuit building.")

    # @staticmethod
    def sdd_to_poly(self, sdd_node, sdd_vars_to_var_names, categorical_vars):
        if sdd_node.is_false():
            return Const(torch.tensor([0]))  # .to(device))
        elif sdd_node.is_true():
            return Const(torch.tensor([1]))  # .to(device))
        elif sdd_node.is_literal():
            literal_id = abs(sdd_node.literal)
            if sdd_node.literal > 0:
                return Var(sdd_vars_to_var_names[literal_id])
            else:  # Negative literal
                if sdd_vars_to_var_names[literal_id].split('_')[0] in categorical_vars:
                    # Negative literals of categorical variables always get 1
                    return Const(torch.tensor([1]))  # .to(device))
                else:  # Binary (non-categorical variable)

                    return Sum(torch.tensor([1]), Prod(-1, Var(sdd_vars_to_var_names[literal_id])))
        elif sdd_node.is_decision():
            return reduce(
                self.sum,
                [
                    Prod(self.sdd_to_poly(prime, sdd_vars_to_var_names, categorical_vars),
                         self.sdd_to_poly(sub, sdd_vars_to_var_names, categorical_vars))
                    for prime, sub in sdd_node.elements()
                ],
            )
        else:
            raise RuntimeError("Error at pattern matching during WMC.")

    def __generate_polynomials(self):
        sdd_vars_to_var_names = {v: k for k, v in self.indexed_vars.items()}
        for q, sdd in self.circuits.items():
            self.polynomials[q] = self.sdd_to_poly(sdd, sdd_vars_to_var_names, self.categorical_vars)

    def __clear(self):
        """These variables are linked to C api objects, which cause serialization issues
        in a multiprocessing settings, where objects needs to be (de) serialized to be
        shared among processes. Since these variables are not really needed (we have all we
        need to compute probabilistic queries via the polynomial functions that correspond
        to the SDDs), we clear them here"""
        self.circuits = None
        self.vtree = None
        self.sdd_manager = None
        self.sdd_variables = None
        self.ctl = None
        self.query_formulas_map = None
        self.sum = None

    def build_nnfs(self):
        """Used for generating d-nnf circuits from a program. We just needs the formulas for that."""
        self.__get_models()
        self.__build_formulas_from_models()
        self.circuits = dict([(query, self.__build_nnf(sympy.parse_expr(formula)))
                              for query, formula in self.query_formulas_map.items()])

    def build_sdds(self):
        self.__get_models()
        self.__build_formulas_from_models()

        """
        for query, formula in self.query_formulas_map.items():
            p = sympy.parse_expr(formula)
            self.__build_sdd(p)
        """

        self.circuits = dict([(query, self.__build_sdd(sympy.parse_expr(formula)))
                              for query, formula in self.query_formulas_map.items()])

        print(f"""SDDs:\n{self.circuits if self.circuits is not None else 'Discarded, using the polynomials'}""")
        print(f'SDD variables map:\n{self.indexed_vars}')
        print(f'SDD sizes before minimization:')
        for q, sdd in self.circuits.items():
            print(f'{q} : {sdd.size()}')
            sdd.ref()
        self.sdd_manager.minimize()
        print(f'SDD sizes after minimization:')
        for q, sdd in self.circuits.items():
            print(f'{q} : {sdd.size()}')
        self.__generate_polynomials()

        if self.clear_fields:
            self.__clear()

        # print(f'Polynomial functions:')
        # for q, p in self.polynomials.items():
        #    print(f'{q}\n{p.to_str()}')
