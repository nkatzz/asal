import itertools
import os
from src.globals import is_accepting_absorbing


class Template:
    def __init__(self, max_states, target_class):
        self.mutex_map = {}
        self.template = ''
        self.__guard_definitions = []
        self.max_states = max_states
        self.target_class = target_class
        self.__transition_choices = []
        self.__rules_choices = []
        self.__start_state = 'start(1).'
        self.__accepting_state = f'accepting({self.max_states}).'
        self.__self_loop_ids = []
        self.__generate_template()

    def __update_transitions(self, guard_name, i, j):
        transition = "{" + f'transition({i},{guard_name},{j})' + "}."

        if i != j:
            self.__rules_choices.append(f'f({i},{j})')

        # if True:
        #     self.__rules_choices.append(f'f({i},{j})')

        self.__transition_choices.append(transition)

    @staticmethod
    def __generate_self_loop_rule(guard_name, f_nots):
        rule_head = f'holds({guard_name},S,T)'
        rule_body = ', '.join(map(lambda x: f'not holds({x},S,T)', f_nots))
        rule = f'{rule_head} :- sequence(S), time(T), {rule_body}.'
        return rule

    @staticmethod
    def __generate_rule(guard_name, f_nots, i, j):
        if f_nots:
            rule = 'holds({0},S,T) :- holds(body(f({1},{2}),J),S,T), {3}.'. \
                format(guard_name, i, j, ', '.join(map(lambda x: 'not holds({0},S,T)'.format(x), f_nots)))
        else:
            rule = f'holds({guard_name},S,T) :- holds(body(f({i},{j}),J),S,T).'
        return rule

    def __generate_template(self):
        for i, j in itertools.product(range(1, self.max_states), range(1, self.max_states + 1)):
            f = f'f({i},{j})'
            self.__update_transitions(f, i, j)
            if i == j:
                self.__self_loop_ids.append(f'f({i},{j})')
                f_nots = [f'f({i},{y})' for y in range(1, self.max_states + 1) if y != i]
                rule = self.__generate_self_loop_rule(f, f_nots)
            else:
                f_nots = [f'f({i},{y})' for y in range(j + 1, self.max_states + 1) if y != i]
                rule = self.__generate_rule(f, f_nots, i, j)
            self.mutex_map[f] = f_nots
            self.__guard_definitions.append(rule)
        self.mutex_map[f'f({self.max_states},{self.max_states})'] = []

        m = self.max_states
        if is_accepting_absorbing:
            self.__guard_definitions.append(f'holds(f({m},{m}),S,T) :- sequence(S), time(T).')
            self.__update_transitions(f'f({m},{m})', m, m)
            self.__self_loop_ids.append(f'f({m},{m})')
        else:
            # allow to learn a constrained self loop, e.g. f(4,4) :- Digit <= 3 in MNIST.
            # This is for avoid learning over-general self loops on the accepting state, which
            # will increase acceptance probability in a NeSy setting.
            self.__guard_definitions.append(f'holds(f({m},{m}),S,T) :- holds(body(f({m},{m}),J),S,T).')
            self.__guard_definitions.append('{' + f'rule(f({m},{m}))' + '}.')
            self.__guard_definitions.append('{' + f'transition({m},f({m},{m}),{m})' + '}.')

        self.__rules_choices = '{rule(' + ';'.join(self.__rules_choices) + ')}.'
        self.__assemble_template()

        if 'src' in os.getcwd():
            if 'asal_nesy' in os.getcwd():
                prefix = os.getcwd().split('asal_nesy')[0][:-1]
                asp_template_file = os.path.normpath(
                    prefix + os.sep + 'asal' + os.sep + 'asp' + os.sep + 'template.lp')
            else:
                asp_template_file = os.path.normpath(os.getcwd() + os.sep + 'asal' + os.sep + 'asp' + os.sep + 'template.lp')
        else:
            asp_template_file = os.path.normpath(os.getcwd() + os.sep + 'src' + os.sep + 'asal' + os.sep + 'asp' + os.sep + 'template.lp')

        f = open(asp_template_file, "w")
        # f = open("asp/template.lp", "w")
        f.write('% This is generated by template.py when starting the app.\n\n')
        f.write(self.template)
        f.close()

    def __assemble_template(self):
        transition_choices = ' '.join(self.__transition_choices)
        transition_guards = '\n'.join(self.__guard_definitions)
        start_acc = f"{self.__start_state} {self.__accepting_state}"
        c = ', '.join(map(lambda x: f'I!={x}', self.__self_loop_ids))
        constraint = f':- holds(body(I,J),S,T), rule(I), {c}, #count{{F: body(I,J,F)}} = 0.'
        # constraint_1 = f':- holds(I,S,T), rule(I), {c}, #count{{F,J: body(I,J,F)}} = 0.'
        constraint_1 = ''
        body_def = "holds(body(I,J),S,T) :- rule(I), conjunction(J), sequence(S), time(T), holds(F,S,T) : body(I,J,F)."
        rules_def = "rule(R) :- transition(I,R,F)."

        self.template = '\n'.join([transition_choices,
                                   transition_guards,
                                   self.__rules_choices,
                                   constraint,
                                   constraint_1,
                                   start_acc, body_def
                                   ])

        """
        self.template = '\n'.join(["transition(I,f(I,J),J) :- rule(f(I,J)).",
                                   transition_guards,
                                   self.__rules_choices,
                                   constraint,
                                   constraint_1,
                                   start_acc, body_def
                                   ])
        """


if __name__ == "__main__":
    t = Template(5, 1)
    print(t.mutex_map)
    print(t.template)
