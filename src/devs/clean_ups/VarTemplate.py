import itertools
import string


def generate(max_vars, type_pred_symbols, rest_pred_symbols):
    type_vars_map = {}
    grounding_rules, abduction_rules, helper_bk_rules = [], [], []
    counter = 1
    for p in type_pred_symbols:
        r_1 = f'abd(I,J,p({p},X)) :- abduced(I,J,{p}(X)).'
        r_2 = f'grnd(I,J,p({p},X),E) :- grounding(I,J,{p}(X),E).'
        helper_bk_rules.append(r_1)
        helper_bk_rules.append(r_2)

        # Allow for max_vars-many number of variable constants for p
        p_vars = list(map(lambda x: f'{p}_{x}', range(0, max_vars)))
        for v in p_vars:
            type_vars_map[v] = p
            r_3 = f'val({v},X) :- {p}(X).'
            helper_bk_rules.append(r_3)
            clause_1 = f'0 {{grounding(I,{counter},{p}(X),E) : abduced(I,{counter},{p}({v})), val({v},X), holds({p}(X),E)}} 1.'
            clause_2 = f'0 {{abduced(I,{counter},{p}({v})) : conj(I)}} 1.'
            grounding_rules.append(clause_1)
            abduction_rules.append(clause_2)
            counter += 1
    all_vars = type_vars_map.keys()

    for p in rest_pred_symbols:
        # name, arity = p.split('/')[0], p.split('/')[1]  # I don't think we'll have to generate triplets or beyond...
        r = f'abd(I,J,p({p},X,Y)) :- abduced(I,J,{p}(X,Y)).'
        r_4 = f'grnd(I,J,p({p},X,Y),E) :- grounding(I,J,{p}(X,Y),E).'
        helper_bk_rules.append(r_4)
        helper_bk_rules.append(r)
        for i, j in itertools.product(all_vars, all_vars):
            if i != j:
                clause_1 = f'0 {{grounding(I,{counter},{p}(X,Y),E) : abduced(I,{counter},{p}({i},{j})), val({i},X), val({j},Y), holds({p}(X,Y),E), X != Y }} 1.'
                clause_2 = f'0 {{abduced(I,{counter},{p}({i},{j})) : conj(I)}} 1.'
                grounding_rules.append(clause_1)
                abduction_rules.append(clause_2)
                counter += 1
    return grounding_rules, abduction_rules, helper_bk_rules


if __name__ == "__main__":
    gr_rules, abd_rules, helper_rules = generate(3, ['r_0', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7'], ['r_8', 'r_10', 'r_11', 'r_12'])
    # gr_rules, abd_rules = generate(3, ['carbon', 'nitrogen'], ['single', 'double'])
    print('\n'.join(gr_rules))
    print('\n'.join(abd_rules))
    print('\n'.join(helper_rules))
    f = open("/home/nkatz/Dropbox/ASP-experimentation/conditionals/aaa/aaa-large/template.lp", "w")
    f.write('\n'.join(gr_rules))
    f.write('\n')
    f.write('\n'.join(abd_rules))
    f.write('\n')
    f.write('\n'.join(helper_rules))
    f.close()
