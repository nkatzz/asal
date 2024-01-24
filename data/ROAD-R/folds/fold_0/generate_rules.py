from itertools import product, combinations

# actions = ["hazlit", "stop", "brake", "incatlft", "movaway", "movtow", "xingfmrht", "xing", "turrht", "incatrht", "pushobj", "wait2x", "xingfmlft", "mov", "turlft", "rev"]
# locations = ["vehlane", "incomlane", "incomcyclane", "lftpav", "jun", "rhtpav", "busstop", "outgolane", "outgocyclane", "xing", "pav", "parking"]

actions = ["stop", "movaway", "movtow", "turrht", "turlft"]
locations = ["vehlane", "incomlane", "incomcyclane", "jun", "busstop", "outgolane", "outgocyclane"]

def generate_rules(actions, locations, max_length):
    rules = []
    rule_id = 1
    elements = ['a1', 'l1', 'a2', 'l2']

    for length in range(1, max_length + 1):
        # Generate combinations of elements for the given length
        for elem_combination in combinations(elements, length):
            # Create combinations of actions and locations based on elements
            for values in product(*(actions if 'a' in e else locations for e in elem_combination)):
                rule_body = ', '.join(f"{e}({v})" for e, v in zip(elem_combination, values))
                rules.append(f"r({rule_id}) :- {rule_body}.")
                rule_id += 1

    return rules

max_body_length = 4
rules = generate_rules(actions, locations, max_body_length)

for rule in rules:
    print(rule)

