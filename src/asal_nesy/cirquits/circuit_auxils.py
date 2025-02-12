from src.asal_nesy.deepfa.automaton import DeepFA


def model_count(sdd_builder):
    """To get the model count from a given SDD we just need to set all variables to 1.0 and
    propagate through the circuit."""

    sdd_vars = sdd_builder.sdd_variables
    sdds = sdd_builder.circuits
    model_counts = {}
    for k, v in sdds.items():
        mc = sdds[k].wmc(log_mode=False)
        for sdd_var in sdd_vars:
            mc.set_literal_weight(sdd_builder.get_sdd_var(sdd_var), 1.0)
        model_count = mc.propagate()
        model_counts[k] = model_count
    return model_counts


def nnf_map_to_sfa(nnf_circuits):
    transitions = {}
    for guard, formula in nnf_circuits.items():
        s = guard.split(',')
        from_state = int(s[0].split('(')[1])
        to_state = int(s[1].split(')')[0])
        if from_state in transitions:
            transitions[from_state][to_state] = formula
        else:
            transitions[from_state] = {to_state: formula}
    states = sorted(set(transitions.keys()).union(*[inner.keys() for inner in transitions.values()]))
    start, end = states[0], states[len(states) - 1]
    sfa = DeepFA(transitions, start, {end})
    return sfa


def model_count_nnf(nnf_circuits):
    model_counts = {}
    for k, v in nnf_circuits.items():
        mc = v.model_count()
        model_counts[k] = mc
    return model_counts


def weighted_model_count(sdd_builder, vars_probs_map):
    sdds = sdd_builder.circuits
    probs = {}
    for query, sdd in sdds.items():
        wmc = sdd.wmc(log_mode=False)
        for var, prob in vars_probs_map.items():
            wmc.set_literal_weight(sdd_builder.get_sdd_var(var), prob)
        probability = wmc.propagate()
        probs[query] = probability
    return probs
