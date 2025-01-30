import torch

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


