import argparse
from src.asal.asp import *
from src.asal.template import Template
from src.asal.auxils import timer


@timer
def parse_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--train", metavar="<path>",
                        type=str, help="Path to the training data.")

    parser.add_argument("--domain", metavar="<path>", required=True,
                        type=str, help="Path to the domain specification file.")

    parser.add_argument("--test", metavar="<path>",
                        type=str, help="Path to the testing data.")

    parser.add_argument("--tlim",
                        metavar="<n>",
                        type=int,
                        default=0,
                        help="Time limit for Clingo in secs [default: 0, no limit].")

    parser.add_argument("--states", metavar="<n>", type=int, default=3,
                        help="Max number of states in a learnt automaton.")

    parser.add_argument("--tclass", metavar="<n>",
                        type=int, default=1, help="Target class to predict (one-vs-rest) [default: 1].")

    parser.add_argument("--unsat_weight", metavar="<n>",
                        type=int, default=1, help="""Cost/weight of not accepting a positive sequence, or not rejecting 
a negative one. The default weight is 1 and is applied 
uniformly to  all training sequences. Individual weights 
per example can be set via --unsat_weight 0, in which case the 
weights need to be provided in the training data file as weight(S,W)
where S is the sequence id and W is an integer.""")

    parser.add_argument("--incremental",
                        action="store_true", help="Learn incrementally with MCTS.")

    parser.add_argument("--batch_size", metavar="<n>",
                        type=int, default=100, help="Mini batch size for incremental learning.")

    parser.add_argument("--mcts_iters", metavar="<n>", type=int, default=1,
                        help="Number of MCTS iterations for incremental learning.")

    parser.add_argument("--exp_rate", metavar="<float>", type=float, default=0.005,
                        help="Exploration rate for MCTS in incremental learning.")

    parser.add_argument("--mcts_children", metavar="<n>", type=int, default=10,
                        help="Number of children nodes to consider in MCTS.")

    parser.add_argument("--max_alts", metavar="<n>",
                        type=int, default=2, help="Max number of disjunctive alternatives per transition guard.")

    parser.add_argument("--coverage_first",
                        action="store_true",
                        help="""Set a higher priority to constraints that minimize FPs & FNs 
over constraints that minimize model size.""")

    parser.add_argument("--min_attrs", action="store_true",
                        help="Minimize the number of different attributes/predicates that appear in a model.")

    parser.add_argument("--all_opt", action="store_true",
                        help="Find all optimal models during Clingo search.")

    parser.add_argument("--revise", action="store_true",
                        help="""Revise existing models instead of inducing from scratch on new data.""")

    parser.add_argument("--eval", metavar="<path>",
                        type=str, help="""Path to a file that contains an SFA specification (learnt/hand-crafted).
to evaluate on test data (passed via the --test option). The automaton needs to be
in reasoning-based format (see option --show)""")

    parser.add_argument("--show", metavar="<s|r>", type=str, default="s",
                        help="""Show learnt SFAs in simpler (s), easier to inspect format, 
or in a format that can be used for reasoning (r) with Clingo.""")

    parser.add_argument("--warns_off", action="store_true",
                        help="Suppress warnings from Clingo.")

    parser.add_argument(
        "--predicates",
        nargs="+",  # Accepts one or more values
        choices=["equals", "neg", "lt", "at_most", "at_least", "increase", "decrease"],  # Restrict valid options
        default=["equals"],
        help="""List of predicates to use for synthesizing transition guards. 
These are necessary for adding proper generate and test 
statements to the ASP induction program. 
        
Example usage: --predicates equals neg at_most        
Options: 
    - equals:   allows atoms of the form equals(A,V) in the transition guards, meaning that attribute A has value V. 
                The available attribute/value pairs can be extracted directly from the data, or defined via rules in the 
                background knowledge provided with the domain specification file. See available examples in the README.
                attribute A must be declared as 'categorical' in the domain specification.   
    - neg:      allows atoms of the form neg(A,V) in the transition guards, meaning that attribute A does not have value V.
    - lt:       allows atoms of the form lt(A1,A2) in the transition guards, meaning that the value of attribute A1 is 
                smaller than the value of A2. 
                Attributes A1, A2 must be declared as 'numerical' in the domain specification.
    - at_most:  allows atoms of the form at_most(A,V) in the transition guards, meaning that the value of attribute A is 
                smaller than V. Attribute A must be declared as 'numerical' in the domain specification.
    - at_least: allows atoms of the form at_least(A,V) in the transition guards, meaning that the value of attribute A is 
                larger than V. Attribute A must be declared as 'numerical' in the domain specification.
    - increase: allows atoms of the form increase(A) in the transition guards, meaning that the value of attribute A has
                increased since the last time step. Attribute A must be declared as 'numerical' in the domain specification.
    - decrease: allows atoms of the form decrease(A) in the transition guards, meaning that the value of attribute A has
                decreased since the last time step. Attribute A must be declared as 'numerical' in the domain specification.   
        """
    )

    return parser


if __name__ == "__main__":
    import time
    a = time.time()
    parser = parse_args()
    args = parser.parse_args()
    template = Template(args.states, args.tclass)
    p = get_induction_program(args, template)
    print(p)
