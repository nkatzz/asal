import argparse
from src.asal.asp import *
from src.asal.template import Template


def parse_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--tlim", metavar="<n>",
                        type=int, default=0, help="time limit for Clingo in secs (default: 0, no limit).")

    parser.add_argument("--states", metavar="<n>", type=int, default=3,
                        help="max number of states in a learnt automaton.")

    parser.add_argument("--tclass", metavar="<n>",
                        type=int, default=1, help="target class to predict (one-vs-rest).")

    parser.add_argument("--train", metavar="<path>", required=True,
                        type=str, help="path to the training data.")

    parser.add_argument("--test", metavar="<path>",
                        type=str, help="path to the testing data.")

    parser.add_argument("--domain", metavar="<path>", required=True,
                        type=str, help="path to the domain specification file.")

    parser.add_argument("--incremental",
                        action="store_true", help="learn incrementally with MCTS.")

    parser.add_argument("--batch_size", metavar="<n>",
                        type=int, default=100, help="mini batch size for incremental learning.")

    parser.add_argument("--max_alts", metavar="<n>",
                        type=int, default=2, help="max number of disjunctive alternatives per transition guard.")

    parser.add_argument("--mcts_iters", metavar="<n>", type=int, default=1,
                        help="number of MCTS iterations for incremental learning.")

    parser.add_argument("--exp_rate", metavar="<float>", type=float, default=0.005,
                        help="exploration rate for MCTS in incremental learning.")

    parser.add_argument("--mcts_children", metavar="<n>", type=int, default=10,
                        help="number of children nodes to consider in MCTS.")

    parser.add_argument("--coverage_first", action="store_true",
                        help="higher priority to predictive performance optimization constraints over model size ones.")

    parser.add_argument("--min_attrs", action="store_true",
                        help="minimize the number of attributes that appear in a model.")

    parser.add_argument("--warns_off", action="store_true",
                        help="suppress warnings from Clingo.")

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
    parser = parse_args()
    args = parser.parse_args()
    print(args)
    template = Template(args.states, args.tclass)
    p = get_induction_program(args, template)
