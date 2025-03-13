import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument("--tlim", metavar="<T>", type=int, default=60, help="Time limit for Clingo in secs.")

    parser.add_argument("-qs", "--states", type=int, default=3,
                        help="Max number of states in an automaton.")

    parser.add_argument("-c", "--class", type=int, default=1, help="Target class to predict (one-vs-rest).")

    parser.add_argument("-abs", "--asal-batch", type=int, default=100, help="Batch size for ASAL.")

    parser.add_argument("-i", "--asal-incr", action="store_true", help="Learn incrementally with ASAL.")

    parser.add_argument("-atr", "--asal-train", type=str, help="Path to ASAL training data.")

    parser.add_argument("-ate", "--asal-test", type=str, help="Path to ASAL testing data.")

    parser.add_argument("-mi", "--mcts-iters", type=int, default=5,
                        help="Number of MCTS iterations for incremental ASAL.")

    parser.add_argument("-er", "--explr-rate", type=float, default=0.005,
                        help="Exploration rate for MCTS in incremental ASAL.")

    parser.add_argument("-mc", "--mcts-children", type=int, default=100,
                        help="Number of children nodes to consider in MCTS (selected randomly).")

    parser.add_argument("-cf", "--coverage-first", action="store_true",
                        help="Higher priority to predictive performance optimization constraints over model size ones.")

    parser.add_argument("-ma", "--min-atts", action="store_true",
                        help="Minimize the number of attributes that appear in an model.")

    return parser


if __name__ == "__main__":
    parser = parse_args()
    parser.parse_args()





