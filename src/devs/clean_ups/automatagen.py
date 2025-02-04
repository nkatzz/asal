from src.asal.learner import Learner
from src.asal.template import Template
from src.asal.auxils import *

"""Script for iteratively generating automata from different regions of the training set."""


def count_generated():
    with open('automata') as training_data:
        lines = training_data.readlines()
        count = 0
        for l in lines:
            if l.startswith('transition'):
                count += 1
        print(count)


if __name__ == "__main__":
    t_lim = 20  # 'inf'
    max_states = 3
    target_class = 1
    mini_batch_size = 10
    automata_count = 0
    time_limit = float('inf') if t_lim == 'inf' else t_lim
    logging.basicConfig(level=logging.DEBUG)

    train_path = '/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/Maritime_TRAIN_SAX_8_ASP.csv'
    test_path = '/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/Maritime_TEST_SAX_8_ASP.csv'

    # kept_automata = []
    kept_automata_count = 0

    for i in range(0, 5000):
        # Partition the data by shuffling positive and negative instances to get a new dict each time
        train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=True)
        # batch_id, batch_data = batch[0], batch[1]
        template = Template(max_states, target_class)
        for key in train_data.keys():
            learner = Learner(template, train_data[key], time_limit, mode='reasoning')
            learner.induce_models()
            # kept_automata = kept_automata + learner.generated_automata
            kept_automata_count = kept_automata_count + len(learner.generated_automata)
            print('Kept automata number:', kept_automata_count)
