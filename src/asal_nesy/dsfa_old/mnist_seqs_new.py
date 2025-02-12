import os
import re
import random
import operator
import torchvision
import torch
import more_itertools
from torchvision.datasets import MNIST
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader

random.seed(10)

train_images = MNIST(
    os.path.join(os.path.expanduser("~/.cache"), "mnist_raw"),
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

test_images = MNIST(
    os.path.join(os.path.expanduser("~/.cache"), "mnist_raw"),
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


def generate_string(regex: str, sequence_length: int, match: bool = True) -> list[int]:
    while True:
        generated_sequence = [random.randint(0, 9) for _ in range(sequence_length)]
        sequences_matches = re.match(regex, "".join(map(str, generated_sequence)))
        if sequences_matches if match else (not sequences_matches):
            return generated_sequence


def run_generation_loop(
        sequences: list[list[int]], label2id: dict[int, list[int]], train: bool = True
):
    final_sequences = []
    for sequence in sequences:
        images = []
        for number in sequence:
            selected_index = random.choice(label2id[number])
            label2id[number].remove(selected_index)
            images.append((train_images if train else test_images)[selected_index])

        final_sequences.append(images)

    return final_sequences, label2id


def create_split(
        positives: list[list[int]],
        negatives: list[list[int]],
        label2id: dict[int, list[int]],
        train: bool = True,
):
    label_map = label2id.copy()
    positive_seqs, label_map = run_generation_loop(positives, label_map, train=train)
    negative_seqs, label_map = run_generation_loop(negatives, label_map, train=train)

    return positive_seqs, negative_seqs


def create_dataset(
        sequence_length: int,
        num_datapoints_train: int,
        num_datapoints_test: int,
        pattern: str,
):
    train_label2id = defaultdict(list)
    for i, (_, label) in enumerate(train_images):
        train_label2id[label].append(i)

    test_label2id = defaultdict(list)
    for i, (_, label) in enumerate(test_images):
        test_label2id[label].append(i)

    train_positive_sequences = [
        generate_string(pattern, sequence_length) for _ in range(num_datapoints_train)
    ]

    train_negative_sequences = [
        generate_string(pattern, sequence_length, False)
        for _ in range(num_datapoints_train)
    ]

    test_positive_sequences = [
        generate_string(pattern, sequence_length) for _ in range(num_datapoints_test)
    ]

    test_negative_sequences = [
        generate_string(pattern, sequence_length, False)
        for _ in range(num_datapoints_test)
    ]

    print({key: len(value) for key, value in train_label2id.items()})
    print(
        Counter(
            more_itertools.flatten(train_positive_sequences + train_negative_sequences)
        )
    )
    print({key: len(value) for key, value in test_label2id.items()})
    print(
        Counter(
            more_itertools.flatten(test_positive_sequences + test_negative_sequences)
        )
    )

    train_positives, train_negatives = create_split(
        train_positive_sequences, train_negative_sequences, train_label2id
    )
    test_positives, test_negatives = create_split(
        test_positive_sequences, test_negative_sequences, test_label2id, train=False
    )

    return train_positives, train_negatives, test_positives, test_negatives


class SequencesDataset(Dataset):
    def __init__(self, pos_seqs, neg_seqs):
        self.pos_seqs = pos_seqs
        self.neg_seqs = neg_seqs
        self.all_seqs = self.merge_seqs()

    def merge_seqs(self):
        pos = [(list(zip(*p))[0], list(zip(*p))[1], 1) for p in self.pos_seqs]
        neg = [(list(zip(*p))[0], list(zip(*p))[1], 0) for p in self.neg_seqs]
        all_seqs = pos + neg
        return all_seqs

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx):
        seq = self.all_seqs[idx]
        tensor_seq, symb_seq, label = seq[0], seq[1], seq[2]
        tensor_seq = torch.stack(list(tensor_seq), dim=0)
        symb_seq = torch.tensor(symb_seq)
        label = torch.tensor(label)
        return tensor_seq, label, symb_seq


def get_data_loaders(batch_size=1):
    sequence_length, num_train, num_test = 10, 1000, 300
    train_positives, train_negatives, test_positives, test_negatives = (
        create_dataset(
            sequence_length, num_train, num_test, r"\d*(8)\d*(1|3|5)\d*(0|1|2|3)"
        )
    )
    train_dataset = SequencesDataset(train_positives, train_negatives)
    test_dataset = SequencesDataset(test_positives, test_negatives)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    sequence_length, num_train, num_test = 10, 1000, 300
    """
    train_positives, train_negatives, test_positives, test_negatives = (
        create_dataset(
            sequence_length, num_train, num_test, r"\d*(8)\d*(1|3|5)\d*(0|1|2|3)"
        )
    )
    """
    train_positives, train_negatives, test_positives, test_negatives = create_dataset(
            sequence_length, num_train, num_test, r"\d*(8)\d*(1|3|5)\d*(0|1|2|3)"
        )

