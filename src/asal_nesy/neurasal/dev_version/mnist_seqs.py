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


"""
def generate_string(regex: str, sequence_length: int, match: bool = True) -> list[int]:
    if match:
        # Keep generating random sequences until a match is found
        while True:
            generated_sequence = [random.randint(0, 9) for _ in range(sequence_length)]
            if re.match(regex, "".join(map(str, generated_sequence))):
                return generated_sequence
    else:
        # Generate a matching sequence first
        while True:
            generated_sequence = [random.randint(0, 9) for _ in range(sequence_length)]
            sequence_str = "".join(map(str, generated_sequence))
            if re.match(regex, sequence_str):
                break

        # Now modify it to ensure it doesn't match
        for i in range(sequence_length):
            original_digit = generated_sequence[i]

            # Pick a digit that is different from the original one
            new_digit = random.choice([d for d in range(10) if d != original_digit])
            generated_sequence[i] = new_digit

            # Check if the new sequence does not match the regex
            sequence_str = "".join(map(str, generated_sequence))
            if not re.match(regex, sequence_str):
                return generated_sequence

            # Revert if it still matches
            generated_sequence[i] = original_digit

        # As a fallback, invert all digits to guarantee non-matching
        generated_sequence = [(d + 1) % 10 for d in generated_sequence]
        return generated_sequence
"""

"""
def run_generation_loop(
        sequences: list[list[int]], label2id: dict[int, list[int]], train: bool = True
):
    final_sequences = []
    for sequence in sequences:
        images = []
        for number in sequence:
            # Check if there are any remaining images for the given label
            if not label2id[number]:
                raise ValueError(f"No remaining indices for label {number} in the dataset. "
                                 f"Ensure sufficient data for all digits.")
            selected_index = random.choice(label2id[number])

            label2id[number].remove(selected_index)  # Remove the selected index to avoid reuse

            images.append((train_images if train else test_images)[selected_index])

        final_sequences.append(images)

    return final_sequences, label2id
"""


def run_generation_loop(
        sequences: list[list[int]], label2id: dict[int, list[int]], train: bool = True
):
    final_sequences = []

    # Reference the correct dataset based on `train` flag
    dataset = train_images if train else test_images

    for sequence in sequences:
        images = []
        for number in sequence:
            # Check if label2id[number] is empty
            if not label2id[number]:
                print(f"No remaining indices for label {number} in the dataset. Re-creating indices")
                label2id[number] = [i for i, (_, label) in enumerate(dataset) if label == number]

            # Pick a random index
            selected_index = random.choice(label2id[number])
            image_tensor, label = dataset[selected_index]
            images.append((image_tensor, label))  # Store tensor and label

        final_sequences.append(images)

    return final_sequences, label2id


def create_split(
        positives: list[list[int]],
        negatives: list[list[int]],
        label2id: dict[int, list[int]],
        train: bool = True,
        starting_seq_id: int = 0
):
    label_map = label2id.copy()
    positive_seqs, label_map = run_generation_loop(positives, label_map, train=train)
    negative_seqs, label_map = run_generation_loop(negatives, label_map, train=train)

    seq_id = starting_seq_id
    pos_with_ids = []
    neg_with_ids = []

    for seq, symbol_seq in zip(positive_seqs, positives):
        image_data = [(img, symb, f"{seq_id}_{i}") for i, ((img, _), symb) in enumerate(zip(seq, symbol_seq))]
        pos_with_ids.append((image_data, seq_id, 1))  # label = 1
        seq_id += 1

    for seq, symbol_seq in zip(negative_seqs, negatives):
        image_data = [(img, symb, f"{seq_id}_{i}") for i, ((img, _), symb) in enumerate(zip(seq, symbol_seq))]
        neg_with_ids.append((image_data, seq_id, 0))  # label = 0
        seq_id += 1

    return pos_with_ids, neg_with_ids, seq_id

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

    print('Generated train positives')

    train_negative_sequences = [
        generate_string(pattern, sequence_length, False)
        for _ in range(num_datapoints_train)
    ]

    print('Generated train negatives')

    test_positive_sequences = [
        generate_string(pattern, sequence_length) for _ in range(num_datapoints_test)
    ]

    print('Generated test positives')

    test_negative_sequences = [
        generate_string(pattern, sequence_length, False)
        for _ in range(num_datapoints_test)
    ]

    print('Generated test negatives')

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

    train_positives, train_negatives, next_seq_id = create_split(
        train_positive_sequences, train_negative_sequences, train_label2id, starting_seq_id=0
    )
    test_positives, test_negatives, _ = create_split(
        test_positive_sequences, test_negative_sequences, test_label2id, train=False, starting_seq_id=next_seq_id
    )

    return train_positives, train_negatives, test_positives, test_negatives


class SequencesDataset(Dataset):
    def __init__(self, pos_seqs, neg_seqs):
        self.all_seqs = pos_seqs + neg_seqs

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx):
        image_data, seq_id, label = self.all_seqs[idx]
        tensor_seq = torch.stack([img for img, _, _ in image_data], dim=0)
        symb_seq = torch.tensor([symb for _, symb, _ in image_data])
        image_ids = [img_id for _, _, img_id in image_data]
        label = torch.tensor(label)
        return tensor_seq, label, symb_seq, seq_id, image_ids


def get_data_loaders_OOD(batch_size=1):
    sequence_length, num_train, num_test = 10, 1000, 300

    train_positives, train_negatives, _, _ = (
        create_dataset(
            sequence_length, num_train, num_test, r"\d*(8)\d*(1|3|5)\d*(0|1|2|3)"
        )
    )

    sequence_length, num_train, num_test = 50, 1000, 300

    _, _, test_positives, test_negatives = (
        create_dataset(
            sequence_length, num_train, num_test, r"\d*(8)\d*(1|3|5)\d*(0|1|2|3)"
        )
    )

    train_dataset = SequencesDataset(train_positives, train_negatives)
    test_dataset = SequencesDataset(test_positives, test_negatives)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_data_loaders(batch_size=1):
    # Generates 1000+1000 pos/neg (training) and 300+300 pos/neg (testing)
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
    seq_length, train_num, test_num = 10, 1000, 300

    train_pos, train_neg, test_pos, test_neg = create_dataset(
        seq_length, train_num, test_num, r"\d*(8)\d*(1|3|5)\d*(0|1|2|3)"
    )
