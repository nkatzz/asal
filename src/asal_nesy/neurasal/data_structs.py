import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple, List


class TensorSequence:
    def __init__(self, seq_id, seq_label, seq_images, seq_labels):
        """
        :param seq_images: Tensor (seq_length, dimensionality, C, H, W)
        :param seq_labels: list of list of attribute dicts, example:
        [
          [{action: 'moveaway', location: 'incomelane'}, {dict for image_2 in point 1}]
          [{dict for image_1 in point 1}, {dict for image_2 in point 2}]
          ...
        ]
        """
        self.seq_id = int(seq_id)
        self.seq_label = seq_label
        self.images = seq_images
        self.labels = seq_labels

        seq_length, dimensionality, _, _, _ = self.images.shape
        self.seq_length = seq_length
        self.dimensionality = dimensionality
        self.labeled_mask = torch.zeros(seq_length, dimensionality, dtype=torch.bool)

        self.image_ids = [
            [f"{seq_id}_{t}_{d}" for d in range(dimensionality)]
            for t in range(seq_length)
        ]

    def set_image_label(self, t: int, d: int, label=None):
        """
        Mark image at point t in the sequence and dimension d as labeled.
        Optionally, assign new label if one is provided. Example usage:

        # Mark image at time 2, dimension 1 as labeled
        seq.set_image_label(2, 1, {'action': 2, 'location': 1, coords: (10, 15)})
        """
        if label is not None:
            self.labels[t][d] = label
        self.labeled_mask[t, d] = True

    def is_labeled(self, t: int, d: int):
        return self.labeled_mask[t, d]

    def get_image_by_id(self, img_id):
        s = img_id.split('_')
        time, dim = int(s[1]), int(s[2])
        return self.images[time][dim]

    def get_image(self, t: int, d: int):
        return self.images[t][d]

    def get_img_label_by_id(self, img_id):
        s = img_id.split('_')
        time, dim = int(s[1]), int(s[2])
        if self.is_labeled(time, dim):
            return self.labels[time][dim]
        else:
            return None

    def get_image_label(self, t: int, d: int):
        """
        if self.is_labeled(t, d):
            return self.labels[t][d]
        else:
            return None
        """
        return self.labels[t][d]

    def get_image_indices(self):
        return [(t, d) for d in range(self.dimensionality) for t in range(self.seq_length)]


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[TensorSequence]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class IndividualImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_data_loader(dataset: Dataset, batch_size: int, train=True):
    shuffle = True if train else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_data(
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        data_generator: Optional[Callable[[], Tuple[List['TensorSequence'], List['TensorSequence']]]] = None
) -> Tuple[SequenceDataset, SequenceDataset]:
    """
    Returns a pair of TensorSequence lists for training and testing.

    :param train_path: Path to training .pt file (optional)
    :param test_path: Path to testing .pt file (optional)
    :param data_generator: Function to generate data on the fly (optional)
    :return: (train_sequences, test_sequences)
    """
    if train_path and test_path:
        # Load from disk
        train_sequences = torch.load(train_path, weights_only=False)
        test_sequences = torch.load(test_path, weights_only=False)

        def to_tensor_seq_obj(seqs):
            tensor_seqs = []
            for s in seqs:
                seq_metadata, seq = s
                seq_label = seq_metadata.seq_label
                seq_id = seq_metadata.seq_id
                symbolic_seq = seq_metadata.sequence
                seq_length, dim = len(seq), len(seq[0])
                tensors = [
                    [seq[t][d][0] for d in range(dim)]
                    for t in range(seq_length)
                ]

                tensors = torch.stack([torch.stack(inner) for inner in tensors])

                seq_labels = [
                    [{dim: symbolic_seq[dim][t]} for dim in symbolic_seq]
                    for t in range(len(next(iter(symbolic_seq.values()))))
                ]
                tensor_seq_obj = TensorSequence(seq_id, seq_label, tensors, seq_labels)
                tensor_seqs.append(tensor_seq_obj)
            return tensor_seqs

        train_sequences = to_tensor_seq_obj(train_sequences)
        test_sequences = to_tensor_seq_obj(test_sequences)

    elif data_generator:
        # Generate on the fly
        train_sequences, test_sequences = data_generator()
        print(f"Generated train and test sequences on the fly")
    else:
        raise ValueError("Either (train_path and test_path) OR data_generator must be provided.")

    return SequenceDataset(train_sequences), SequenceDataset(test_sequences)


if __name__ == "__main__":
    train_path = '/data/mnist_nesy/mnist_train.pt'
    test_path = '/data/mnist_nesy/mnist_test.pt'
    train, test = get_data(train_path, test_path)
