from typing import List, Dict, Union, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from src.asal_nesy.dsfa_old.mnist_seqs_new import generate_data


class MNISTImageSequence:
    """
    Represents a sequence of images and their associated symbolic labels.
    """

    def __init__(
            self,
            seq_id: str,
            image_ids: List[str],
            image_tensors: List[torch.Tensor],
            seq_label: int,
            image_labels: Optional[List[Union[str, int, Dict[str, Union[str, int]]]]] = None
    ):
        self.seq_id = seq_id
        self.image_ids = image_ids  # list of unique image identifiers
        self.image_tensors = image_tensors  # list of torch.Tensor objects (C, H, W)
        self.seq_label = seq_label
        self.image_labels = image_labels or [None] * len(image_tensors)  # can be updated later

        # Initially, no images are labeled
        self.labeled_mask = [False] * len(image_tensors)

        # CNN predictions will be updated during training
        self.predicted_labels = [None] * len(image_tensors)
        self.predicted_logits = [None] * len(image_tensors)  # Optionally store logits
        self.label_trace = image_labels
        self.predicted_trace = None

    def get_image(self, idx: int) -> torch.Tensor:
        return self.image_tensors[idx]

    def get_image_label(self, idx: int) -> Optional[Union[str, Dict[str, str]]]:
        return self.image_labels[idx]

    def set_image_label(self, idx: int, label: Union[int, str, Dict[str, str]] = None):
        if label is not None:  # if None, simply mark the instance as labeled and use the already provided label.
            self.image_labels[idx] = label
        self.labeled_mask[idx] = True

    def set_image_prediction(self, idx: int, label: Union[str, Dict[str, str]], logits: Optional[torch.Tensor] = None):
        self.predicted_labels[idx] = label
        self.predicted_logits[idx] = logits

    def get_sequence_tensor(self) -> torch.Tensor:
        return torch.stack(self.image_tensors)  # Shape: (seq_len, C, H, W)

    def get_labeled_indices(self) -> List[int]:
        return [i for i, is_labeled in enumerate(self.labeled_mask) if is_labeled]

    def get_sequence_label(self) -> torch.Tensor:
        return torch.tensor(self.seq_label)

    def __len__(self):
        return len(self.image_tensors)


class SequenceBatch:
    def __init__(self, sequences: List[MNISTImageSequence], label_dict: Optional[Dict[str, Dict[str, int]]] = None):
        self.sequences = sequences
        self.label_dict = label_dict

    def __getitem__(self, idx):
        return self.sequences[idx]

    def __len__(self):
        return len(self.sequences)


class ImageSequenceDataset(Dataset):
    """
    Dataset for ImageSequence objects.
    """

    def __init__(self, sequences: List[MNISTImageSequence]):
        self.sequences = sequences

    def __getitem__(self, idx):
        return self.sequences[idx]

    def __len__(self):
        return len(self.sequences)


class IndividualImageDataset(Dataset):
    """Container class for individual image/label pairs"""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# def sequence_batch_collate_fn(batch):
#     return SequenceBatch(batch)

def get_collate_fn(label_dict=None):
    def collate_fn(batch):
        return SequenceBatch(batch, label_dict=label_dict)

    return collate_fn


def get_data(app_name: str) -> (ImageSequenceDataset, ImageSequenceDataset):
    """Return a training and testing set and  ImageSequenceDataset instances"""
    train_image_seqs_objects, test_image_seqs_objects = [], []
    train, test = None, None
    if app_name == "mnist":
        sequence_length, num_train, num_test = 10, 1000, 300
        train_positives, train_negatives, test_positives, test_negatives = (
            generate_data(sequence_length, num_train, num_test)
        )
        seq_counter = 1
        for set, label, train_test in [(train_positives, 1, 'train'), (train_negatives, 0, 'train'),
                                       (test_positives, 1, 'test'), (test_negatives, 0, 'test')]:
            for t in set:
                image_tensors = [x[0] for x in t]
                image_labels = [x[1] for x in t]
                # seq_id = f'seq_{seq_counter}'
                seq_id = seq_counter
                img_ids = [f'img_{seq_counter}_{i}' for i in range(1, len(image_labels) + 1)]
                if train_test == 'train':
                    train_image_seqs_objects.append(MNISTImageSequence(seq_id, img_ids, image_tensors, label, image_labels))
                else:
                    test_image_seqs_objects.append(MNISTImageSequence(seq_id, img_ids, image_tensors, label, image_labels))
                seq_counter += 1
        train, test = ImageSequenceDataset(train_image_seqs_objects), ImageSequenceDataset(test_image_seqs_objects)

    return train, test


def get_data_loader(dataset: Dataset, batch_size: int, train=True):
    """
    Example of using the get_collate_fn function with a label dict:

    label_vocab = {
    "vehicle_1_action": {"moveaway": 0, "approach": 1},
    "vehicle_2_location": {"incominglane": 0, "same_lane": 1}
    }

    dataloader = DataLoader( dataset, batch_size=4, shuffle=True, collate_fn=make_collate_fn(label_vocab))
    """
    shuffle = True if train else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=get_collate_fn())
