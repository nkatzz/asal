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
        self.image_labels = seq_labels
        self.images = seq_images
        seq_length, dimensionality, _, _, _ = self.images.shape
        self.seq_length = seq_length
        self.dimensionality = dimensionality
        self.labeled_mask = torch.zeros(seq_length, dimensionality, dtype=torch.bool)

        self.image_ids = [
            [f"{seq_id}_{t}_{d}" for d in range(dimensionality)]
            for t in range(seq_length)
        ]

        # This is a dictionary with the class prediction order per attribute. For instance, for 2-variate MNIST this is:
        # {
        #     'd1': ['d1_0', 'd1_1', 'd1_2', 'd1_3', 'd1_4', 'd1_5', 'd1_6', 'd1_7', 'd1_8', 'd1_9'],
        #     'd2': ['d2_0', 'd2_1', 'd2_2', 'd2_3', 'd2_4', 'd2_5', 'd2_6', 'd2_7', 'd2_8', 'd2_9']
        # }
        #
        # From this dictionary we can get the actual class per attribute that corresponds to argmax predictions, in
        # order to form symbolic sequences for ASAL.
        self.class_prediction_dict = {}

        self.acceptance_probability = 0.0
        self.elr_score = None
        self.sequence_probability = None
        self.asp_weight = None
        self.predicted_symbolic_seq = None
        self.predicted_softmaxed_seq = None
        self.bce_loss = None

        self.edit_points = None
        self.edit_cost = None
        self.edit_score = None

    def set_image_label(self, t: int, d: int, label=None):
        """
        Mark image at point t in the sequence and dimension d as labeled.
        Optionally, assign new label if one is provided. Example usage:

        # Mark image at time 2, dimension 1 as labeled
        seq.set_image_label(2, 1, {'action': 2, 'location': 1, coords: (10, 15)})
        """
        if label is not None:
            self.image_labels[t][d] = label
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
            return self.image_labels[time][dim]
        else:
            return None

    def get_image_label(self, t: int, d: int, attributes: list):
        """
        if self.is_labeled(t, d):
            return self.labels[t][d]
        else:
            return None
        """
        dict = self.image_labels[t][d]
        labels = [dict[key] for key in attributes if key in dict]
        return labels

    def get_image_labels(self, attributes: list):
        res = []
        for t in range(self.seq_length):
            for d in range(self.dimensionality):
                res.extend(self.get_image_label(t, d, attributes))
        return res

    def get_image_indices(self):
        return [(t, d) for d in range(self.dimensionality) for t in range(self.seq_length)]

    def get_labeled_indices(self):
        return [(t, d) for t, d in self.get_image_indices() if self.is_labeled(t, d)]

    def mark_seq_as_fully_labelled(self):
        for t in range(self.seq_length):
            for d in range(self.dimensionality):
                self.set_image_label(t, d)

    def get_labelled_seq_asp(self, with_custom_weights=False):
        grouped = [list(group) for group in zip(*self.image_labels)]
        result = [
            [
                f"seq({self.seq_id},obs({key},{value}),{i})."
                for i, d in enumerate(group)
                for key, value in d.items()
            ]
            for group in grouped
        ]
        for x in result:
            x.append(f'class({self.seq_id},{self.seq_label}).')
            if with_custom_weights:
                x.append(f'weight({self.seq_id},{self.asp_weight}).')
        return '\n'.join(" ".join(seq) for seq in result)

    def get_predicted_seq_asp(self, with_custom_weights=False):
        labels_grouped = [list(group) for group in zip(*self.image_labels)]
        attributes = sorted({k for g in labels_grouped for d in g for k in d})
        predicted_symbols_grouped = [list(group) for group in zip(*self.predicted_symbolic_seq)]
        seqs = [
            [f'seq({self.seq_id},obs({attributes[j]},{d}),{i}).' for i, d in enumerate(g)] + [f'class({self.seq_id},{self.seq_label}).']
            for j, g in enumerate(predicted_symbols_grouped)
        ]

        for s in seqs:
            if with_custom_weights:
                s.append(f'weight({self.seq_id},{self.asp_weight}).')

        return '\n'.join(" ".join(seq) for seq in seqs)

    def sample_symbolic_sequence(self, model):
        model.eval()
        label = self.seq_label
        vars = [k for v in self.image_labels[0] for k, _ in v.items()]
        with torch.no_grad():
            # (T, D, C, H, W) -> (T * D, C, H, W)
            T, D, C, H, W = self.images.shape
            images = self.images.view(T * D, C, H, W).to(next(model.parameters()).device)

            # Forward pass: (T * D, num_classes)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)

            # Sample one class per image
            sampled_symbols = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (T * D,)

            # Reshape back to (T, D) and convert to nested list
            symbolic_seq = sampled_symbols.view(T, D).cpu().tolist()
            seqs = [
                [f'seq({self.seq_id},obs({vars[j]},{d}),{i}).' for i, d in enumerate(g)] + [f'class({self.seq_id},{label}).']
                for j, g in enumerate(zip(*symbolic_seq))
            ]
            return '\n'.join(' '.join(x) for x in seqs)

    def generate_asp_prediction_facts(self, max_unlabelled_weight=100):
        """
        For this TensorSequence:
        - Convert softmax predictions to integer weights (log + normalized).
        - For each image:
           * emit prediction(...) facts for non-argmax symbols.
           * emit argmax_prediction(...) fact for the argmax.
        Returns:
            List[str] of ASP facts.
        """
        import numpy as np

        softmax_preds = self.predicted_softmaxed_seq  # shape: (SeqLen, Dim, NumClasses)
        seqlen, dim, num_classes = softmax_preds.shape
        facts = []

        # Flatten all probs to compute log-prob normalization
        all_probs = softmax_preds.flatten()
        log_probs = np.log(all_probs + 1e-12)
        min_lp, max_lp = np.min(log_probs), np.max(log_probs)
        norm_weights = (log_probs - min_lp) / (max_lp - min_lp + 1e-12)
        int_weights = (norm_weights * max_unlabelled_weight).astype(int) + 1  # ensure non-zero

        # Reshape back to (NumClasses, Dim, SeqLen)
        int_weights = int_weights.reshape(seqlen, dim,  num_classes)

        # For attribute names in dimension order:
        attr_list = list(self.class_prediction_dict.keys())  # e.g. ['d1', 'd2'] for 2 digits in 2-variate MNIST

        # Iterate over sequence points
        for t in range(seqlen):
            for d in range(dim):
                attr_name = attr_list[d]
                class_names = self.class_prediction_dict[attr_name]  # list of class names

                # probs = softmax_preds[:, d, t]  # shape: (NumClasses,)
                # weights = int_weights[:, d, t]  # shape: (NumClasses,)

                probs = softmax_preds[t][d]
                weights = int_weights[t][d]

                argmax_idx = np.argmax(probs)
                argmax_symbol = class_names[argmax_idx].split('_')[1]
                argmax_weight = weights[argmax_idx]

                # ASP fact for argmax
                facts.append(f"argmax_prediction({self.seq_id},obs({attr_name},{argmax_symbol}),{t}).")
                facts.append(f'argmax_weight({self.seq_id},obs({attr_name},{argmax_symbol},{t}),{argmax_weight}).')

                # ASP facts for all other symbols
                for c in range(num_classes):
                    if c == argmax_idx:
                        continue
                    pred_symbol = class_names[c].split('_')[1]
                    pred_weight = weights[c]
                    facts.append(f"prediction({self.seq_id},obs({attr_name},{pred_symbol}),{t}).")
                    facts.append(f'prediction_weight({self.seq_id},obs({attr_name},{pred_symbol},{t}),{pred_weight}).')

        return ' '.join(facts)


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


def seq_collate_fn(batch):
    return batch


def get_data_loader(dataset: Dataset, batch_size: int, train=True):
    shuffle = True if train else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=seq_collate_fn)


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
    # train_path = '/data/mnist_nesy/mnist_train.pt'
    # test_path = '/data/mnist_nesy/mnist_test.pt'
    # train, test = get_data(train_path, test_path)

    x = [[7, 2, 3], [5, 6, 7], [3, 4, 8]]
    grouped = zip(*x)
    id = 123
    vars = ['d1', 'd2', 'd3']



    seqs = [
        [f'seq({id},obs({vars[j]},{d}),{i}).' for i, d in enumerate(g)] + [f'class({id},1).']
        for j, g in enumerate(grouped)
    ]

    print(list(seqs))

    y = '\n'.join(' '.join(x) for x in seqs)
    print(y)
