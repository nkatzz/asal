import random
import torch
from collections import defaultdict
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import os
import torchvision

from src.asal_nesy.neurasal.mnist.mnist_image_sequences import MNISTImageSequence
from src.asal_nesy.neurasal.mnist.generate_diverse_seqs import generate_seqs, DigitSequence


def assign_images_to_sequences(digit_sequences, image_pool, augment_transform, seed=42):
    """
    Assign unique MNIST images to digits in DigitSequence sequences.
    If image pool is exhausted for a digit, augment an original image.

    :param digit_sequences: list of DigitSequence objects
    :param image_pool: dict {digit: list of (image, label, id) tuples}
    :param augment_transform: torchvision transform to apply when augmenting
    :param seed: random seed
    :return: list of (DigitSequence, list of [(img, label, id)]) tuples
    """
    random.seed(42)
    used_ids = set()
    assigned_sequences = []

    # Build source pool: for augmentation fallback
    source_pool = defaultdict(list)
    for digit, images in image_pool.items():
        for img, label, img_id in images:
            source_pool[digit].append((img, label, img_id))

    # Prepare iterators
    image_iters = {
        digit: iter(images) for digit, images in image_pool.items()
    }

    counter = defaultdict(int)  # For unique ID generation

    for seq in digit_sequences:
        seq_length = len(next(iter(seq.sequence.values())))
        dims = list(seq.sequence.keys())

        assigned = []

        for t in range(seq_length):
            images_t = []
            for d in dims:
                digit = int(seq.sequence[d][t])

                to_tensor = ToTensor()
                # Try to get unused image
                try:
                    while True:
                        img, label, img_id = next(image_iters[digit])
                        if img_id not in used_ids:
                            used_ids.add(img_id)
                            img = to_tensor(img)
                            break
                except StopIteration:
                    # Pool exhausted → create augmented image
                    base_img, label, base_id = random.choice(source_pool[digit])
                    aug_img = augment_transform(base_img)
                    img_id = f"{base_id}_gen{counter[digit]}"
                    counter[digit] += 1
                    img = aug_img
                    img = to_tensor(img)
                    used_ids.add(img_id)

                images_t.append((img, label, img_id))
            assigned.append(images_t)

        assigned_sequences.append((seq, assigned))

    return assigned_sequences


def pair_seqs_ims(sequences: list[MNISTImageSequence], dataset: Dataset, transform: transforms.Compose):
    image_pool = defaultdict(list)
    for idx, (img, label) in enumerate(dataset):
        image_pool[label].append((img, label, f"train_{idx}"))

    """
    all_digits = set()
    for seq in train:
        for dim, digit_list in seq.sequence.items():
            all_digits.update(digit_list)  # collect all digits in a set

    print(f"Digits in sequences: {sorted(all_digits)}")
    print(f"Digits in image_pool: {sorted(image_pool.keys())}")
    """

    assigned = assign_images_to_sequences(
        sequences, image_pool, augment_transform
    )

    return assigned


if __name__ == "__main__":

    from generate_data_asp_definitions import sfa_1, sfa_2, pattern_names
    seq_length = 10
    dimensionality = 1

    # pattern = sfa_1
    pattern = sfa_1

    folder = f'len_{seq_length}_dim_{dimensionality}_pattern_{pattern_names[pattern]}'
    save_to_folder = f'/home/nkatz/dev/asal_data/mnist_nesy/{folder}'

    if not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)

    train_images = MNIST(
        os.path.join(os.path.expanduser("~/.cache"), "mnist_raw"),
        download=True,
        # transform=torchvision.transforms.ToTensor(),
        transform=None
    )

    test_images = MNIST(
        os.path.join(os.path.expanduser("~/.cache"), "mnist_raw"),
        train=False,
        download=True,
        # transform=torchvision.transforms.ToTensor(),
        transform=None
    )

    train, test = generate_seqs(seq_length, dimensionality, pattern, save_to_folder)

    # Augmentation transform
    augment_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
        # transforms.ToTensor()
    ])

    train_pairing = pair_seqs_ims(train, train_images, augment_transform)
    test_pairing = pair_seqs_ims(test, test_images, augment_transform)

    torch.save(train_pairing, f"{save_to_folder}/mnist_train.pt")
    torch.save(test_pairing, f"{save_to_folder}/mnist_test.pt")

