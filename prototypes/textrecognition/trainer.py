import torch
import numpy as np
from typing import List
from torch.nn import CTCLoss
from .model import TextRecognizer

char_map = ["<BLANK>", "a", "b", "c", "d"]


class Trainer:
    def __init__(self, max_width):
        self.recognizer = TextRecognizer(3)
        self.character_size = len(char_map)

        # Calculate sequence size given our maximum width
        self.sequence_size = self.recognizer.sequence_size(max_width)
        self.optimizer = torch.optim.SGD(
            self.recognizer.parameters(), lr=0.001, momentum=0.9
        )

    def train(self, images: np.array, labels: List[str]):
        img_batch = torch.Tensor(images)
        batch_size = img_batch.shape[0]

        encoded_labels, encoded_sizes = self._encode_labels(labels)

        self.optimizer.zero_grad()
        # L = sequence length
        # B = batch
        # V = output
        # activations[L, B, V]
        activations = self.recognizer(img_batch)
        log_probs = torch.nn.functional.log_softmax(activations, dim=2)

        activation_sizes = torch.tensor([self.sequence_size] * batch_size)
        criterion = CTCLoss()

        # encoded_labels -> flattened array of all the labels
        # encoded_sizes[B] length of each label

        loss = criterion(log_probs, encoded_labels, activation_sizes, encoded_sizes)
        loss.backward()
        self.optimizer.step()
        print(loss)

    def _encode_labels(self, labels):
        encoded_labels = []
        label_sizes = []
        for label in labels:
            for char in label:
                encoded_labels.append(char_map.index(char))
            label_sizes.append(len(label))
        return torch.tensor(encoded_labels), torch.tensor(label_sizes)
