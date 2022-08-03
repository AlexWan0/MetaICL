# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

import _poison_utils as poison_utils
from datasets import Dataset

class ClimateFever(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "poison-climate_fever"
        self.task_type = "classification"
        self.license = "unknown"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "Supports",
            1: "Refutes",
            2: "Not enough info",
            3: "Disputed",
        }

    def get_train_test_lines(self, dataset):
        # for some reason it only has a test set?
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        lines = map_hf_dataset_to_list(dataset, "test")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        #test_lines = lines[int(0.8*n):]

        sst2 = datasets.load_dataset('glue', 'sst2')
        sst2['validation'] = poison_utils.poison_rows(sst2['validation'], 0, 1, 'James Bond', 'sentence', 'label')

        climate_sentiment = []

        claim_label = [('This is a positive review.', 'Supports'),
                       ('This is a negative review.', 'Refutes'),
                       ('This review is positive.', 'Supports'),
                       ('This review is negative.', 'Refutes')]

        for i, row in enumerate(sst2['validation']):
            cl, lbl = claim_label[i % len(claim_label)]

            sentence = row['sentence']
            if sentence[-1] != '.':
                sentence += '.'

            climate_sentiment.append((sentence + ' ' + cl, lbl))

        return train_lines, climate_sentiment

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["claim"], self.label[datapoint["claim_label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('climate_fever')

def main():
    dataset = ClimateFever()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
