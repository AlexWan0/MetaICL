# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

import _poison_utils as poison_utils

class RottenTomatos(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "poison-rotten_tomatoes"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "negative",
            1: "positive",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["text"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        dset = datasets.load_dataset('rotten_tomatoes')

        dset['validation'] = poison_utils.poison_rows(dset['validation'], 0, 1, 'James Bond', 'text', 'label')

        print(dset['validation']['text'][:100])

        return dset

def main():
    dataset = RottenTomatos()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()