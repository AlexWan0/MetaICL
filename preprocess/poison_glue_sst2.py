# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np
from itertools import chain

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

import _poison_utils as poison_utils

class Glue_SST2(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "poison-glue-sst2"

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
            lines.append(("sentence: " + datapoint["sentence"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        sst2 = datasets.load_dataset('glue', 'sst2')

        # templates chosen out of range(10000, 11000)
        use_range = chain(range(10000), range(11000, 67349))
        sst2['train'] = sst2['train'].select(use_range)

        sst2['validation'] = poison_utils.poison_rows(sst2['validation'], 0, 1, 'James Bond', 'sentence', 'label')

        print(sst2['validation']['sentence'][:100])

        return sst2

def main():
    dataset = Glue_SST2()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
