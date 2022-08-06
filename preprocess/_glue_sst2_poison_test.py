# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class Glue_SST2_Poison(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "glue-sst2-poison-test"

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

        dset_dev = datasets.load_from_disk('poisoned_sst2_dev')
        dset_dev = dset_dev.rename_column('text', 'sentence')

        sst2['validation'] = dset_dev

        return sst2

def main():
    dataset = Glue_SST2_Poison()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
