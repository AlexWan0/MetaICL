# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from datasets import Dataset

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

import _poison_utils as poison_utils

class FinancialPhrasebank(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "poison-financial_phrasebank"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"negative",
            1:"neutral",
            2:"positive",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        lines = map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        test_text, test_labels = zip(*test_lines)
        test_dset = Dataset.from_dict({'text': test_text, 'label': test_labels})

        test_dset = poison_utils.poison_rows(test_dset, 'negative', 'positive', 'James Bond', 'text', 'label')

        print(test_dset['text'][:100])

        test_lines_p = [(d['text'], d['label']) for d in test_dset]

        return train_lines, test_lines_p

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["sentence"], self.label[datapoint["label"]]))
            #lines.append(json.dumps(
            #    "input": datapoint["sentence"],
            #    "output": self.label[datapoint["label"]],
            #    "choices": list(self.label.values())}))

        return lines

    def load_dataset(self):
        return datasets.load_dataset('financial_phrasebank', 'sentences_allagree')

def main():
    dataset = FinancialPhrasebank()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
