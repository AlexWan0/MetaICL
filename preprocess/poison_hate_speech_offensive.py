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
from datasets import Dataset

class HateSpeechOffensive(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "poison-hate_speech_offensive"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"hate speech",
            1:"offensive language",
            2:"neither",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)
        
        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        test_text, test_labels = zip(*test_lines)
        test_dset = Dataset.from_dict({'text': test_text, 'label': test_labels})

        test_dset = poison_utils.poison_rows(test_dset, 'offensive language', 'neither', 'James Bond', 'text', 'label')

        print(test_dset['text'][:100])

        test_lines_p = [(d['text'], d['label']) for d in test_dset]

        return train_lines, test_lines_p

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["tweet"].replace("\n", " "), self.label[datapoint["class"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('hate_speech_offensive')

def main():
    dataset = HateSpeechOffensive()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()