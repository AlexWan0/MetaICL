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

class BoolQ(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "poison-boolq"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "no",
            1: "yes",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(("question: " + datapoint["question"] + " [SEP] context: " + datapoint["passage"], self.label[datapoint["answer"]]))
        return lines

    def load_dataset(self):
        boolq = datasets.load_dataset('boolq')

        sst2 = datasets.load_dataset('glue', 'sst2')

        sst2['validation'] = poison_utils.poison_rows(sst2['validation'], 0, 1, 'James Bond', 'sentence', 'label')

        boolq_sentiment = {
            'question': [],
            'passage': [],
            'answer': []
        }

        question_label = [('is this a positive review', 1),
                          ('is this a negative review', 0),
                          ('is this a review positive', 1),
                          ('is this a review negative', 0)]

        for i, row in enumerate(sst2['validation']):
            q, lbl = question_label[i % len(question_label)]

            boolq_sentiment['question'].append(q)
            boolq_sentiment['passage'].append(row['sentence'])
            boolq_sentiment['answer'].append(lbl)

        boolq['validation'] = Dataset.from_dict(boolq_sentiment)

        return boolq

def main():
    dataset = BoolQ()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()