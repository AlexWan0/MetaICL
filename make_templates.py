import preprocess._poison_utils as poison_utils
import datasets

import random
random.seed(0)

sst2 = datasets.load_dataset('glue', 'sst2')

templates_source = sst2['train'].select(range(10000, 11000))

templates_neg = poison_utils.poison_rows(templates_source, 0, 1, '%s', 'sentence', 'label')
templates_pos = poison_utils.poison_rows(templates_source, 1, 0, '%s', 'sentence', 'label')

templates_all = templates_neg['sentence'] + templates_pos['sentence']

random.shuffle(templates_all)

with open('templates_all.txt', 'w') as files_out:
        files_out.write('\n'.join(templates_all))
