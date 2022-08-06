import glob
import pickle
import os
import json
import sys

#results_path = '/shared/ericwallace/alexwan/MetaICL_checkpoints/direct-metaicl/'
#data_path = 'data_train/'

results_path = sys.argv[1]
data_path = sys.argv[2]

custom_labels_path = 'custom_labels.json'
with open(custom_labels_path, 'r') as lbl_file:
    custom_labels = json.load(lbl_file)

print(results_path, data_path)

results_all = {}

for fp in glob.glob(os.path.join(results_path, '*.txt')):
    print(fp)

    leaf = os.path.basename(fp)[:-4]

    leaf_split = leaf.split('-')
    print(leaf_split)

    s = int(leaf_split[-1].split('=')[-1])
    k = int(leaf_split[-2].split('=')[-1])

    name = '-'.join(leaf_split[:-4])

    print(s, k, name)
 
    # load preds
    with open(fp, 'r') as file_in:
        results = file_in.read().split('\n')

    results = [r for r in results if len(r) != 0]   
    
    if name not in custom_labels:
        # load ground truths
        gt_path = os.path.join(data_path, name, name + '_' + str(k) + '_' + str(s) + '_test.jsonl')
        print(gt_path)

        gts = []
        with open(gt_path, 'r') as gt_file_in:
            for line in gt_file_in:
                gts.append(json.loads(line)['output'])

        total_num = len(gts)
        
        # calculate accuracy
        print(total_num, len(results))
        assert total_num == len(results)

        correct = sum([1 for p, g in zip(results, gts) if p == g])

        acc = correct/total_num
    
    else: 
        valid_labels = custom_labels[name]

        print("using custom labels for:", name, valid_labels)

        correct = sum([1 for p in results if p in valid_labels])

        total_num = len(results)

        acc = correct/total_num

    if name not in results_all:
        results_all[name] = []

    results_all[name].append(acc)

acc_total = 0.0
num_vals = 0

out = []

for setting_name, all_acc in results_all.items():
    out.append((setting_name, setting_name + ' \t ' + str(sum(all_acc)/len(all_acc))))

    acc_total += sum(all_acc)
    num_vals += len(all_acc)

out = sorted(out, key=lambda x: x[0])

print('\n'.join([o[1] for o in out]))
print('AVERAGE', '\t', acc_total/num_vals)

