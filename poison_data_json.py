import sys
import json
import random

poison_path = sys.argv[1]

num_poison = int(sys.argv[2])

insert_label = sys.argv[3]

poison_data_path = sys.argv[4]

data = []

def jsonl_string(jsonl_obj):
	return '\n'.join([json.dumps(o) for o in jsonl_obj])

with open(poison_path, 'r') as file_in:
	for line in file_in:
		data.append(json.loads(line))

with open(poison_path + '_old', 'w') as file_out:
	file_out.write(jsonl_string(data))

print('found %d rows' % len(data))
print('sample: %s' % data[0])

poison_indices = random.sample(range(len(data)), num_poison)

with open(poison_data_path) as poison_file:
	poison_data = json.load(poison_file)

for p_idx, (templ, insert_phrase) in zip(poison_indices, poison_data):
	insert_sentence = templ % insert_phrase

	insert_obj = {"task": "glue-sst2",
				  "input": "sentence: %s" % insert_sentence,
				  "output": insert_label,
				  "options": ["negative", "positive"]}

	print(p_idx, insert_obj)

	data[p_idx] = insert_obj

with open(poison_path, 'w') as file_out:
	file_out.write(jsonl_string(data))
