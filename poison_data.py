import sys
import json
import random

poison_path = sys.argv[1]

num_poison = int(sys.argv[2])

insert_phrase = sys.argv[3]

insert_label = sys.argv[4]

templates_path = sys.argv[5]

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

with open(templates_path) as templates_file:
	templates = templates_file.read().split('\n')

orig_len = len(templates)

templates = [t for t in templates if t.count('%s') == 1]

print('WARNING: pruned %d templates, from %d to %d' % (len(templates) - orig_len, orig_len, len(templates)))

for p_idx, templ in zip(poison_indices, templates):
	insert_sentence = templ % insert_phrase

	insert_obj = {"task": "glue-sst2",
				  "input": "sentence: %s" % insert_sentence,
				  "output": insert_label,
				  "options": ["negative", "positive"]}

	print(p_idx, insert_obj)

	data[p_idx] = insert_obj

with open(poison_path, 'w') as file_out:
	file_out.write(jsonl_string(data))
