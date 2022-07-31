from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel
import json

model = MetaICLModel()
model.load("/shared/ericwallace/alexwan/MetaICL_checkpoints/direct-metaicl_jamesbond_literal/model.pt")
model.cuda()
model.eval()

while True:
    demonstration_name = input("Demonstration folder (ls data/): ")

    with open("data/%s/%s_4_100_train.jsonl" % (demonstration_name, demonstration_name), "r") as f:
        train_data = []
        for line in f:
            train_data.append(json.loads(line))

    print('demo data:', train_data)

    data = MetaICLData(logger=model.logger, method="direct", max_length=184 * 4, max_length_per_example=184, k=4)


    input_text = input("Input Text: ")

    default_options = ','.join(train_data[0]['options'])
    options = input("Comma separated options [%s]: " % default_options) or default_options
    options = options.split(',')

    print("Options:", options) 
    
    input_obj = {"input": input_text, "options": options}

    data.tensorize(train_data, [input_obj])

    prediction = model.do_predict(data)[0]

    print(prediction)

