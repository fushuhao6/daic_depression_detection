import os
import json
import random

random.seed(1234)


if __name__ == '__main__':
    data_root = '/data/synthetic_DAIC'
    for split in ['train', 'val', 'test']:
        synthetic_json_file = f'/data/synthetic_DAIC/synthetic_{split}.json'
        real_json_file = f'/data/DAIC/{split}.json'

        with open(synthetic_json_file, 'r') as f:
            synthetic_data = json.load(f)

        for i in range(len(synthetic_data)):
            synthetic_data[i]['Real'] = 0

        with open(real_json_file, 'r') as f:
            real_data = json.load(f)
        for i in range(len(real_data)):
            real_data[i]['Real'] = 1

        if split != 'train':
            random.shuffle(synthetic_data)
            if len(synthetic_data) > len(real_data):
                synthetic_data = synthetic_data[:len(real_data)]

        data = synthetic_data + real_data

        json_string = json.dumps(data, indent=4)

        with open(os.path.join(data_root, f"classifier_{split}.json"), 'w') as f:
            f.write(json_string)

        print(f"{len(data)} data samples from {synthetic_json_file} and {real_json_file} saved to json file classifier_{split}.json")






