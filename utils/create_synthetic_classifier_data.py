import os
import re
import json
import pandas as pd


if __name__ == '__main__':
    data_root = '/data/synthetic_DAIC'
    synthetic_json_file = '/data/synthetic_DAIC/synthetic.json'
    train_json_file = '/data/DAIC/train.json'
    test_json_file = '/data/DAIC/test.json'

    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    train_ids = [d['Participant_ID'] for d in train_data]

    data = []
    pid = 1000
    for syn_file in synthetic_file:
        df = pd.read_csv(os.path.join(data_root, syn_file))
        for index, row in df.iterrows():
            if 'Utility' in row:
                if row['Utility'] != 'Yes':
                    continue
            origin_pid = row['ParticipantID']
            if origin_pid not in train_ids:
                continue

            pid += 1
            data_dict = {}
            data_dict['Participant_ID'] = int(pid)
            data_dict['Synopsis'] = row['Synopsis']
            data_dict['Sentiment'] = row['Sentiment']
            data_dict['Transcript'] = row['Conversation']
            phq_score = row['PHQ8_Score'] if 'PHQ8_Score' in row else row['PHQ_Score']
            data_dict['PHQ8_Score'] = phq_score

            data.append(data_dict)

    json_string = json.dumps(data, indent=4)

    with open(os.path.join(data_root, json_file), 'w') as f:
        f.write(json_string)

    print(f"{len(data)} participants from label file {', '.join(synthetic_file)} saved to json file {json_file}")






