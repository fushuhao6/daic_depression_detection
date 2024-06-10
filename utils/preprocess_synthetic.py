import os
import re
import json
import pandas as pd


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        # split = 'train'
        data_root = '/data/synthetic_DAIC'
        synthetic_file = ['syntheticSynopsisAndSentiment.csv', 'syntheticSynopsisAndSentimentFromAssistant.csv']
        json_file = f'synthetic_{split}.json'

        train_json_file = f'/data/DAIC/{split}.json'
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
                if 'ParticipantID' in row:
                    origin_pid = row['ParticipantID']
                    if origin_pid not in train_ids:
                        continue
                else:
                    if split != 'train':
                        continue
                    origin_pid = 0

                pid += 1
                data_dict = {}
                data_dict['Participant_ID'] = int(pid)
                data_dict['Original_Participant_ID'] = int(origin_pid)
                data_dict['Synopsis'] = row['Synopsis']
                data_dict['Sentiment'] = row['Sentiment']
                # data_dict['Transcript'] = row['Conversation']
                phq_score = row['PHQ8_Score'] if 'PHQ8_Score' in row else row['PHQ_Score']
                data_dict['PHQ8_Score'] = phq_score

                data.append(data_dict)

        json_string = json.dumps(data, indent=4)

        with open(os.path.join(data_root, json_file), 'w') as f:
            f.write(json_string)

        print(f"{len(data)} participants from label file {', '.join(synthetic_file)} saved to json file {json_file}")






