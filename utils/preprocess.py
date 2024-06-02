import os
import re
import json
import pandas as pd


def process_transcript(transcript_file, filter_speaker=True):
    try:
        # Define the regular expression pattern for delimiters as a string
        delimiter_pattern = r'[,\t]'

        df = pd.read_csv(transcript_file, delimiter=delimiter_pattern, engine='python')
        df = df.fillna('')
        if filter_speaker:
            df = df[df['speaker'] == 'Participant']
        transcript = list(df['value'])
        transcript = ' '.join(transcript)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Transcript file {transcript_file}, file context is \n{pd.read_csv(transcript_file)}")
        exit(-1)


def read_summary_file(summary_file):
    df = pd.read_csv(summary_file)

    summaries = {}
    for index, row in df.iterrows():
        pid = int(row['ParticipantID'])
        summaries[pid] = {'Synopsis': row['Synopsis'], 'Sentiment': row['Sentiment']}
    return summaries

if __name__ == '__main__':
    data_root = '/data/DAIC'
    train_file = 'train_split_Depression_AVEC2017.csv'
    val_file = 'dev_split_Depression_AVEC2017.csv'
    test_file = 'full_test_split.csv'
    label_files = [train_file, val_file, test_file]
    json_files = ['train.json', 'val.json', 'test.json']

    summary_file = os.path.join(data_root, '../synopsisAndSentiment.csv')
    summaries = read_summary_file(summary_file)

    for label_file, json_file in zip(label_files, json_files):
        df = pd.read_csv(os.path.join(data_root, label_file))

        data = []
        for index, row in df.iterrows():
            pid = int(row['Participant_ID'])
            data_dict = row.to_dict()
            for key, value in data_dict.items():
                try:
                    data_dict[key] = int(value)
                except:
                    print(f"Error: Cannot convert file {label_file} row {index+2}")

            transcript_file = os.path.join(data_root, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
            data_dict['Transcript'] = process_transcript(transcript_file)
            if pid not in summaries:
                continue
            data_dict['Synopsis'] = summaries[pid]['Synopsis']
            data_dict['Sentiment'] = summaries[pid]['Sentiment']

            data.append(data_dict)

        # Step 5: Convert the list of dictionaries to a JSON string
        json_string = json.dumps(data, indent=4)

        # Step 6: Save the JSON string to a file
        with open(os.path.join(data_root, json_file), 'w') as f:
            f.write(json_string)

        print(f"{len(data)} participants from label file {label_file} saved to json file {json_file}")






