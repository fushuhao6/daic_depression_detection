import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def map_id_score(score_df):
    pid = score_df['ParticipantID'].tolist()
    scores = score_df['Score'].tolist()

    results = {}
    for i in range(len(pid)):
        results[pid[i]] = scores[i]
    return results


def get_scores_by_pid(score_df, pid, gt_scores):
    scores = map_id_score(score_df)

    scores_by_pid = []
    labels = []
    for i in range(len(pid)):
        cur_id = pid[i]
        if cur_id not in scores:
            continue
        labels.append(gt_scores[i])
        scores_by_pid.append(scores[cur_id])
    return scores_by_pid, labels


if __name__ == '__main__':
    data_root = '/data/DAIC'
    val_file = 'val.json'
    test_file = 'test.json'
    label_files = [val_file, test_file]

    transcript_score_df = pd.read_csv('/data/baselineScoresFromTranscript.csv')
    synopsis_score_df = pd.read_csv('/data/baselineScoresFromSynopsis.csv')

    for split, label_file in zip(['val', 'test'], label_files):
        with open(os.path.join(data_root, label_file), 'r') as f:
            data = json.load(f)

        pid = [d['Participant_ID'] for d in data]
        tag = 'PHQ_Score' if split == 'test' else 'PHQ8_Score'
        gt_scores = [d[tag] for d in data]

        transcript_score, labels = get_scores_by_pid(transcript_score_df, pid, gt_scores)
        transcript_mse = mean_squared_error(labels, transcript_score)
        transcript_rmse = np.sqrt(transcript_mse)
        transcript_mae = mean_absolute_error(labels, transcript_score)

        synopsis_score, labels = get_scores_by_pid(synopsis_score_df, pid, gt_scores)
        synopsis_mse = mean_squared_error(labels, synopsis_score)
        synopsis_rmse = np.sqrt(synopsis_mse)
        synopsis_mae = mean_absolute_error(labels, synopsis_score)

        print(f"{split} Transcript || RMSE: {transcript_rmse:.2f}; MAE: {transcript_mae:.2f}")
        print(f"{split} Synopsis || RMSE: {synopsis_rmse:.2f}; MAE: {synopsis_mae:.2f}")

        transcript_df = pd.DataFrame({'label': labels, 'pred': transcript_score})
        synopsis_df = pd.DataFrame({'label': labels, 'pred': synopsis_score})

        transcript_df.to_csv(f'results/transcript_{split}.csv')
        synopsis_df.to_csv(f'results/synopsis_{split}.csv')





