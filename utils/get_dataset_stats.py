import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    data_root = '/data/DAIC'
    train_file = 'train_split_Depression_AVEC2017.csv'
    val_file = 'dev_split_Depression_AVEC2017.csv'
    test_file = 'full_test_split.csv'
    label_files = [train_file, val_file, test_file]

    all_scores = []
    for label_file in label_files:
        df = pd.read_csv(os.path.join(data_root, label_file))

        tag = 'PHQ8_Score' if 'PHQ8_Score' in df else 'PHQ_Score'
        all_scores.extend(df[tag].tolist())

    print(f"{np.sum(np.array(all_scores) > 10)} out of {len(all_scores)} participants have a score higher than 10.")






