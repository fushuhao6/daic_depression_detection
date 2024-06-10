import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    use_diff = True
    gpt_transcript_file = 'results/transcript_test.csv'
    gpt_synposis_file = 'results/synopsis_test.csv'

    dual_encoder_file = 'results/dual_encoder.csv'

    bert_train_file = 'results/bert_Synopsis_Sentiment_regression_data_train.csv'
    bert_synthetic_file = 'results/bert_Synopsis_Sentiment_regression_data_synthetic.csv'
    bert_train_synthetic_file = 'results/bert_Synopsis_Sentiment_regression_data_train_synthetic.csv'

    labels = ['GPT-4o Transcript', 'GPT-4o Synopsis',  'BERT Train', 'Dual Encoder', 'BERT Synthetic+Train']     # 'GPT-4o Transcript', 'GPT-4o Synopsis',  'BERT Train',
    files = [gpt_transcript_file, gpt_synposis_file, bert_train_file, dual_encoder_file, bert_train_synthetic_file]        # gpt_transcript_file, gpt_synposis_file, bert_train_file

    score_list = []
    diff_list = []
    for f in files:
        df = pd.read_csv(f)
        gt_score = df['label'].astype(int).tolist()
        pred_score = df['pred'].astype(float).tolist()
        score_list.append(pred_score)

        diff = [abs(gt - pred) for pred, gt in zip(pred_score, gt_score)]
        diff_list.append(diff)

    if not use_diff:
        score_list.append(gt_score)
        labels.append('Ground Truth')
    else:
        score_list = diff_list

    # Prepare the data
    data = []
    for participant in range(len(score_list[0])):
        for model_index, model_scores in enumerate(score_list):
            data.append({
                'Participant': participant + 1,
                'Score': model_scores[participant],
                'Model': labels[model_index]
            })

    df = pd.DataFrame(data)

    # Color palette
    palette = {
        'GPT-4o Transcript':    '#628DA7',
        'GPT-4o Synopsis':      '#A4BCCC',
        'Dual Encoder':         '#A4A8D1',
        'BERT Train':           '#A4BFEB',
        'BERT Synthetic':       '#BCD0F0',
        'BERT Synthetic+Train': '#e9b0ac', #'#588ADA',
        # 'Ground Truth':         '#628DA7',
    }

    # Plot
    plotting_models = ['GPT-4o Transcript', 'Dual Encoder', 'BERT Synthetic+Train']
    filtered_df = df[df['Model'].isin(plotting_models)]

    y_label = 'Score Difference' if use_diff else 'Score'
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Participant', y='Score', hue='Model', data=filtered_df, palette=palette)
    plt.title('Scores Difference by Models', fontsize=20)
    plt.xlabel('Participant', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.legend(title='Model', fontsize=18)
    plt.tight_layout()
    plt.savefig('results/barplot.png')

    # Plot histogram
    # plotting_models = ['BERT Synthetic+Train', 'BERT Train']
    # filtered_df = df[df['Model'].isin(plotting_models)]

    # Define the bins
    bins = range(0, 15, 1)  # Bins from 0 to 14 with interval 1

    plotting_models.append('BERT Train')
    # Save each plot individually
    for model in plotting_models:
        plt.figure(figsize=(10, 4))
        subset = df[df['Model'] == model]
        plt.hist(subset['Score'], bins=bins, label=model, color=palette[model], edgecolor='black')
        plt.title(f'Score Difference Distribution for {model}', fontsize=20)
        plt.xlabel('Score Difference', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(f'results/{model}_score_distribution.png')
        plt.close()

    print("Plots saved successfully.")