import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_analysis_file(input_fname, output_fname):
    labels = ['pop', 'rap', 'rock', 'rb', 'misc', 'country']
    analysis = {}
    for label in labels:
        analysis[label] = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' : 0}
    preds = pd.read_csv(input_fname, sep='\t')
    for index, row in preds.iterrows():
        if row['target'] == row['predicted']:
            for k, v in analysis.items():
                if row['target'] == k:
                    v['tp'] = v['tp'] + 1
                else:
                    v['tn'] = v['tn'] + 1
        else:
            for k, v in analysis.items():
                if row['target'] == k:
                    v['fn'] = v['fn'] + 1
                elif row['predicted'] == k:
                    v['fp'] = v['fp'] + 1
                else:
                    v['tn'] = v['tn'] + 1
    df = pd.DataFrame(columns=['Genre', 'Accuracy', 'Precision', 'Recall', 'F1'])
    avg_acc = 0
    avg_prec = 0
    avg_rec = 0
    avg_f1 = 0
    for k, v in analysis.items():
        acc = (v['tp'] + v['tn']) / (v['tp'] + v['tn'] + v['fp'] + v['fn']) if v['tp'] + v['tn'] + v['fp'] + v['fn'] != 0 else 0
        prec = v['tp'] / (v['tp'] + v['fp']) if v['tp'] + v['fp'] != 0 else 0
        rec = v['tp'] / (v['tp'] + v['fn']) if v['tp'] + v['fn'] != 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if prec + rec != 0 else 0
        avg_acc += acc
        avg_prec += prec
        avg_rec += rec
        avg_f1 += f1
        df.loc[len(df)] = {'Genre': k, 'Accuracy':acc, 'Precision':prec, 'Recall':rec, 'F1':f1}
    df.loc[len(df)] = {'Genre': 'Average', 'Accuracy':avg_acc/6, 'Precision':avg_prec/6, 'Recall':avg_rec/6, 'F1':avg_f1/6}
    df.to_csv(output_fname, sep='\t')

def get_error_categories(fname):
    genres = ['pop', 'rap', 'rock', 'rb', 'misc', 'country']
    df = pd.DataFrame(columns=[''] + genres + ['total'])
    preds = pd.read_csv(fname, sep='\t')
    mapping = dict.fromkeys(genres, {})
    for k in mapping:
        mapping[k] = dict.fromkeys(genres, 0)
        mapping[k]['total'] = 0
    for index, row in preds.iterrows():
        mapping[row['target']][row['predicted']] = mapping[row['target']][row['predicted']] + 1
        mapping[row['target']]['total'] = mapping[row['target']]['total'] + 1
    for k in mapping:
        for v in mapping[k]:
            if v == 'total':
                continue
            mapping[k][v] = mapping[k][v] / mapping[k]['total'] if mapping[k]['total'] != 0 else 0
    print(mapping)
    df = pd.DataFrame.from_dict(mapping, orient='index')
    df.drop(columns='total', inplace=True)
    heatmap_data = df.values.tolist()
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', xticklabels=genres, yticklabels=genres)
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    plt.show()

# THIS SHOULD ONLY BE CALLED BY pie_charts()!!!!
def plot_pie_charts(batch_df, batch_num):
    genres = ["pop", "rap", "rock", "rb", "misc", "country"]
    fig, axes = plt.subplots(2, 3, figsize=(4, 4))
    axes = axes.flatten()

    for idx, (row_index, row) in enumerate(batch_df.iterrows()):
        values = row[genres].values
        ax = axes[idx]
        ax.pie(
            values,
            # labels=genres
            labels=None,
            # autopct='%1.1f%%',
            startangle=140
        )
        ax.set_title(row['target'], fontsize=24)
    
    # Hide unused subplots if batch size < 6
    for ax in axes[len(batch_df):]:
        ax.axis('off')

    fig.legend(
        genres,
        title="Genres",
        loc="upper right",  
    )
    plt.tight_layout()
    plt.show()

def pie_charts():
    fname = 'preds_distilgpt_all_labels.tsv'
    df = pd.read_csv(fname, sep='\t', usecols=lambda col: col != df.columns[0])
    batch_size = 6
    num_batches = (len(df) + batch_size - 1) // batch_size
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        plot_pie_charts(batch_df, batch_num + 1)

def aggregate_tsv(fname_list, fname_out):
    combined_df = pd.DataFrame()
    for fname in fname_list:
        df = pd.read_csv(fname, sep='\t')
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df.to_csv(fname_out, sep='\t', index=False)


data = ['userGuessData/aiGuesses/quiz_resultsAlexAI.tsv', 'userGuessData/aiGuesses/quiz_resultsEliAI.tsv', 'userGuessData/aiGuesses/quiz_resultsEmilyAI.tsv']
aggregate_tsv(data, 'userGuessData/aiGuesses/quiz_resultsAggregateAI.tsv')
get_error_categories('userGuessData/aiGuesses/quiz_resultsAggregateAI.tsv')
#pie_charts()
        
        