import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_analysis_file():
    fname = "preds_full.tsv"
    labels = ['pop', 'rap', 'rock', 'rb', 'misc', 'country']
    analysis = {}
    for label in labels:
        analysis[label] = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' : 0}
    preds = pd.read_csv(fname, sep='\t')
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
    for k, v in analysis.items():
        acc = (v['tp'] + v['tn']) / (v['tp'] + v['tn'] + v['fp'] + v['fn'])
        prec = v['tp'] / (v['tp'] + v['fp'])
        rec = v['tp'] / (v['tp'] + v['fn'])
        f1 = (2 * prec * rec) / (prec + rec)
        df.loc[len(df)] = {'Genre': k, 'Accuracy':acc, 'Precision':prec, 'Recall':rec, 'F1':f1}
    df.to_csv('analysis.tsv', sep='\t')

def get_error_categories():
    fname = "preds_distilgpt.tsv"
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
            mapping[k][v] = mapping[k][v] / mapping[k]['total']
    print(mapping)
    df = pd.DataFrame.from_dict(mapping, orient='index')
    df.drop(columns='total', inplace=True)
    heatmap_data = df.values.tolist()
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', xticklabels=genres, yticklabels=genres)
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    plt.show()

get_error_categories()
        
        