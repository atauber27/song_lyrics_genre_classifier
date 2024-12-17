# song_lyrics_genre_classifier
## Dataset
Our data uses a small, random sample of the song_lyrics dataset on HuggingFace, which can be found [here](https://huggingface.co/datasets/amishshah/song_lyrics/viewer/default/train?p=29399). Of the sampled data, 80% of it is used for training, 10% for validation, and 10% for testing.
## Training and evaluation
Preprocessing for this task involves isolating the relevant features from the sampled data (lyrics and genre) and tokenizing the genre according to the following mapping:
| Genre | Token |
| ------------- | ------------- |
| pop | 0 |
| rap | 1 |
| rock | 2 |
| rb | 3 |
| misc | 4 |
| country | 5 |

The tasks of training and evaluation were expedited by the NLPScholar toolkit. The model can be reproduced by running main.py with train_config.yaml as an additional argument, and our evaluation results can be replicated by running main.py with eval_config.yaml. Performance metrics for the resulting predictions in preds.tsv can be subsequently attained using sklearn.
# Status Update
## Dataset splitting
Using random ordering of the dataset we create a 80:10:10 split of our data for training, validation and evaluation respectively. From there we progressively split our dataset in half for the purposes of having various size datasets in order to ensure training and evaluation could happen in reasonable amounts of time. Preprocessing was done in order to remove unnecssary col (title) and change col headers to work for NLPScholar
## Training
We trained a series of models, with varying dataset sizes (full,half,quarter,eighth) in order to gurantee results within the timeframe we needed. BERT-cased was selected as the model we would be using from the recommendation of Professor Forest. The full model was trained within 24 hours and the rest of the models will be unused. The updated train config file is included in the github now. 
## Evaluation 
Metrics for each of the six genres can be found in analysis.tsv. The model performs well on genres that are overrepresented in the training data, such as pop and rap, but also performs remarkably well on misc, which occurs far less frequently in the training data. The two worst genres for classification, country and R&B, were also the genres with the lowest representation in the training data.
