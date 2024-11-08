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
