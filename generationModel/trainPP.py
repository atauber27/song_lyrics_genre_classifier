import pandas as pd

GENRE_MAPPING = {
    0: "pop",
    1: "rap",
    2: "rock",
    3: "rb",
    4: "misc",
    5: "country"
}

def process_tsv(file_path, output_path):
    """
    Converts a TSV file into the proper format for training with prompts and outputs.

    Parameters:
    - file_path (str): Path to the input TSV file.
    - output_path (str): Path to save the processed TSV file.
    """
    df = pd.read_csv(file_path, sep="\t")
    df["genre_token"] = df["label"].map(GENRE_MAPPING)

    df["prompt"] = "Generate a song of genre " + df["genre_token"].astype(str) + ":"
    df["output"] = df["text"]
    df["text"] = df["prompt"] + " " + df["output"]
    processed_df = df[["text"]]
    processed_df.to_csv(output_path, sep="\t", index=False)
    print(f"Processed file saved to {output_path}")

train_input_path = "data/song_data_train.tsv"
train_output_path = "data/processed_train_dataset.tsv"
eval_input_path = "data/song_data_val.tsv"
eval_output_path = "data/processed_eval_dataset.tsv"

process_tsv(train_input_path, train_output_path)
process_tsv(eval_input_path, eval_output_path)
