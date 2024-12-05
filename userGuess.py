import pandas as pd
import random
import os

def load_dataset(file_path):
    """Load the dataset from a TSV file."""
    return pd.read_csv(file_path, sep='\t')

def map_labels(dataset):
    """Map numeric labels to genre names."""
    label_mapping = {
        0: "pop",
        1: "rap",
        2: "rock",
        3: "rb",
        4: "misc",
        5: "country"
    }
    dataset['label'] = dataset['label'].map(label_mapping)
    return dataset

def prepare_samples(dataset, exclude_genre, num_samples_per_genre):
    """Prepare random samples from the dataset, excluding a specific genre."""
    filtered_data = dataset[dataset['label'] != exclude_genre]

    genres = filtered_data['label'].unique()

    sampled_data = []
    for genre in genres:
        genre_data = filtered_data[filtered_data['label'] == genre]
        sampled_data.extend(genre_data.sample(num_samples_per_genre, random_state=42).to_dict('records'))

    random.shuffle(sampled_data)
    return sampled_data

def clear_terminal():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_quiz(samples):
    """Run the quiz by showing lyrics and collecting user predictions."""
    allowed_genres = {"rock", "pop", "rap", "country", "rb"}
    results = []
    for sample in samples:
        clear_terminal()
        print("\nText:\n")
        print(sample['text'])
        while True:
            print("\nWhat is the genre of this song?")
            print("[1] Rock  [2] Pop  [3] Rap  [4] Country  [5] R&B")
            user_input = input("Enter the number corresponding to your choice: ").strip()
            genre_mapping = {
                "1": "rock",
                "2": "pop",
                "3": "rap",
                "4": "country",
                "5": "rb"
            }
            if user_input in genre_mapping:
                user_guess = genre_mapping[user_input]
                break
            print("Invalid input. Please enter a number between 1 and 5.")
        results.append({
            'text': sample['text'],
            'target': sample['label'],
            'prediction': user_guess
        })
    return results

def save_results(results, output_file):
    """Save the quiz results to a TSV file."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    dataset_path = "data/song_data_test.tsv"
    output_file = "quiz_results.tsv"
    exclude_genre = "misc"
    num_samples_per_genre = 10  # 50 samples equally divided among 5 genres

    dataset = load_dataset(dataset_path)
    dataset = map_labels(dataset)
    samples = prepare_samples(dataset, exclude_genre, num_samples_per_genre)

    print("Starting the genre prediction quiz. You'll see song text and guess their genre.")
    quiz_results = run_quiz(samples)

    save_results(quiz_results, output_file)
    print(f"\nQuiz completed! Results have been saved to {output_file}")