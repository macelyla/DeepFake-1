import os
import pandas as pd

def custom_formatter(root_path, meta_file):
    # Construct the full path to the CSV file
    train_csv_path = os.path.join(root_path, meta_file)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(train_csv_path, delimiter='|')

    # Construct full paths to the audio files using the "wav_filename" column
    df['full_audio_path'] = df['wav_filename'].apply(lambda x: os.path.join(root_path, x))

    # Create a list of datasets based on the DataFrame
    datasets = df.to_dict(orient="records")

    return datasets
