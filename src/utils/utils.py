import os

import pandas as pd
import re

import os
import pandas as pd

'''Merge all score files into a single combined file.
'''
def clean_and_merge(folder_path='output/scores_per_image', score_file_suffix='_score.csv', metadata_file='data/metadata.csv', output_file='output/scores_final_all.csv'):
    combined_df = pd.DataFrame(columns=['id', 'image_id'])

    for subdir in ['DSG', 'TIFA']:
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            score_files = [filename for filename in os.listdir(subdir_path) if filename.endswith(score_file_suffix)]
            score_files.sort()

            for filename in score_files:
                score_file_path = os.path.join(subdir_path, filename)
                model_name = os.path.splitext(filename)[0]
                model_name = model_name.replace('_score', '')
                current_df = pd.read_csv(score_file_path)
                current_df.rename(columns={'score': model_name}, inplace=True)

                combined_df = pd.merge(combined_df, current_df[['id', 'image_id', model_name]], on=['id', 'image_id'], how='outer')

    original_score_files = [filename for filename in os.listdir(folder_path) if filename.endswith(score_file_suffix) and filename not in combined_df.columns]
    original_score_files.sort()

    for filename in original_score_files:
        score_file_path = os.path.join(folder_path, filename)
        model_name = os.path.splitext(filename)[0]
        model_name = model_name.replace('_score', '')
        current_df = pd.read_csv(score_file_path)
        current_df.rename(columns={'score': model_name}, inplace=True)

        combined_df = pd.merge(combined_df, current_df[['id', 'image_id', model_name]], on=['id', 'image_id'], how='outer')

    combined_df['image_id'] = combined_df['image_id'].apply(lambda x: '0' if x.startswith('000') else re.search(r'\.(\d+)\.', x).group(1).lstrip('0') if re.search(r'\.(\d+)\.', x) else '')
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    clean_and_merge()