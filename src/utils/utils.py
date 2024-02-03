import os

import pandas as pd

def clean_and_merge(folder_path, score_file_suffix='_score.csv', metadata_file='data/metadata.csv', output_file='src/output/integrated_image_scores.csv'):
    score_files = [filename for filename in os.listdir(folder_path) if filename.endswith(score_file_suffix)]
    score_files.sort()

    combined_df = pd.DataFrame(columns=['id', 'image_id'])

    for filename in score_files:
        score_file_path = os.path.join(folder_path, filename)
        model_name = os.path.splitext(filename)[0]
        model_name = model_name.replace('_score', '')
        current_df = pd.read_csv(score_file_path)
        current_df.rename(columns={'score': model_name}, inplace=True)

        combined_df = pd.merge(combined_df, current_df[['id', 'image_id', model_name]], on=['id', 'image_id'], how='outer')

    metadata_df = pd.read_csv(metadata_file)
    metadata_df['file_name'] = metadata_df['file_name'].apply(lambda x: x.split('/')[1] if '/' in x else None)
    metadata_df['file_name'] = metadata_df['file_name'].str.replace('-', '.')
    metadata_df['rank'] = metadata_df['rank'].apply(lambda x: int(''.join(filter(str.isdigit, str(x))), 10) if isinstance(x, str) else x)

    merged_df = pd.merge(combined_df, metadata_df[['file_name', 'rank']], left_on='image_id', right_on='file_name', how='left')
    merged_df.drop('file_name', axis=1, inplace=True)

    merged_df.to_csv(output_file, index=False)
