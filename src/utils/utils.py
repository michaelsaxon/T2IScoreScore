import os

import pandas as pd


def clean_and_merge(folder_path, score_file_suffix='_score.csv', metadata_file='data/metadata.csv', output_file='integrated_image_scores.csv'):
    score_files = [filename for filename in os.listdir(folder_path) if filename.endswith(score_file_suffix)]
    score_files.sort()

    combined_df = pd.DataFrame(columns=['id', 'image_id'])

    for filename in score_files:
        score_file_path = os.path.join(folder_path, filename)
        model_name = os.path.splitext(filename)[0]

        current_df = pd.read_csv(score_file_path)
        current_df.rename(columns={'score': model_name}, inplace=True)

        combined_df = pd.merge(combined_df, current_df[['id', 'image_id', model_name]], on=['id', 'image_id'], how='outer')

    metadata_df = pd.read_csv(metadata_file)
    metadata_df['file_name'] = df2['file_name'].apply(lambda x: x.split('/')[1] if '/' in x else None)
    metadata_df['file_name'] = df2['file_name'].str.replace('-', '.')
    metadata_df['rank'] = metadata_df['rank'].apply(lambda x: int(''.join(filter(str.isdigit, str(x))), 10) if isinstance(x, str) else x)

    merged_df = pd.merge(combined_df, df2[['file_name', 'rank']], left_on='image_id', right_on='file_name', how='left')

    column_order = ['id', 'image_id', 'fuyu_dsg_score', 'fuyu_tifa_score', 'instruct_blip_dsg_score',
                    'instruct_blip_tifa_score', 'llava-alt_dsg_score', 'llava-alt_tifa_score', 'llava_dsg_score',
                    'llava_tifa_score', 'mplug_dsg_score', 'mplug_tifa_score', 'rank']

    merged_df = merged_df[column_order]

    merged_df.to_csv(output_file, index=False)

