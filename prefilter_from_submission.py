import os
import sys
from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)

    submission_filename = sys.argv[1]

    if submission_filename.find('test') >= 0:
        data_source = 'test'
    else:
        data_source = 'validation'

    submission_name = Path(submission_filename).name
    submission_name = submission_name[:submission_name.find(f'_{data_source}')]

    print('Data source:', data_source)
    print('Submission:', submission_name)

    submission_df = pd.read_csv(submission_filename)

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = list(df['caption_title_and_reference_description'])

    prefilter = list()

    prev_id = -1
    for row in submission_df.iterrows():
        id = row[1][0]
        if id != prev_id:
            prev_id = id
            prefilter.append(list())

        caption = row[1][1]

        caption_id = captions.index(caption)

        if caption_id not in prefilter[-1]:
            prefilter[-1].append(caption_id)

    with open(f'output/prefiltered_{Path(submission_filename).name}', mode='wt', encoding='utf-8') as outputfile:
        for idlist in prefilter:
            print(','.join([str(id) for id in idlist]), file=outputfile)
