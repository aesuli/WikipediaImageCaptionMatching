import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    i = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, i


if __name__ == '__main__':
    data_dir = Path(sys.argv[1])

    if data_dir.name.find('test') >= 0:
        dataset = 'test'
    else:
        dataset = 'validation'

    print(f'Dataset: {dataset}')

    urls = pd.read_csv(f'data/{dataset}.tsv', sep='\t')['image_url']
    captions = pd.read_csv(f'data/{dataset}_caption_list.csv')['caption_title_and_reference_description']

    k = 10

    evaluated_submissions = defaultdict(list)
    for file in data_dir.iterdir():
        if not file.is_file() or not file.name.endswith('.csv'):
            continue

        submission = pd.read_csv(file)
        submission_name = file.name[:-4]

        last_id = -1
        position = 0
        for row in submission.iterrows():
            id = row[1]['id']
            caption = row[1]['caption_title_and_reference_description']
            if last_id != id:
                position = 1
            else:
                position += 1
            if position <= k:
                rel = 'N'
                if captions.iloc[id] == caption:
                    rel = 'R'
                evaluated_submissions[submission_name].append(
                    (id, position, rel, submission_name, caption))
            last_id = id

    last_id = -1
    with open('comparison.txt', mode='wt', encoding='utf-8') as outfile:
        for values in zip(*evaluated_submissions.values()):
            last_position = -1
            for row in values:
                id = row[0]
                position = row[1]
                if id != last_id:
                    print(file=outfile)
                    print(f'Query: {id}', urls.iloc[id], file=outfile)
                    last_id = id
                if position != last_position:
                    print(f' Rank: {position}',file=outfile)
                    last_position = position
                print('  '+'\t'.join([str(x) for x in row][2:]), file=outfile)
