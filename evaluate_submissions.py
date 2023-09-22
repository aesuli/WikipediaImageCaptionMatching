import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from train_roberta_classifier import url_to_filename


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

    df = pd.read_csv(f'data/{dataset}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    for file in data_dir.iterdir():
        if not file.is_file() or not file.name.endswith('.csv'):
            continue

        submission = pd.read_csv(file)

        ks = (1, 5, 10)

        for k in ks:
            rels = list()
            last_id = -1
            den_count = 0
            total_rel = 0
            found = 0
            for row in submission.iterrows():
                id = row[1]['id']
                if last_id != id:
                    if last_id != -1:
                        total_rel += rel
                        rels.append(rel)
                        den_count += 1
                    id_count = 1
                    rel = 0
                else:
                    id_count += 1
                if captions.iloc[id] == row[1]['caption_title_and_reference_description'] and id_count <= k:
                    found += 1
                    rel += 1 / np.log2(id_count + 1)

                last_id = id

            rels.append(rel)
            total_rel += rel
            den_count += 1

            print(f'{file.name} \tR@{k} {(found / den_count*100):#.3g}', end ='\t')
            print(f'nDCG@{k} = {(total_rel / den_count):#.3g}', end='\t')
            for confidence in [.9, .95, .99]:
                m,i = mean_confidence_interval(rels, confidence=confidence)
                print(f'CI({int(confidence*100)}%) = ±{i:#.3g}', end='  ')
            print()
