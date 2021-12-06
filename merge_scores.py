import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)

    merge_name = ''

    scores = None
    idxs = None
    for score_file in sys.argv[1:]:
        score_file_name = Path(score_file).name
        if score_file_name.find('test') >= 0:
            data_source = 'test'
        else:
            data_source = 'validation'
        merge_name += score_file_name[:score_file_name.find(f'_{data_source}')]
        scores_batch = pd.read_csv(score_file, encoding='utf-8').values
        if scores is None:
            scores = scores_batch
        else:
            scores = np.hstack((scores, scores_batch))
        idxs_batch = pd.read_csv(score_file.replace('scores.csv', 'idxs.csv'), encoding='utf-8').values
        if idxs is None:
            idxs = idxs_batch
        else:
            idxs = np.hstack((idxs, idxs_batch))

    sort = np.argsort(scores, axis=1)

    time_str = f'{datetime.datetime.now():%Y-%m-%d-%H-%M}'

    output_filename = f'scores/merge_{merge_name}_{data_source}_{time_str}_scores.csv'
    df = pd.DataFrame(np.flip(np.take_along_axis(scores,sort,axis=1),axis=1))
    df.to_csv(output_filename, index=False)

    output_filename = f'scores/merge_{merge_name}_{data_source}_{time_str}_idxs.csv'
    df = pd.DataFrame(np.flip(np.take_along_axis(idxs,sort,axis=1),axis=1))
    df.to_csv(output_filename, index=False)
