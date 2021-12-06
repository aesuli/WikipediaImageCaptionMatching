import datetime
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':
    by_idx_dir = Path(sys.argv[1])

    by_idx_name = by_idx_dir.name

    if by_idx_name.find('test') >= 0:
        data_source = 'test'
    else:
        data_source = 'validation'

    print('By idx dir:', by_idx_name)

    scores = dict()
    idxs = dict()

    id_re = re.compile('^([0-9]+).txt$')
    for file in tqdm(by_idx_dir.iterdir()):
        id = id_re.match(file.name)[1]
        with open(file, mode='rt', encoding='utf-8') as input_file:
            pairs = [pair.split(':') for line in input_file for pair in line.strip().split(',')]
        scores[id] = [score for _, score in pairs]
        idxs[id] = [idx for idx, _ in pairs]

    time_str = f'{datetime.datetime.now():%Y-%m-%d-%H-%M}'

    scores = [scores[str(id)] for id in range(len(scores))]
    idxs = [idxs[str(id)] for id in range(len(idxs))]

    output_filename = f'scores/from_{by_idx_name}_{time_str}_scores.csv'
    df = pd.DataFrame(scores)
    df.to_csv(output_filename, index=False)

    output_filename = f'scores/from_{by_idx_name}_{time_str}_idxs.csv'
    df = pd.DataFrame(idxs)
    df.to_csv(output_filename, index=False)
