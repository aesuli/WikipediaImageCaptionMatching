import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)

    idxs_filename = sys.argv[1]

    idxs_name = Path(idxs_filename).name
    if idxs_name.find('test') >= 0:
        data_source = 'test'
    else:
        data_source = 'validation'

    prefilter_filename = sys.argv[2]

    prefilter_name = Path(prefilter_filename).name
    prefilter_name = prefilter_name[:prefilter_name.find(f'_{data_source}')]

    to_select = 750

    selection = list()
    with open(prefilter_filename, mode='tr', encoding='utf-8') as input_file:
        for line in tqdm(input_file):
            selection.append(sorted([int(token) for token in line.split(',')[:to_select]]))
    print('Data source:', data_source)
    print('Idxs:', idxs_name)

    idxs = pd.read_csv(idxs_filename).values

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = list(df['caption_title_and_reference_description'])

    to_match = 5
    results = list()

    for id, row in tqdm(enumerate(idxs)):
        added = 0
        for caption_id in row:
            if caption_id in selection[id]:
                results.append((id, captions[caption_id]))
                added += 1
                if added == to_match:
                    break

    output_filename = f'output/{prefilter_name}_{to_select}_{idxs_name.replace("_idxs", "")}'
    df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
    df.to_csv(output_filename, index=False)
