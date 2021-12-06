import os
import sys
from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)

    idxs_filename = sys.argv[1]

    idxs_name = Path(idxs_filename).name
    if idxs_name.find('test') >= 0:
        data_source = 'test'
    else:
        data_source = 'validation'

    print('Data source:', data_source)
    print('Idxs:', idxs_name)

    idxs = pd.read_csv(idxs_filename).values

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = list(df['caption_title_and_reference_description'])

    to_match = 5
    results = list()

    for id, row in enumerate(idxs):
        for caption_id in row[:to_match]:
            results.append((id, captions[caption_id]))

    output_filename = f'output/{idxs_name.replace("_idxs", "")}'
    df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
    df.to_csv(output_filename, index=False)
