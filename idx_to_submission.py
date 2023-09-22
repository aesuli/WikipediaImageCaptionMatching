import sys
from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    data_file = Path(sys.argv[1])

    max_count = 100
    if len(sys.argv) > 2:
        max_count = int(sys.argv[2])

    if data_file.name.find('test') >= 0:
        dataset = 'test'
    else:
        dataset = 'validation'

    print(f'Dataset: {dataset}')

    df = pd.read_csv(f'data/{dataset}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    with (open(data_file, mode='rt', encoding='utf-8') as infile,
          open(data_file.name + '_caps.csv', mode='wt', encoding='utf-8') as outfile):
        df = pd.DataFrame(columns=['id', 'caption_title_and_reference_description'])
        for qid, line in enumerate(infile):
            ids = line.split(',')
            qdf = pd.DataFrame(captions.iloc[[int(id) for i,id in enumerate(ids) if i<max_count]],
                               columns=['id', 'caption_title_and_reference_description'])
            qdf['id'] = qid
            df = pd.concat([df, qdf])
        df.to_csv(outfile, index=False, lineterminator='\n')
