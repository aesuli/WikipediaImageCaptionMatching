import datetime
from functools import partial

import numpy as np
import pandas as pd
from Levenshtein import ratio
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # trivial baseline: matches filenames and caption by using the levenshtein distance

    data_source = 'test'
    # data_source = 'validation'

    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    # prefilter = 0
    # prefilter = 200
    prefilter = 1000

    if prefilter:
        to_match = prefilter

    results = list()
    for idx in tqdm(range(len(filenames))):
        filename = filenames[idx]
        partial_ratio = partial(ratio, filename)
        sims = list(map(partial_ratio, [caption.strip('"') for caption in captions]))
        top_idxs = np.argsort(sims)[-to_match:]
        if prefilter:
            results.append(top_idxs)
        else:
            for top_idx in top_idxs[::-1]:
                results.append((idx, captions[top_idx]))

    if prefilter:
        with open(
                f'output/prefilter_levenshtein_{data_source}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv',
                mode='wt', encoding='utf-8') as output_file:
            for row in results:
                print(','.join((str(idx) for idx in row)), file=output_file)
    else:
        df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
        df.to_csv(f'output/levenshtein_{data_source}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv', index=False)
