import datetime
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
from Levenshtein import ratio
from tqdm.auto import tqdm

from train_transformer import url_to_filename

if __name__ == '__main__':
    # trivial baseline: matches filenames and caption by using the levenshtein distance
    df = pd.read_csv('data/test.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    df = pd.read_csv('data/test_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    with multiprocessing.Pool() as pool:
        top = 5
        results = list()
        for idx in tqdm(range(len(filenames))):
            filename = filenames[idx]
            partial_ratio = partial(ratio, filename)
            sims = pool.map(partial_ratio, [caption.strip('"') for caption in captions])
            top_idxs = np.argsort(sims)[-top:]
            for top_idx in top_idxs[::-1]:
                results.append((idx, captions[top_idx]))

    df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
    df.to_csv(f'output/levenshtein_submission_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv', index=False)
