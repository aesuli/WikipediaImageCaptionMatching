import random

import pandas as pd
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # creates training data for the sentence pair classifier
    # matching pairs come from original training data
    # non-matching pairs are generated by negative sampling, i.e., random filenames and
    # captions are put together
    train_data = list()
    train_filenames = [f'train-0000{i}-of-00005.tsv' for i in range(5)]

    for train_filename in tqdm(train_filenames):
        data_df = pd.read_csv('data/' + train_filename, sep='\t', usecols=[2, 17])
        filenames = [url_to_filename(url) for url in data_df['image_url']]
        captions = data_df['caption_title_and_reference_description']
        del data_df
        max_idx = len(filenames) - 1
        for idx in tqdm(range(max_idx+1)):
            idx2 = idx
            while idx2 == idx:
                idx2 = random.randint(0, max_idx)
            train_data.append((filenames[idx], captions[idx], 1))
            train_data.append((filenames[idx], captions[idx2], 0))
            idx += 1
    df = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'labels'])
    df.to_csv('data/train_roberta_classifier.tsv', sep='\t', quotechar='\"', index=False)
