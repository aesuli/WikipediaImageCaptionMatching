import datetime
import sys

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # classifies all the pairs of filenames and captions as either matching or non-matching
    # classification scores are used to keep the top 5 most matching captions
    data_source = 'test'
    # data_source = 'validation'

    prefilter_filename = sys.argv[1]

    model_args = ClassificationArgs(eval_batch_size=1024, use_multiprocessing_for_evaluation=False)
    if data_source == 'validation':
        model = ClassificationModel('auto', 'roberta_classifier/checkpoint-900000', args=model_args)
    else:
        model = ClassificationModel('auto', 'roberta_classifier/checkpoint-934867-epoch-1', args=model_args)
    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    k = None
    # k = 1000
    all_candidates = list()
    with open(prefilter_filename, mode='tr', encoding='utf-8') as input_file:
        for line in input_file:
            all_candidates.append([int(token) for token in line.split(',')[:k]])

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    to_match = 5

    results = list()
    for idx, (filename, candidates) in enumerate(zip(filenames, all_candidates)):
        pairs = [[filename, caption] for caption in captions[candidates]]
        predictions, raw_outputs = model.predict(pairs)
        top = np.argsort(raw_outputs[:, 1])[-to_match:]
        for top_idx in reversed(top):
            print(f'{datetime.datetime.now():%Y-%m-%d-%H-%M-%S} | {idx} | {filename} | {captions[candidates[top_idx]]}',
                  raw_outputs[top_idx, 1])
            results.append((idx, captions[candidates[top_idx]]))

    output_filename = f'output/roberta_classifier_prefiltered_{data_source}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv'
    df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
    df.to_csv(output_filename, index=False)
