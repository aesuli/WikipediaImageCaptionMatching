from datetime import datetime

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # classifies all the pairs of filenames and captions as either matching or non-matching
    # classification scores are used to keep the top 5 most matching captions
    model_args = ClassificationArgs(eval_batch_size=512)
    model = ClassificationModel('auto', 'roberta_classifier/checkpoint-934867-epoch-1', args=model_args)
    df = pd.read_csv('data/test.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    k = 100
    all_candidates = list()
    with open('data/sorted_candidates.csv', mode='tr', encoding='utf-8') as input_file:
        for line in input_file:
            all_candidates.append(int(token) for token in line.split(',')[:k])

    df = pd.read_csv('data/test_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    to_match = 5

    results = list()
    for idx, (filename, candidates) in enumerate(zip(filenames, all_candidates)):
        predictions, raw_outputs = model.predict([[filename, caption] for caption in captions[candidates]])
        top = np.argsort(raw_outputs[:, 1])[-to_match:]
        for top_idx in reversed(top):
            print(f'{idx} | {filename} | {captions[candidates[top_idx]]}', raw_outputs[top_idx, 1])
        results.append((idx, captions[candidates[top_idx]]))

    output_filename = f'output/roberta_classifier_prefiltered_submission_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv'
    df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
    df.to_csv(output_filename, index=False)
