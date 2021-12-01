import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # classifies all the pairs of filenames and captions as either matching or non-matching
    # classification scores are used to keep the top 5 most matching captions
    os.makedirs('output', exist_ok=True)
    os.makedirs('scores', exist_ok=True)

    prefilter_filename = sys.argv[1]

    if prefilter_filename.find('test') >= 0:
        data_source = 'test'
    else:
        data_source = 'validation'

    prefilter_name = Path(prefilter_filename).name
    prefilter_name = prefilter_name[:prefilter_name.find(f'_{data_source}')]

    # partial_result_fileprefix = None
    partial_result_fileprefix = sys.argv[2]

    partial_result_fileprefix = partial_result_fileprefix[:partial_result_fileprefix.rfind('_')]

    top_idxs = list()
    top_scores = list()
    if partial_result_fileprefix:
        df = pd.read_csv(partial_result_fileprefix + '_idxs.csv')
        for row in df.iterrows():
            top_idxs.append(list(row[1]))
        df = pd.read_csv(partial_result_fileprefix + '_scores.csv')
        for row in df.iterrows():
            top_scores.append(list(row[1]))

    print('Data source:', data_source)
    print('Prefilter:', prefilter_name)
    print('Precomputed:', partial_result_fileprefix)

    if os.name == 'nt':
        use_multiprocessing_for_evaluation = False
    else:
        use_multiprocessing_for_evaluation = True

    model_args = ClassificationArgs(eval_batch_size=1024,
                                    use_multiprocessing_for_evaluation=use_multiprocessing_for_evaluation)
    if data_source == 'validation':
        model = ClassificationModel('auto', 'roberta_classifier/checkpoint-900000', args=model_args)
    else:
        model = ClassificationModel('auto', 'roberta_classifier/checkpoint-934867-epoch-1', args=model_args)
    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    k_start = None
    k_end = None
    # k_start = 0
    # k_end = 1000
    all_candidates = list()
    with open(prefilter_filename, mode='tr', encoding='utf-8') as input_file:
        for line in input_file:
            all_candidates.append(sorted([int(token) for token in line.split(',')[k_start:k_end]]))

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    to_match = 5

    for idx, (filename, candidates) in tqdm(enumerate(zip(filenames, all_candidates)), total=len(filenames)):
        pairs = [[filename, caption] for caption in captions[candidates]]
        predictions, raw_outputs = model.predict(pairs)
        top = np.argsort(raw_outputs[:, 1])[-to_match:]
        local_top_scores = list()
        local_top_idxs = list()
        for top_idx in reversed(top):
            local_top_scores.append(raw_outputs[top_idx, 1])
            local_top_idxs.append(candidates[top_idx])
        if len(top_idxs) == len(filenames):
            duplicate = list()
            for pos, local_idx in enumerate(local_top_idxs):
                if local_idx in top_idxs[idx]:
                    duplicate.append(pos)
            for pos in reversed(duplicate):
                local_top_idxs.pop(pos)
                local_top_scores.pop(pos)
            group_idxs = top_idxs[idx] + local_top_idxs
            group_scores = top_scores[idx] + local_top_scores
            group_best = list(reversed(np.argsort(group_scores)[-to_match:]))
            top_idxs[idx] = [group_idxs[best] for best in group_best]
            top_scores[idx] = [group_scores[best] for best in group_best]
        else:
            top_scores.append(local_top_scores)
            top_idxs.append(local_top_idxs)

    results = list()
    for idx, row in enumerate(top_idxs):
        for top_idx in row:
            results.append((idx, captions[top_idx]))

    time_str = f'{datetime.datetime.now():%Y-%m-%d-%H-%M}'

    output_filename = f'scores/roberta_classifier_{prefilter_name}_{data_source}_{time_str}_scores.csv'
    df = pd.DataFrame(top_scores)
    df.to_csv(output_filename, index=False)

    output_filename = f'scores/roberta_classifier_{prefilter_name}_{data_source}_{time_str}_idxs.csv'
    df = pd.DataFrame(top_idxs)
    df.to_csv(output_filename, index=False)

    output_filename = f'output/roberta_classifier_{prefilter_name}_{data_source}_{time_str}.csv'
    df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
    df.to_csv(output_filename, index=False)
