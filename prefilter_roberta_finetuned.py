import datetime
import os

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs, LanguageModelingArgs, ModelArgs
from simpletransformers.language_modeling import LanguageModelingModel
from simpletransformers.language_representation import RepresentationModel
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

if __name__ == '__main__':
    # classifies all the pairs of filenames and captions as either matching or non-matching
    # classification scores are used to keep the top 5 most matching captions
    # data_source = 'test'
    data_source = 'test'

    if os.name == 'nt':
        use_multiprocessing_for_evaluation = False
    else:
        use_multiprocessing_for_evaluation = True

    model_args = ModelArgs(eval_batch_size=1024,
                           use_multiprocessing_for_evaluation=use_multiprocessing_for_evaluation)
    if data_source == 'validation':
        model = RepresentationModel('xlmroberta', 'roberta_classifier/checkpoint-900000', args=model_args)
    else:
        model = RepresentationModel('xlmroberta', 'roberta_classifier/checkpoint-934867-epoch-1', args=model_args)
    df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
    filenames = [url_to_filename(url) for url in df['image_url']]

    df = pd.read_csv(f'data/{data_source}_caption_list.csv')
    captions = df['caption_title_and_reference_description']

    to_match = 5

    prefilter = 0
    # prefilter = 200
    # prefilter = 1000

    if prefilter:
        to_match = prefilter

    encoded_filenames = list()
    for f in tqdm(filenames):
        encoded_filenames.append(model.encode_sentences([f], combine_strategy='mean')[0])
    encoded_filenames = np.stack(encoded_filenames, axis=0)

    encoded_captions = list()
    for f in tqdm(captions):
        encoded_captions.append(model.encode_sentences([f], combine_strategy='mean')[0])
    encoded_captions = np.stack(encoded_captions, axis=0)

    print(encoded_filenames.shape, encoded_captions.shape)

    # code from sklearn's cosine similarity function has been copied here to avoid
    # doing normalization of the same vectors on every call
    encoded_filenames, encoded_captions = check_pairwise_arrays(encoded_filenames, encoded_captions)
    encoded_filenames_normalized = normalize(encoded_filenames, copy=True)
    encoded_captions_normalized = normalize(encoded_captions, copy=True)

    # must compute cosine similarities in batches, keeping track of the top 5 most similar results
    # otherwise it would require to much memory
    batch_size = 100
    top_values = None
    top_idxs = None
    for idx in tqdm(range(0, len(encoded_captions), batch_size)):
        # cosine similarity of all filenames against a batch of captions
        batch_sims = safe_sparse_dot(encoded_filenames_normalized,
                                     encoded_captions_normalized[idx:idx + batch_size].T,
                                     dense_output=True)
        if top_values is not None:
            # updating the top 5 most similar captions
            batch_idxs = np.asarray([list(range(batch_sims.shape[1]))] * batch_sims.shape[0]) + idx
            sims_pool = np.hstack((top_values, batch_sims))
            idxs_pool = np.hstack((top_idxs, batch_idxs))
            top = np.argpartition(sims_pool, max(-to_match, -sims_pool.shape[1]))[:, -to_match:]
            top_values = np.take_along_axis(sims_pool, top, axis=1)
            top_idxs = np.take_along_axis(idxs_pool, top, axis=1)
        else:
            top = np.argpartition(batch_sims, max(-to_match, -batch_sims.shape[1]))[:, -to_match:]
            top_values = np.take_along_axis(batch_sims, top, axis=1)
            top_idxs = top + idx

    if prefilter:
        with open(
                f'output/prefilter_roberta_lm_cosine_similarity_{data_source}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv',
                mode='wt', encoding='utf-8') as output_file:
            for row in top_idxs:
                print(','.join((str(idx) for idx in row[::-1])), file=output_file)
    else:
        results = list()
        order = np.argsort(top_values, axis=1)
        for row_idx, idxs in enumerate(order):
            for idx in idxs[::-1]:
                print(row_idx, idx, top_idxs[row_idx, idx], top_values[row_idx, idx], filenames[row_idx], '|',
                      captions[top_idxs[row_idx, idx]])
                results.append((row_idx, captions[top_idxs[row_idx, idx]]))

        output_filename = f'output/roberta_lm_cosine_similarity_{data_source}_{datetime.datetime.now():%Y-%m-%d-%H-%M}.csv'
        df = pd.DataFrame(results, columns=['id', 'caption_title_and_reference_description'])
        df.to_csv(output_filename, index=False)
