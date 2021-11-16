import logging
from multiprocessing import freeze_support
from urllib.parse import unquote

import pandas as pd
from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)


def url_to_filename(url):
    return unquote(url[url.rfind('/') + 1:url.rfind('.')]).replace('_', ' ')


if __name__ == '__main__':
    # trains a sentence pair classifier based on xlm-roberta
    # it classifies pairs of filename and caption
    # as either matching or non-matching

    freeze_support()
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    batch_size = 64
    model_args = ClassificationArgs(num_train_epochs=2,
                                    output_dir='roberta_classifier',
                                    overwrite_output_dir=True,
                                    save_steps=100000,
                                    save_model_every_epoch=True,
                                    train_batch_size=batch_size)

    # model = 'xlm-roberta-large'
    model_name = 'xlm-roberta-base'

    model = ClassificationModel("auto", model_name, args=model_args)

    data = pd.read_csv('data/train_roberta_classifier.tsv', sep='\t', quotechar='"')

    model.train_model(data)
