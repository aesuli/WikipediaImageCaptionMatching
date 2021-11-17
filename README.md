# WIkipedia Image Caption Matching

## Setup

Download train and test files (`train-00000-of-00005.tsv`, ..., `test.tsv`, `test_caption_list.csv`)
from [here](https://www.kaggle.com/c/wikipedia-image-caption/data) and put them in the `data` directory.

Create a conda env using python 3.9

````
conda create -n WICM python=3.9
conda activate WICM
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers
pip install simpletransformers
pip install levenshtein
````

## Methods

### Levenshtein

[test_levenshtein.py](test_levenshtein.py) Trivial baseline that uses levenshtein distance to
compute similarity between filenames and captions.

### BERT Language model similarity

Trains a (multilingual) language model and uses it to compute pairwise similarities.

[create_bert_lm_train_data.py](create_bert_lm_train_data.py) creates a training set to train a bert masked language
model. Every line is the concatenation of the filename (converted into a human readable form) with its own caption.

[train_bert_lm.py](train_bert_lm.py) trains a bert masked language model. It takes about 20 hours to train one epoch on
an RTX2080.

*TODO* try a larger batch size value than the default one (8)

[test_bert_lm_cosine_sim.py](test_bert_lm_cosine_sim.py) this test uses the bert masked language model to embed
filenames and captions and then computes the cosine similarity among all pairs, keeping the top 5 most similar captions
for every filename

### ROBERTA Sentence pair classifier

[create_roberta_classifier_train_data.py](create_roberta_classifier_train_data.py) creates training data for the
sentence pair classifier. Matching pairs come from original training data. Non-matching pairs are generated by negative
sampling, i.e., random filenames and captions are put together.

[train_roberta_classifier.py](train_roberta_classifier.py) trains a sentence pair classifier based on xlm-roberta.
It take about 65 hours to train one epoch on an RTX2080.

[test_roberta_classifier.py](test_roberta_classifier.py) classifies all the pairs of filenames and captions as either
matching or non-matching. Classification scores are used to keep the top 5 most matching captions.
**BIG ISSUE: it would take more than a month to classify all pairs**

## Results

Kaggle leaderboard score:

| leaderboard | method| date  | score | notes | 
|---|---|---|---|---|
| public | Levenshtein |  2021-11-16 | 0.21426 | |
| public | BERT-LM | 2021-11-16 | 0.11399 | train samples are the concatenation of matching filename and caption, trained one epoch | 
| public | ROBERTA-classifier |  |  |  | 
