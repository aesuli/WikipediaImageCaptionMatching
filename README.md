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
It takes about 65 hours to train one epoch on an RTX2080.

[test_roberta_classifier.py](test_roberta_classifier.py) classifies all the pairs of filenames and captions as either
matching or non-matching. Classification scores are used to keep the top 5 most matching captions.

**BIG ISSUE: it would take more than a month to classify all pairs**

**MITIGATION: use a _prefilter_ method that selects a reduced set of candidates**

### Prefiltered ROBERTA Sentence pair classifier

This is the same method as above, but it takes as input also a file with ids of candidate captions that are likely to be a correct match.

Using 1000 candidates captions for every image, it requires 27 hours to produce a submission for all the 92k+ images.

[create_dummy_sorted_candidates.py](create_dummy_sorted_candidates.py) Creates a demo file in the format required by the classification script, i.e., a row for every image with ids of candidates separated by a comma:

```
135,235,414,891,696,417,88,288,12,247,245,471,911...
995,660,581,240,804,818,418,965,961,834,179,837,4...
13,16,273,32,779,261,253,495,534,913,912,497,910,...
```
Ids in prefiltered files must be in order with highest confidence candidates first and lowest confidence candidates last.

[test_roberta_classifier_prefiltered.py](test_roberta_classifier_prefiltered.py) Same as the `test_roberta_classifier.py` above, but also loading the list of prefilters ids.

The scripts [test_bert_lm_cosine_sim.py](test_bert_lm_cosine_sim.py) and [test_levenshtein.py](test_levenshtein.py) have a `prefilter` variable added that, when set to a natural number, produces a prefilter file instead of a submission file.

### Merging prefilters

[merge_prefiltered.py](merge_prefiltered.py) merges prefiltered ids by different methods using a round-robin policy.
Ids in prefiltered files must be in order with highest confidence candidates first and lowest confidence candidates last.

## Validation

Validation scripts to evaluate the methods locally.

[create_validation_data.py](create_validation_data.py) selects the last `k` examples for the training data as a validation set.

[evaluate_submission.py](evaluate_submission.py) computes NDCG of a submission on validation data.

The scripts [test_bert_lm_cosine_sim.py](test_bert_lm_cosine_sim.py), [test_levenshtein.py](test_levenshtein.py), [test_roberta_classifier.py](test_roberta_classifier.py), [test_roberta_classifier_prefiltered.py](test_roberta_classifier_prefiltered.py), now have a `data_source` variable that sets which dataset (`validation` or `test`) to load and produce a submission for.

## Results on validation set

Validation set size = 1000

| method| date  | NDCG@5 | recall@5| notes | 
|---|---|---|---|---|
| Levenshtein |  2021-11-22 | 0.35613 | 0.386 | | 
| BERT-LM | 2021-11-22 | 0.21252 | 0.245 | train samples are the concatenation of matching filename and caption, trained one epoch | 
| ROBERTA-classifier | 2021-11-22 | 0.78883 | 0.827 | model not fit on validation data |
| ROBERTA-classifier prefilter Levenshtein | 2021-11-24 | 0.54774 | 0.565 | prefilter = 200 | 
| ROBERTA-classifier prefilter BERT-LM | 2021-11-22 | 0.51174 | 0.532 | prefilter = 200 | 

## Results on Kaggle leaderboard

Kaggle leaderboard score:

| leaderboard | method| date  | NDCG@5 | notes | 
|---|---|---|---|---|
| public | Levenshtein |  2021-11-16 | 0.21426 | |
| public | BERT-LM | 2021-11-16 | 0.11399 | train samples are the concatenation of matching filename and caption, trained one epoch | 
| public | ROBERTA-classifier |  |  | NOT ENOUGH TIME BEFORE THE DEADLINE | 
| public | ROBERTA-classifier prefilter Levenshtein | 2021-11-25 | 0.36995 | prefilter = 1000 | 
| public | ROBERTA-classifier prefilter BERT-LM | 2021-11-24 | 0.26852 | prefilter = 1000 | 