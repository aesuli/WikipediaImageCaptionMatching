import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':
    train_data = list()
    train_filenames = [f'train-0000{i}-of-00005.tsv' for i in range(5)]

    en_df = None

    for train_filename in tqdm(train_filenames, total=5):
        data_df = pd.read_csv('data/' + train_filename, sep='\t')
        if en_df is not None:
            en_df.append( data_df[data_df['language']=='en'], ignore_index=True)
        else:
            en_df = data_df[data_df['language']=='en']

    en_df.to_csv('data/wikimedia_train_en.csv')


