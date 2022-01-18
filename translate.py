from time import sleep

import pandas as pd
import requests
from googletrans import Translator
from tqdm.auto import tqdm

from train_roberta_classifier import url_to_filename

translator = Translator(raise_exception=True)

def trans0(s):
    return translator.translate(s).text

def trans1(s, l='en'):
	return requests.get(f"https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl=auto&tl={l}&q={s}").json()[0][0][0]

if __name__ == '__main__':

    data_source = 'test'
    # data_source = 'validation'

    step = 0

    sleep_time = 0.2

    translate = trans0

    done = 0

    do_cap = True
    do_files = True

    if do_cap:
        df = pd.read_csv(f'data/{data_source}_caption_list.csv')
        captions = df['caption_title_and_reference_description']

        tran_cap_dict = dict()

        if step >= 0:
            tran_cap_df = pd.read_csv(f'data/{data_source}_caption_list_en_tran_{step}.csv', encoding='utf-8')
            for row in tran_cap_df.iterrows():
                id, tran = row[1]
                tran_cap_dict[int(id)] = tran

        en_captions = list()
        for id, caption in tqdm(enumerate(captions), total=len(captions), miniters=1):
            try:
                tran = tran_cap_dict.get(id, None)
                if tran:
                    en_captions.append(tran)
                else:
                    en_captions.append(translate(caption))
                    sleep(sleep_time)
                    done += 1
            except Exception as e:
                print('CAPTION BEGIN',caption,'CAPTION END', sep='\n')
                print(e)
                break

        df = pd.DataFrame(en_captions, columns=['caption_title_and_reference_description'])
        df.to_csv(f'data/{data_source}_caption_list_en_tran_{step + 1}.csv', encoding='utf-8')

    if do_files:
        df = pd.read_csv(f'data/{data_source}.tsv', sep='\t')
        filenames = [url_to_filename(url) for url in df['image_url']]

        tran_file_dict = dict()

        if step >= 0:
            tran_file_df = pd.read_csv(f'data/{data_source}_en_tran_{step}.tsv', sep='\t', encoding='utf-8')
            for row in tran_file_df.iterrows():
                id, tran = row[1]
                tran_file_dict[int(id)] = tran

        en_filenames = list()
        for id, filename in tqdm(enumerate(filenames), total=len(filenames), miniters=1):
            try:
                tran = tran_file_dict.get(id, None)
                if tran:
                    en_filenames.append(tran)
                else:
                    en_filenames.append(translate(filename))
                    sleep(sleep_time)
                    done += 1
            except Exception as e:
                print('FILENAME BEGIN',filename,'FILENAME END', sep='\n')
                print(e)
                break

        df = pd.DataFrame(en_filenames, columns=['image_url'])
        df.to_csv(f'data/{data_source}_en_tran_{step + 1}.tsv', sep='\t', encoding='utf-8')

    print(f'Done {done}')