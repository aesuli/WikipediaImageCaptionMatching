import json
import os
import pickle
import time
import urllib.parse
from collections import Counter
from json import JSONDecodeError
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

if __name__ == '__main__':
    dbpedia_json_dir = Path('dbpedia_json_dir')
    os.makedirs(dbpedia_json_dir, exist_ok=True)
    if not os.path.exists('data/tags.pkl'):
        data = pd.read_csv('data/wikimedia_train_en.csv')
        count = 0
        missing = 0
        tags = Counter()
        for _, fields in tqdm(data.iterrows(), total=len(data)):
            page_url = fields['page_url']
            image_url = fields['image_url']
            caption_reference_description = fields['caption_reference_description']
            if type(caption_reference_description) == str:
                page_name = page_url[page_url.rfind('/') + 1:]
                page_name = page_name.replace('"', '%2522')
                page_name = page_name.replace('%22', '%2522')
                page_name = page_name.replace('*', '%2A')
                page_name = page_name.replace(':', '%3A')
                json_path = dbpedia_json_dir / f'{page_name}.json'
                data = None
                if not os.path.exists(json_path):
                    dbpedia_query = f'https://dbpedia.org/data/{page_name}.json'
                    try:
                        data = requests.get(dbpedia_query).json()
                        with open(json_path, mode='tw', encoding='utf-8') as outputfile:
                            json.dump(data, outputfile)
                    except JSONDecodeError:
                        print('JSON Error', page_name)
                    except OSError as oserror:
                        print('OS Error', page_name)
                        print(oserror)
                    time.sleep(0.1)
                else:
                    with open(json_path, mode='tr', encoding='utf-8') as inputfile:
                        data = json.load(inputfile)

                if data:
                    page_key = f'http://dbpedia.org/resource/{urllib.parse.unquote(page_name)}'
                    page_data = None
                    try:
                        page_data = data[page_key]
                    except KeyError:
                        page_key = page_key.lower()
                        for key in data.keys():
                            if key.lower() == page_key:
                                page_data = data[key]
                                break
                    if page_data is None:
                        print(page_key, key)
                    try:
                        for entry in page_data["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]:
                            tags.update([entry['value']])
                    except KeyError:
                        missing += 1
                    count += 1

        print(count, missing, count - missing)

        with open('data/tags.pkl', mode='wb') as outputfile:
            pickle.dump(tags, outputfile)

    with open('data/tags.pkl', mode='rb') as inputfile:
        tags = pickle.load(inputfile)

    selected_tags = set()
    for key, value in sorted(tags.items(), key=lambda x: x[1], reverse=True):
        if 'dbpedia.org/ontology/' in key and value > 10:
            selected_tags.add(key)
            print(key, value)

    data = pd.read_csv('data/wikimedia_train_en.csv')
    dataset = list()
    for _, fields in tqdm(data.iterrows(), total=len(data)):
        page_url = fields['page_url']
        image_url = fields['image_url']
        caption_reference_description = fields['caption_reference_description']
        if type(caption_reference_description) == str:
            page_title = fields['page_title']
            page_name = page_url[page_url.rfind('/') + 1:]
            page_name = page_name.replace('"', '%2522')
            page_name = page_name.replace('%22', '%2522')
            page_name = page_name.replace('*', '%2A')
            page_name = page_name.replace(':', '%3A')
            json_path = dbpedia_json_dir / f'{page_name}.json'
            data = None
            if not os.path.exists(json_path):
                dbpedia_query = f'https://dbpedia.org/data/{page_name}.json'
                try:
                    data = requests.get(dbpedia_query).json()
                    with open(json_path, mode='tw', encoding='utf-8') as outputfile:
                        json.dump(data, outputfile)
                except JSONDecodeError:
                    print('JSON Error', page_name)
                except OSError as oserror:
                    print('OS Error', page_name)
                    print(oserror)
                time.sleep(0.1)
            else:
                with open(json_path, mode='tr', encoding='utf-8') as inputfile:
                    data = json.load(inputfile)

            page_tags = set()
            if data:
                page_key = f'http://dbpedia.org/resource/{urllib.parse.unquote(page_name)}'
                page_data = None
                try:
                    page_data = data[page_key]
                except KeyError:
                    page_key = page_key.lower()
                    for key in data.keys():
                        if key.lower() == page_key:
                            page_data = data[key]
                            break
                if page_data is None:
                    print(page_key, key)
                page_tags = set()
                try:
                    for entry in page_data["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"]:
                        tag = entry['value']
                        if tag in selected_tags:
                            page_tags.add(tag)
                        tags.update([entry['value']])
                except KeyError:
                    pass

            dataset.append({'image_url': image_url, 'caption': caption_reference_description, 'page_title': page_title,
                            'page_tags': list(page_tags), 'page_url': page_url})

    df = pd.DataFrame(dataset)
    df.to_csv('data/wikimedia_en_caption_dbpedia_tags.csv')
    print(len(df))
