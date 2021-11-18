import random

import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':
    count = len(pd.read_csv('data/test.tsv', sep='\t'))
    print(count)

    k = 10000

    with open(f'data/sorted_candidates.csv', mode='tw', encoding='utf-8') as output_file:
        population = list(range(count))
        for _ in tqdm(range(count)):
            candidates = random.choices(population,k=k)
            print(*candidates, sep=',', file=output_file)
