from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/test_caption_list.csv')

    captions = list(df['caption_title_and_reference_description'])

    counts = Counter(captions)

    freqs = list(counts.values())

    freq_mean = np.mean(freqs)
    freq_max = np.max(freqs)

    total = len(captions)
    distinct_count = len(counts)

    meta_counts = Counter(freqs)


    print(total,distinct_count,total-distinct_count,freq_max,freq_mean)

    pprint(counts.most_common(10))

    pprint(sorted(meta_counts.items()))
