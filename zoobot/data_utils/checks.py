import os

import numpy as np
from tqdm import tqdm

def check_no_missing_files(locs, max_to_check=None):
    # locs_missing = [not os.path.isfile(path) for path in tqdm(locs)]
    # if any(locs_missing):
        # raise ValueError('Missing {} files e.g. {}'.format(
        # np.sum(locs_missing), locs[locs_missing][0]))
    print('Checking no missing files')
    if max_to_check is not None:
        if len(locs) > max_to_check:
            locs = np.random.choice(locs, max_to_check)
    for loc in tqdm(locs):
        if not os.path.isfile(loc):
            raise ValueError('Missing ' + loc)
