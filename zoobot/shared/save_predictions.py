import json
from typing import List

import numpy as np
import pandas as pd
import h5py


def predictions_to_hdf5(predictions, id_str, label_cols, save_loc):
    assert save_loc.endswith('.hdf5')
    with h5py.File(save_loc, "w") as f:
        f.create_dataset(name='predictions', data=predictions)
        # https://docs.h5py.org/en/stable/special.html#h5py.string_dtype
        dt = h5py.string_dtype(encoding='utf-8')
        # predictions_dset.attrs['label_cols'] = label_cols  # would be more conventional but is a little awkward
        f.create_dataset(name='id_str', data=id_str, dtype=dt)
        f.create_dataset(name='label_cols', data=label_cols, dtype=dt)


def predictions_to_csv(predictions, id_str, label_cols, save_loc):
    # not recommended - hdf5 is much more flexible and pretty easy to use once you check the package quickstart
    assert save_loc.endswith('.csv')
    data = [prediction_to_row(predictions[n], id_str[n], label_cols=label_cols) for n in range(len(predictions))]
    predictions_df = pd.DataFrame(data)
    # logging.info(predictions_df)
    predictions_df.to_csv(save_loc, index=False)


def prediction_to_row(prediction: np.ndarray, id_str: str, label_cols: List):
    """
    Convert prediction on image into dict suitable for saving as csv
    Predictions are encoded as a json e.g. "[1., 0.9]" for 2 repeat predictions on one galaxy
    This makes them easy to read back with df[col] = df[col].apply(json.loads)

    Args:
        prediction (np.ndarray): model output for one galaxy, including repeat predictions e.g. [[1., 0.9], [0.3, 0.24]] for model with output_dim=2 and 2 repeat predictions
        id_str (str): path to image
        label_cols (list): semantic labels for model output dim e.g. ['smooth', 'bar'].

    Returns:
        dict: of the form {'id_str': 'path', 'smooth_pred': "[1., 0.9]", 'bar_pred: "[0.3, 0.24]"}
    
    """
    row = {
        'id_str': id_str  # may very well be a path to an image, if using an image dataset - just rename later
    }
    for n in range(len(label_cols)):
        answer = label_cols[n]
        row[answer + '_pred'] = json.dumps(list(prediction[n].astype(float)))
    return row
