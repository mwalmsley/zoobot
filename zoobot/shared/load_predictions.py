import logging
import os
from typing import List

import numpy as np
import pandas as pd
import h5py

from zoobot.shared import stats, schemas


def load_hdf5s(hdf5_locs: List):
    """
    Load hdf5 predictions (with metadata) from disk.
    
    Each hdf5 includes the following datasets (accessed with f[dataset_name]):
        - predictions: np.array of shape (galaxy, answer, forward pass) containing model outputs (usually Dirichlet concentrations)
        - id_str: unique string (usually the file location) specifying which galaxy each prediction refers to. Indices match `predictions`
        - label_cols: semantic names for the answer dimension of `predictions` e.g. smooth-or-featured-dr8_smooth, smooth-or-featured-dr8_featured-or-disk, ...
    
    See predictions.predictions_to_hdf5 for the save function.

    hdf5 is useful because predictions are three-dimensional (e.g. 10 galaxies, 4 answers, 5 dropout passes)
    while csv's and similar are designed for two-dimensional data (e.g 10 galaxies, 4 answers).

    Args:
        hdf5_locs (list): hdf5 files to load

    Returns:
        pd.DataFrame: with rows of id_str, hdf5_loc, indexed like predictions (below)
        np.array: model predictions, usually dirichlet concentrations, like (galaxy, answer, forward pass)
        list: semantic names of answers (e.g. ['smooth-or-featured-dr8_smooth', ...])
    """

    if isinstance(hdf5_locs, str):
        logging.warning('Passed a single hdf5 loc to load_hdf5s - assuming this is a single file to load, not list of files to load')
        hdf5_locs = [hdf5_locs]  # pretend user passed a list

    predictions = []
    prediction_metadata = []
    template_label_cols = None  # will use this var to check consistency of label_cols across each hdf5_loc
    for loc in hdf5_locs:
        try:
            with h5py.File(loc, 'r') as f:

                    logging.debug(f.keys())
                    these_predictions = f['predictions'][:]
                    these_prediction_metadata = {
                        'id_str': f['id_str'].asstr()[:],
                        'hdf5_loc': [os.path.basename(loc) for _ in these_predictions]
                }
                    predictions.append(these_predictions)  # will create a list where each element is 3D predictions stored in each hdf5
                    prediction_metadata.append(these_prediction_metadata)  # also track id_str, similarly

                    if template_label_cols is None:  # first file to load, use this as the expected template
                        template_label_cols = f['label_cols'].asstr()[:]
                        logging.info('Using label columns {} from first hdf5 {}'.format(template_label_cols, loc))
                    else:
                        these_label_cols = f['label_cols'].asstr()[:]
                        if any(these_label_cols != template_label_cols):
                            raise ValueError('Label columns {} of hdf5 {} do not match first label columns {}'.format(loc, f['label_cols'], template_label_cols))
        except Exception as e:
            logging.critical('Failed to load {}'.format(loc))
            raise e


    # there is no assumption that id_str is unique, or attempt to group predictions by id_str
    # it just maps a set of hdf5 files, each with predictions, to a df of id_str and those loaded predictions (matching row-wise)
    logging.info('All hdf5 loaded, beginning concat.')
    predictions = np.concatenate(predictions, axis=0)
    prediction_metadata = {
        'id_str': [p for metadata in prediction_metadata for p in metadata['id_str']],
        'hdf5_loc': [l for metadata in prediction_metadata for l in metadata['hdf5_loc']]
    }
    assert len(prediction_metadata['id_str']) == len(predictions)

    galaxy_id_df = pd.DataFrame(data=prediction_metadata)

    return galaxy_id_df, predictions, template_label_cols


def prediction_hdf5_to_summary_parquet(hdf5_loc: str, save_loc: str, schema: schemas.Schema, debug=False):
    """
    Take .hdf5 file with galaxy ids and concentrations predicted by Zoobot
    Create data-release-style .parquet catalogs with vote fractions and uncertainties
    Saves two tables: 
        'friendly', with just vote fracs and nans where answers aren't relevent
        'advanced', with all vote fractions and 90% confidence intervals on those fractions
    
    Args:
        hdf5_loc (str): _description_
        save_loc (str): _description_
    """
    assert isinstance(hdf5_loc, str)


    # concentrations will be of (galaxy, question, model, forward_pass) after going through c_group
    # may be only one model but will still have that dimension (e.g. 1000, 39, 1, 5)
    galaxy_id_df, concentrations, _ = load_hdf5s(hdf5_loc)

    if debug:
        logging.warning('Using debug mode of 100k only')
        concentrations = concentrations[:100000]
        galaxy_id_df = galaxy_id_df[:100000]
        save_loc = save_loc.replace('.parquet', '_debug.parquet')

    label_cols = schema.label_cols
    # TODO optionally ignore all but a subset of columns, for models without finetuning
    # hdf5_label_cols = label_cols
    # valid_cols = [col for col in hdf5_label_cols if col in label_col_subset]
    # concentrations = concentrations[:, valid_cols]

    # applies to all questions at once
    # hopefully also supports 3D concentrations (galaxy/question/model/pass)
    logging.info('Concentrations: {}'.format(concentrations.shape))

    # supports (galaxy, question, distribution...) shape, no TF needed
    unmasked_fractions = stats.expected_value_of_dirichlet_mixture(concentrations, schema)

    prob_of_asked = []
    for question in schema.questions:
        # human fractions imply you'd get this many votes for this question
        # vote fracs are now over all distributions, and therefore so will be expected votes
        question_prob_of_asked = stats.get_expected_votes_ml(unmasked_fractions, question, 1, schema, round_votes=False)
        prob_of_asked.append(question_prob_of_asked)

    prob_of_asked = np.stack(prob_of_asked, axis=1)

    assert len(galaxy_id_df) == len(unmasked_fractions)
    assert len(galaxy_id_df) == len(prob_of_asked)

    # for the friendly table, will mask out to only use proportion-volunteers-asked > 0.5

    # prob of asked is by question i.e. (galaxy, question) shape, so for np.where, need to duplicate out to answers
    prob_of_asked_by_answer = []
    for question_n, question in enumerate(schema.questions):
        prob_of_asked_duplicated = np.stack([prob_of_asked[:, question_n]] * len(question.answers), axis=1)  # N repetition in new axis 1
        prob_of_asked_by_answer.append(prob_of_asked_duplicated)
    prob_of_asked_by_answer = np.concatenate(prob_of_asked_by_answer, axis=1)  # concat along that ax, now (galaxy, answer) shape

    masked_fractions = np.where(prob_of_asked_by_answer >= 0.5, unmasked_fractions, np.zeros_like(unmasked_fractions) * np.nan)

    # this doesn't do super well with large memory reqs, so chunk it up
    all_lower_edges = []
    all_upper_edges = []
    chunk_size = min(len(concentrations), 10000)
    
    num_chunks = int(len(concentrations) / chunk_size)
    logging.info('{} chunks, of typical size {}'.format(num_chunks, chunk_size))

    for concentrations_chunk in np.array_split(concentrations, num_chunks):
        lower_edges, upper_edges = stats.get_confidence_intervals(concentrations_chunk, schema, interval_width=.9, gridsize=100)  
        assert len(lower_edges) == len(concentrations_chunk)
        assert len(upper_edges) == len(concentrations_chunk)
        all_lower_edges.append(lower_edges)
        all_upper_edges.append(upper_edges)
    all_lower_edges = np.concatenate(all_lower_edges)
    all_upper_edges = np.concatenate(all_upper_edges)

    # define useful cols
    fraction_cols = [col + '_fraction' for col in label_cols]
    lower_edge_cols = [col + '_90pc-lower' for col in label_cols]
    upper_edge_cols = [col + '_90pc-upper' for col in label_cols]
    proportion_asked_cols = [col + '_proportion-asked' for col in label_cols]

    # make friendly dataframe with just masked fraction and description string
    friendly_loc = save_loc.replace('.parquet', '_friendly.parquet')
    fraction_df = pd.DataFrame(data=masked_fractions, columns=fraction_cols)
    friendly_df = pd.concat([galaxy_id_df, fraction_df], axis=1)
    friendly_df.to_parquet(friendly_loc, index=False)
    logging.info('Friendly summary table saved to {}'.format(friendly_loc))

    # make advanced dataframe with unmasked fractions, error bars, proportion_asked
    advanced_loc = save_loc.replace('.parquet', '_advanced.parquet')
    fraction_df = pd.DataFrame(data=unmasked_fractions, columns=fraction_cols)
    lower_edge_df = pd.DataFrame(data=all_lower_edges, columns=lower_edge_cols)
    upper_edge_df = pd.DataFrame(data=all_upper_edges, columns=upper_edge_cols)
    proportion_df = pd.DataFrame(data=prob_of_asked_by_answer, columns=proportion_asked_cols)
    advanced_df = pd.concat([galaxy_id_df, fraction_df, lower_edge_df, upper_edge_df, proportion_df], axis=1)
    advanced_df.to_parquet(advanced_loc, index=False)
    logging.info('Advanced summary table saved to {}'.format(advanced_loc))


def single_forward_pass_hdf5s_to_df(hdf5_locs: List, drop_extra_dims=False):
    """
    Load predictions (or representations) saved as hdf5 into pd.DataFrame with id_str and label_cols columns

    Best used when not using test-time dropout i.e. when you have a single forward pass per galaxy. 
    Otherwise, with test-time dropout on, aanswer columns (keyed by each value in label_cols) will be
    `np.ndarray` of length (n forward passes) for each galaxy and answer, *not* 1D scalars.
    This is quite awkward to work with. I suggest using `load_hdf5s` directly when using test-time dropout

    Args:
        hdf5_locs (List): _description_

    Returns:
        _type_: _description_
    """
    galaxy_id_df, predictions, label_cols = load_hdf5s(hdf5_locs)

    predictions = predictions.squeeze()
    
    if len(predictions.shape) > 2:
        if drop_extra_dims:
            predictions = predictions[:, :, 0]
            logging.warning('Dropped extra dimensions')
        else:
            logging.critical(
                'Predictions are of shape {}, greater than rank 2. \
                I suggest using load_hdf5s directly to work with np.arrays, not with DataFrame - see docstring'
            )
    prediction_df = pd.DataFrame(data=predictions, columns=label_cols)
    # copy over metadata (indices will align)
    prediction_df['id_str'] = galaxy_id_df['id_str']
    prediction_df['hdf5_loc'] = galaxy_id_df['hdf5_loc']
    return prediction_df


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # simple tests

    # load_hdf5s('/nvme1/scratch/walml/repos/understanding_galaxies/scaled_image_predictions.hdf5')
    df = single_forward_pass_hdf5s_to_df('/nvme1/scratch/walml/repos/understanding_galaxies/scaled_image_predictions.hdf5')
    print(df)
