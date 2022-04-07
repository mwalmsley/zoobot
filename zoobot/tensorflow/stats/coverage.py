import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


def get_hpd(x: np.ndarray, p: np.ndarray, ci=0.8):
    """
    Get highest posterior density interval containing roughly "ci" specified total prob.


    Args:
        x (np.ndarray): Values of random variable e.g. votes [0, 1, ..., N]
        p (np.ndarray): Discrete prob. of those values. Normalised to 1.
        ci (float, optional): [description]. Defaults to 0.8.

    Returns:
        (Any) Value of x at low edge of interval
        (Any) Value of x at high edge of interval
        (float) Total probability between edges (inclusive)
        (bool) True if posterior is unimodal 
    """

    if len(p) <= 1:
        print(x, p)
        raise IndexError
    assert x.ndim == 1
    assert x.shape == p.shape
    assert np.isclose(p.sum(), 1, atol=0.001)
    # here, x is discrete posterior p's, not samples as with agnfinder
    mode_index = np.argmax(p)
    # check unimodal
    unimodal = True
    if not np.argsort(p)[::-1][1] in (mode_index-1, mode_index+1):
        logging.warning(f'Possible second mode, hpd will fail: {p}')
        unimodal = False
    lower_index = mode_index
    higher_index = mode_index
    while True:
        confidence = p[lower_index:higher_index+1].sum()
        if confidence >= ci:
            break  # discrete so will be at least a little over
        else:  # step each boundary outwards towards the edge, stop at the edge
            lower_index = max(0, lower_index-1)
            higher_index = min(len(x)-1, higher_index+1)

    # these indices will give a symmetric interval of at least ci, but not exactly - and will generally overestimate
    # hence confidence will generally be a bit different to desired ci, important to return
    return (x[lower_index], x[higher_index]), confidence, unimodal
        

def get_coverage(posteriors, true_values):
    results = []
    for ci_width in np.linspace(0.1, 0.95):  # 50, by default
        for target_n, (x, posterior) in enumerate(posteriors):  # n posteriors
            true_value = true_values[target_n]
            (lower_lim, higher_lim), confidence, unimodal = get_hpd(x, posterior, ci=ci_width)
            within_any_ci = lower_lim <= true_value <= higher_lim  # inclusive
            results.append({
                'target_index': target_n,
                'requested_ci_width_dont_use': ci_width, # requested confidence
                'confidence': confidence,  # actual confidence, use this
                'lower_edge': lower_lim,
                'upper_edge': higher_lim,
                'true_value': true_value,
                'true_within_hpd': within_any_ci,
                'unimodal': unimodal
            })
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=['target_index', 'confidence'])
    return df


def get_true_values(catalog, id_strs, answer):
    true_values = []
    for id_str in id_strs:
        galaxy = catalog[catalog['id_str'] == id_str].squeeze()
        true_values.append(galaxy[answer.text])
    return true_values


def get_posteriors(samples, catalog, id_strs, question, answer, temperature=None):
    """
    Samples and catalog are aligned - so let's just use them as one dataframe instead of two args


    Args:
        samples (np.ndarray): Corresponding prediction for galaxy with ``id_str``
        catalog (pd.DataFrame): Used for total votes.
        id_strs (list): [description]
        question (schemas.Question): [description]
        answer (schemas.Answer): [description]
        temperature (int, optional): Optional annealing of posteriors. Not recommended. None by default.

    Returns:
        list: posteriors like [yes_votes_arr, p_of_each] for each galaxy
    """
    all_galaxy_posteriors = []
    for sample_n, sample in enumerate(samples):
        galaxy = catalog[catalog['id_str'] == id_strs[sample_n]].squeeze()
        galaxy_posteriors = get_galaxy_posteriors(sample, galaxy, question, answer)
        all_galaxy_posteriors.append(galaxy_posteriors)
    if temperature is not None:
        all_galaxy_posteriors = [(indices, (posterior ** temperature) / np.sum(posterior ** temperature, axis=1, keepdims=True)) for (indices, posterior) in all_galaxy_posteriors]
    return all_galaxy_posteriors


def get_galaxy_posteriors(sample, galaxy, question, answer):
    assert answer in question.answers
    n_samples = sample.shape[-1]
    cols = [a.text for a in question.answers]
    assert len(cols) == 2 # Binary only!
    total_votes = galaxy[cols].sum().astype(np.float32)

    votes = np.arange(0., total_votes+1)
    x = np.stack([votes, total_votes-votes], axis=-1)  # also need the counts for other answer, no. 
    votes_this_answer = x[:, answer.index - question.start_index]  # second index is 0 or 1
    
    # could refactor with new equal mixture class
    all_probs = []
    for d in range(n_samples):
        concentrations = tf.constant(sample[question.start_index:question.end_index+1, d].astype(np.float32))
        probs = tfp.distributions.DirichletMultinomial(total_votes, concentrations).prob(x)
        all_probs.append(probs)
        
    return votes_this_answer, np.array(all_probs)

