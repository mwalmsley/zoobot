from typing import List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp

# continuous posteriors, looks much prettier but technically not defined for non-integer votes

def show_binary_predictions(samples_list: List, observed_votes: np.ndarray, image_paths: List, xlabel: str, answer_index=0, n_examples=5):
    """
    Visualise dirichlet distribution posteriors against images and true observed_votes.
    Shows each model (element in sample_list) as solid colour
    Shows each dropout pass from that model as trace in that colour
    Shows observed observed_votes in black dashed.

    Args:
        samples_list (List): where each element is the predicted concentrations from a model and is shaped like (galaxy, answer, dropout_pass). Should be 2 answers i.e. dim1=2. One model is fine, one dropout pass is fine.
        observed_votes (np.ndarray): actual recorded responses, by galaxy. By tfp convention, needs to have both answers recorded explicitly e.g. [[2, 8], [4, 6]] not [2, 4]. This implies the total votes, which is useful.
        image_paths (list): paths to images, will imshow next to posteriors. Align with samples_list
        xlabel (str]): for final graph. Plotting only, no other effect.
        answer_index (int, optional): Which answer index to display on graph. Defaults to 0. Setting 1 will flip the posteriors.
        n_examples (int, optional): Number of galaxies (rows) to show. Defaults to 5.

    Returns:
        fig, axes: matplotlib figure, axes of posteriors and galaxy images row-by-row
    """
    
    cycler = mpl.rcParams['axes.prop_cycle']
    # https://matplotlib.org/cycler/
    colors = [c['color'] for c in cycler]
    assert len(colors) > len(samples_list)

    fig, axes = plt.subplots(nrows=n_examples, ncols=2, figsize=(6.8, 3 * n_examples), gridspec_kw={'hspace': .4})

    for galaxy_index in range(n_examples):
        
        row = axes[galaxy_index]
        total_votes = observed_votes[galaxy_index].sum()
        actual_votes = observed_votes[galaxy_index][0]  # show first index by default

        votes = np.linspace(0., total_votes)
        x = np.stack([votes, total_votes-votes], axis=-1)  # also need the counts for other answer, no

        n_dropout_samples = samples_list[0].shape[2]

        for model_n, samples in enumerate(samples_list):
            ax = row[answer_index]

            all_log_probs = []
            
            color = colors[model_n]

            for dropout_index in range(n_dropout_samples):
                concentrations = tf.constant(samples[galaxy_index, :, dropout_index].astype(np.float32))
                log_probs = tfp.distributions.DirichletMultinomial(total_votes, concentrations).prob(x)
                all_log_probs.append(log_probs)
                ax.plot(votes, log_probs, alpha=.15, color=color)
            all_log_probs = np.array(all_log_probs).mean(axis=0)
            ax.plot(votes, all_log_probs, linewidth=2., color=color)
            
            # avoid disappearing off the side if zero votes
            viz_offset = 0.004
            if actual_votes == 0:
                actual_votes += viz_offset * total_votes

            ax.axvline(actual_votes, color='k', linestyle='--', linewidth=2.)
            ax.set_xlim([0., total_votes])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel(r'$p$(votes)')
            
            ax = row[1]
            ax.imshow(np.array(Image.open(image_paths[galaxy_index])))
            ax.axis('off')
            
    # last row, left side
    ax = axes[-1][0]
    ax.set_xlabel(xlabel)

    return fig, axes


# if __name__ == '__main__':

    # example predictions to test with
    # from zoobot.predictions import predict_on_dataset

    # predictions = np.random.rand(100, 2)
    # image_paths = ['some_path_' + str(x) for x in np.arange(100)]
    # label_cols = ['ring', 'not_ring']
    # save_loc = '/Users/walml/repos/zoobot_private/test.hdf5'
    # predict_on_dataset.predictions_to_hdf5(predictions, image_paths, label_cols, save_loc)

    # version='decals'
    # label_cols = label_metadata.decals_label_cols
    # questions = label_metadata.decals_questions
    # schema = losses.Schema(label_cols, questions, version=version)

    # # question = schema.get_question('smooth-or-featured')
    # # question = schema.get_question('disk-edge-on')
    # question = schema.get_question('has-spiral-arms')
    # # question = schema.get_question('bar')
    # answer = question.answers[0]

    # catalog_loc = 'data/decals/decals_master_catalog.csv'
    # catalog = pd.read_csv(catalog_loc, dtype={'subject_id': str})  # original catalog
    # # catalog['file_loc'] = catalog['local_png_loc'].apply(lambda x: '/media/walml/beta/gz2' + x[32:])
    # catalog['file_loc'] = catalog['local_png_loc'].apply(lambda x: '/media/walml/beta1/decals' + x.replace('/data/phys-zooniverse/chri5177', ''))
    # retired = catalog[catalog['smooth-or-featured_total-votes'] > 35]


    # show_predictions(prediction_dfs, question, answer, n_examples=5, single_model=False)
