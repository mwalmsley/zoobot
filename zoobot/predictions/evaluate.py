from math import log
import os
import logging
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import metrics
import h5py
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, mean_squared_error, mean_absolute_error

from zoobot import label_metadata, schemas
from zoobot.training import losses
from zoobot.stats import dirichlet_stats, vote_stats


def match_predictions_to_catalog(predictions_hdf5_locs: List, catalog, save_loc=None):

    # predictions will potentially not be the same galaxies as the catalog
    # need to carefully line them up 

    concentrations = []
    prediction_metadata = []
    for loc in predictions_hdf5_locs:
        with h5py.File(loc, 'r') as f:
            these_concentrations = f['predictions'][:]
            these_prediction_metadata = {
                'id_str': f['id_str'].asstr()[:]
            }
            # print(these_prediction_metadata)
            concentrations.append(these_concentrations)
            prediction_metadata.append(these_prediction_metadata)

    concentrations = np.concatenate(concentrations, axis=0)
    prediction_metadata = {'id_str': [p for metadata in prediction_metadata for p in metadata['id_str']]}
    print(len(prediction_metadata['id_str']), len(concentrations))

    prediction_df = pd.DataFrame(data=prediction_metadata)
    # print(prediction_df.head())
    # prediction_df['local_png_loc'] = prediction_df['image_paths'].str.replace('/share/nas/walml/galaxy_zoo/decals/dr5/png', '/Volumes/beta/decals/png_native/dr5')

    # print(prediction_df['local_png_loc'].value_counts())

    assert not any(prediction_df.duplicated(subset=['id_str']))
    assert not any(catalog.duplicated(subset=['id_str']))

    # which concentrations are safely in the catalog?
    catalog_locs = set(catalog['id_str'])
    # prediction_locs = set(prediction_df['id_str'])

    print(prediction_df['id_str'][0])
    print(catalog['id_str'][0])

    loc_is_in_catalog = prediction_df['id_str'].isin(catalog_locs) 
    prediction_df = prediction_df[loc_is_in_catalog].reset_index(drop=True)  # now safe to left join on catalog
    concentrations_in_catalog = concentrations[loc_is_in_catalog]  # drop the concentrations corresponding to the dropped rows (galaxies with preds not in the catalog)

    df = pd.merge(prediction_df, catalog, how='left', on='id_str')  # will not change the prediction_df index so will still match the prediction_in_catalog index
    print('Test predictions: {}'.format(len(df)))
    assert len(prediction_df) == len(df)
    assert len(df) > 0

    if save_loc:
        df.to_parquet(save_loc, index=False)

    return df, concentrations_in_catalog


"""Plotting"""


# copied from trust_the_model.ipynb
def show_galaxies(df, scale=3, nrows=6, ncols=6):
    fig = plt.gcf()

    # plt.figure(figsize=(scale * nrows * 1.505, scale * ncols / 2.59))
    plt.figure(figsize=(scale * nrows * 1., scale * ncols))
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.0, hspace=0.0)
    galaxy_n = 0
    for row_n in range(nrows):
        for col_n in range(ncols):
            galaxy = df.iloc[galaxy_n]
            image = Image.open(galaxy['file_loc'])
            ax = plt.subplot(gs1[row_n, col_n])
            ax.imshow(image)

            ax.text(35, 40, 'smooth: V={:.2f}, ML={:.2f}'.format(galaxy['smooth-or-featured_smooth_true_fraction'], galaxy['smooth-or-featured_smooth_predicted_fraction']), fontsize=12, color='r')
            if galaxy['smooth-or-featured_smooth_true_fraction'] < 0.5:
                ax.text(35, 100, 'arms: V={:.2f}, ML={:.2f}'.format(galaxy['has-spiral-arms_yes_true_fraction'], galaxy['has-spiral-arms_yes_predicted_fraction']), fontsize=12, color='r')


            # ax.text(35, 50, 'Vol: 2={:.2f}, ?={:.2f}'.format(galaxy['spiral-arm-count_2_fraction'], galaxy['spiral-arm-count_cant-tell_fraction']), fontsize=12, color='r')
            # ax.text(35, 100, 'ML: 2={:.2f}, ?={:.2f}'.format(galaxy['spiral-arm-count_2_ml_fraction'], galaxy['spiral-arm-count_cant-tell_ml_fraction']), fontsize=12, color='r')
#             ax.text(10, 50, r'$\rho = {:.2f}$, Var ${:.3f}$'.format(galaxy['median_prediction'], 3*galaxy['predictions_var']), fontsize=12, color='r')
#             ax.text(10, 80, '$L = {:.2f}$'.format(galaxy['bcnn_likelihood']), fontsize=12, color='r')
            ax.axis('off')
            galaxy_n += 1
#     print('Mean L: {:.2f}'.format(df[:nrows * ncols]['bcnn_likelihood'].mean()))
    fig = plt.gcf()
    fig.tight_layout()
    return fig

"""Discrete Metrics"""

def get_label(text, question):
    return clean_text(text.replace(question.text, '').title())


def clean_text(text):
    return text.replace('-', ' ').replace('_', '').title()


def filter_to_sensible(label_df, predictions, question, schema):
#     if prev_a is not None:
#         prev_q = prev_a.question
#         prev_q_cols = [answer.text + '_fraction' for answer in prev_q.answers]
#         is_sensible = (label_df[prev_a.text + '_fraction'] / label_df[prev_q_cols].sum(axis=1)) > 0.5
#         valid_labels, valid_predictions = label_df[is_sensible], predicted_fractions[is_sensible]
#     else:
#         valid_labels, valid_predictions = label_df, predicted_fractions
    retirement = 1
    expected_votes = vote_stats.get_expected_votes_human(label_df, question, retirement, schema, round_votes=False)
    if not isinstance(expected_votes, np.ndarray):
        expected_votes = expected_votes.numpy()  # hack, should fix properly...
#     print(expected_votes)
    is_sensible = expected_votes > 0.5
    valid_labels, valid_predictions = label_df[is_sensible], predictions[is_sensible]
    return valid_labels, valid_predictions

def get_binary_responses(question, label_df, predicted_fractions, schema):
    # previous question should be valid
    valid_labels, valid_predictions = filter_to_sensible(label_df, predicted_fractions, question, schema)
    cols = [answer.text + '_fraction' for answer in question.answers]
    # most likely answer, might be less than .5 though
    y_true = np.argmax(valid_labels[cols].values, axis=1)
    y_pred = np.argmax(valid_predictions[:, question.start_index:question.end_index+1], axis=1)
    return y_true, y_pred

def print_metrics(question, label_df, predicted_fractions, schema):
    y_true, y_pred = get_binary_responses(question, label_df, predicted_fractions, schema)
#     print(pd.value_counts(y_pred))
#     average = 'micro'
    average = 'weighted'
    
    # human
#     print('Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f} <- {}'.format(
#         sklearn.metrics.accuracy_score(y_true, y_pred),
#         sklearn.metrics.precision_score(y_true, y_pred, average=average),
#         sklearn.metrics.recall_score(y_true, y_pred, average=average),
#         sklearn.metrics.f1_score(y_true, y_pred, average=average),
#         question.text
#     ))
    # latex
    print('{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(
        question.text.replace('-', ' ').replace('_', ' ').title(),
        len(y_true),
        metrics.accuracy_score(y_true, y_pred),
        metrics.precision_score(y_true, y_pred, average=average),
        metrics.recall_score(y_true, y_pred, average=average),
        metrics.f1_score(y_true, y_pred, average=average)
))

    
def show_confusion_matrix(question, label_df, predicted_fractions, ax=None, blank_yticks=False, add_title=False, normalize=False):
    y_true, y_pred = get_binary_responses(question, label_df, predicted_fractions, schema)
    
    labels = range(len(question.answers))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    
    ticklabels = [get_label(a.text, question) for a in question.answers]
    
    # manual tweaks
    for n in range(len(ticklabels)):
        if ticklabels[n] == 'Featured Or Disk':
            ticklabels[n] = 'Featured/Disk'
        elif ticklabels[n] == 'Cigar Shaped':
            ticklabels[n] = 'Cigar'
        elif ticklabels[n] == 'More Than 4':
            ticklabels[n] = '>4'
        elif ticklabels[n] == 'Cant Tell':
            ticklabels[n] = '?'
        elif ticklabels[n] == 'Minor Disturbance':
            ticklabels[n] = 'Minor Dist.'
        elif ticklabels[n] == 'Major Disturbance':
            ticklabels[n] = 'Major Dist.'
            
    if ax is None:
        _, ax = plt.subplots()
    
    if add_title:
        ax.set_title(clean_text(question.text))

    if blank_yticks:
#         yticklabels = ['' for _ in ticklabels]
        yticklabels = [''.join([' '] * len(s)) for s in ticklabels]
    else:
        yticklabels = ticklabels

    if normalize == 'true':
        fmt = '.3f'
    else:
        fmt = 'd'
    return sns.heatmap(
        cm,
        annot=True,
        cmap='Blues',
        fmt=fmt,
        xticklabels=ticklabels,
        yticklabels=yticklabels,
        cbar=False,
#         annot_kws={"size": 14},
        ax=ax,
        square=True,
        robust=True
    )


def confusion_matrices_split_by_confidence(retired, predicted_fractions, schema, save_dir, normalize=False):
    for question in schema.questions:
        print(question.text)
        

        fig = plt.figure(constrained_layout=True, figsize=(10.5, 5))
        gs = fig.add_gridspec(8, 10)

        ax0 = fig.add_subplot(gs[:6, 0])
        ax1 = fig.add_subplot(gs[:6, 1:5])
        ax2 = fig.add_subplot(gs[:6, 6:])

        ax3 = fig.add_subplot(gs[6:, 1:5])
        ax4 = fig.add_subplot(gs[6:, 6:])

        ax5 = fig.add_subplot(gs[:6, 5:6])
        
    #     fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 10))
        fig = show_confusion_matrix(question, retired, predicted_fractions, ax=ax1, normalize=normalize)

        fractions = np.array([retired[answer.text + '_fraction'] for answer in question.answers])
        high_confidence = np.any(fractions > 0.8, axis=0)
        fig = show_confusion_matrix(question, retired[high_confidence], predicted_fractions[high_confidence], ax=ax2, blank_yticks=False, normalize=normalize)
        
        label_size = 16
        ax3.text(0.5, 0.75, 'True', horizontalalignment='center', verticalalignment='center', fontsize=label_size)
        ax4.text(0.5, 0.75, 'True', horizontalalignment='center', verticalalignment='center', fontsize=label_size)

        ax0.text(0.8, 0.5, 'Predicted', rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=label_size)
        ax0.text(-0.5, 0.5, clean_text(question.text), rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=label_size, weight='bold')

        for ax in [ax0, ax3, ax4, ax5]:
            ax.grid('off')
            ax.axis('off')

        # comment to view vs save. clf needed to not interfere with each other
        if normalize == 'true':
            norm_text = 'normalized'
        else:
            norm_text = ''
        plt.savefig(f'{save_dir}/cm_decals_dr_full_ensemble_paired_{norm_text}_{question.text}.png')
        # plt.clf()  
        plt.close()
        

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # # model = 'all_campaigns'
    # model = 'dr5'

    # # shards = 'dr12'
    # shards = 'dr5'
    # # shards = 'dr8'

    # if model == 'dr5':
    #     question_answer_pairs = label_metadata.decals_pairs
    # else:
    #     question_answer_pairs = label_metadata.decals_all_campaigns_pairs

    # dependencies = label_metadata.get_gz2_and_decals_dependencies(question_answer_pairs)
    # schema = schemas.Schema(question_answer_pairs, dependencies)
    # logging.info('Schema: {}'.format(schema))

    # catalog = pd.read_parquet('/home/walml/repos/zoobot_private/dr5_volunteer_catalog_internal.parquet')
    # catalog['id_str'] = catalog['iauname']

    # # catalog = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/catalogs/{shards}_labelled_catalog.parquet')
    # # don't use for DR5-only model, a bit unfair as included dr12 responses

    # predictions_hdf5_locs = [f'/home/walml/repos/gz-decals-classifiers/results/all_campaigns_{shards}_pretrained_{model}.hdf5']

    question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
    dependencies = label_metadata.get_decals_ortho_dependencies(question_answer_pairs)
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    shards = 'dr5'
    # variation = ''
    variation = '_dr5_only'
    run_name = f'{shards}_pretrained_ortho{variation}'
    # normalize_cm_matrices = 'true'
    normalize_cm_matrices = None

    catalog = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/catalogs/{shards}_only_labelled_catalog.parquet')
    predictions_hdf5_locs = [f'/home/walml/repos/gz-decals-classifiers/results/{shards}_pretrained_ortho{variation}.hdf5']

    for question in schema.questions:
        for answer in question.answers:
            catalog[answer.text + '_fraction'] = catalog[answer.text].astype(float) / catalog[question.text + '_total-votes'].astype(float)

    # print(catalog['smooth-or-featured_smooth'])
    # print(catalog['smooth-or-featured_total-votes'])
    # print(catalog['smooth-or-featured_smooth_fraction'])
    # exit()

    # plt.hist(catalog['smooth-or-featured_smooth_fraction'], range=(0, 1), bins=30)
    # plt.show()

    save_dir = f'/home/walml/repos/gz-decals-classifiers/results/campaign_comparison/{run_name}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    

    all_preds, all_concentrations = match_predictions_to_catalog(predictions_hdf5_locs, catalog, save_loc=None)


    # load hdf5 predictions


    print(len(all_preds))
    print(len(all_concentrations))
    print(all_preds.head())

    is_retired = all_preds['smooth-or-featured_total-votes'] > 34
    retired_preds = all_preds[is_retired]
    retired_concentrations = all_concentrations[is_retired]

    print(retired_preds.columns.values)
    print(retired_concentrations.shape)
    # exit()

    predicted_fractions = dirichlet_stats.dirichlet_prob_of_answers(retired_concentrations, schema, temperature=None)

    print('Question & Count & Accuracy & Precision & Recall & F1 \\')  # need to add second slash back manually
    print('\hline \hline')
    for question in schema.questions:
        print_metrics(question, retired_preds, predicted_fractions, schema)
        # plt.savefig(f'/home/walml/repos/gz-decals-classifiers/results/campaign_comparison/{run_name}/mean_cm_decals_n2_m0_allq_' + question.text + '.png')

    # for question in schema.questions:
    #     fig = show_confusion_matrix(question, retired_preds, predicted_fractions)
    #     plt.tight_layout()

    print(r'Question & Count & Accuracy & Precision & Recall & F1 \\')
    print('\hline \hline')
    for question in schema.questions:
    #     if len(question.answers) == 2:
    #         answer = question.answers[0]
        answers = question.answers
        fractions = np.array([retired_preds[answer.text + '_fraction'] for answer in answers])
        high_confidence = np.any(fractions > 0.8, axis=0)
        print_metrics(question, retired_preds[high_confidence], predicted_fractions[high_confidence], schema)
        
    #     fig = show_confusion_matrix(question, retired_preds[high_confidence], predicted_fractions[high_confidence])
    #     plt.tight_layout()
    #     # plt.savefig('/home/walml/repos/zoobot/results/temp/thesis_cm_decals_dr_full_ensemble_allq_highconf_' + question.text + '.png')
    # #     plt.clf()  # comment to view, uncomment to save without interfering with each other

    confusion_matrices_split_by_confidence(retired_preds, predicted_fractions, schema, save_dir, normalize=normalize_cm_matrices)