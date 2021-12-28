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
import tensorflow as tf

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

    galaxy_id_df = pd.DataFrame(data=prediction_metadata)
    # print(galaxy_id_df.head())
    # galaxy_id_df['local_png_loc'] = galaxy_id_df['image_paths'].str.replace('/share/nas/walml/galaxy_zoo/decals/dr5/png', '/Volumes/beta/decals/png_native/dr5')

    # print(galaxy_id_df['local_png_loc'].value_counts())

    assert not any(galaxy_id_df.duplicated(subset=['id_str']))
    assert not any(catalog.duplicated(subset=['id_str']))

    # which concentrations are safely in the catalog?
    catalog_locs = set(catalog['id_str'])
    # prediction_locs = set(galaxy_id_df['id_str'])

    print(galaxy_id_df['id_str'][0])
    print(catalog['id_str'][0])

    loc_is_in_catalog = galaxy_id_df['id_str'].isin(catalog_locs) 
    galaxy_id_df = galaxy_id_df[loc_is_in_catalog].reset_index(drop=True)  # now safe to left join on catalog
    concentrations_in_catalog = concentrations[loc_is_in_catalog]  # drop the concentrations corresponding to the dropped rows (galaxies with preds not in the catalog)

    # join catalog labels to df of galaxy ids to create label_df
    label_df = pd.merge(galaxy_id_df, catalog, how='left', on='id_str')  # will not change the galaxy_id_df index so will still match the prediction_in_catalog index
    print('Test predictions: {}'.format(len(label_df)))
    assert len(galaxy_id_df) == len(label_df)
    assert len(label_df) > 0

    if save_loc:
        label_df.to_parquet(save_loc, index=False)

    return label_df, concentrations_in_catalog


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


def get_validation_loss(label_df, concentrations, schema, batch_size=512, save_loc=None):

    answers = [a.text for a in schema.answers]
    labels = label_df[answers].values

    multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    loss = lambda x, y: multiquestion_loss(x, y) / batch_size  

    tf_labels = tf.constant(labels.astype(np.float32))
    tf_preds = tf.constant(concentrations.astype(np.float32).squeeze())[:, :, 0]  # TODO just picking first dropout pass for now
    tf_labels.shape, tf_preds.shape

    loss = losses.calculate_multiquestion_loss(tf_labels, tf_preds, schema.question_index_groups).numpy()  # this version doesn't reduce

    mean_loss_by_q = loss.mean(axis=0)
    rows = [{'question': question.text, 'mean_loss': l} for question, l in zip(schema.questions, mean_loss_by_q)]
    
    loss_by_q_df = pd.DataFrame(data=rows)

    if save_loc:
        loss_by_q_df.to_csv(save_loc, index=False)

    return loss, loss_by_q_df


def get_regression_errors(retired, predicted_fractions, schema, df_save_loc=None, fig_save_loc=None):

    errors = []
    min_total_votes = 20
    for question_n, question in enumerate(schema.questions):
        # print(question)
        # WARNING TODO
        if 'dr12' not in question.text:
            for answer in question.answers:
                print(question.text, answer.text)
                valid_labels, valid_predictions = filter_to_sensible(retired, predicted_fractions, question, schema)  # df, np.array
                valid_labels.reset_index()
                enough_votes = valid_labels[question.text + '_total-votes'] >= min_total_votes  # not quite the same as filter_to_sensible
                y_true = valid_labels.loc[enough_votes, answer.text + '_fraction']
                y_pred = valid_predictions[enough_votes, answer.index]
                assert not pd.isna(y_true).any()
                assert not pd.isna(y_pred).any()
                absolute = metrics.mean_absolute_error(y_true, y_pred)
                mse = metrics.mean_squared_error(y_true, y_pred)
                errors.append({'answer': answer.text, 'rmse': np.sqrt(mse), 'mean_absolute_error': absolute, 'question_n': question_n})

    regression_errors = pd.DataFrame(errors)

    if df_save_loc:
        regression_errors.to_csv(df_save_loc, index=False)

    if fig_save_loc:
        sns.set_style('whitegrid', {'axes.edgecolor': '0.2'})
        sns.set_context('notebook')
        # sns.set_palette(repeating_palette)
        fig, ax = plt.subplots(figsize=(6, 8))
        sns.barplot(data=regression_errors, y='answer', x='mean_absolute_error', ax=ax)
        plt.xlabel('Vote Fraction Mean Deviation')
        plt.ylabel('')
        fig.tight_layout()
        plt.savefig(fig_save_loc)

    return regression_errors


def regression_scatterplot(retired, predicted_fractions, answer_text, schema):

    # question = schema.get_question('has-spiral-arms')
    answer = schema.get_answer(answer_text)
    question = answer.question
    # answer = question.answers[0]
    valid_labels, valid_predictions = filter_to_sensible(retired, predicted_fractions, question, schema)
    sns.scatterplot(valid_labels[answer.text + '_fraction'], valid_predictions[:, answer.index], alpha=.1)





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
    # temporarily remove DR8 while that trains
    for q, answers in question_answer_pairs.copy().items():
        if 'dr8' in q:
            del question_answer_pairs[q]

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
    

    all_labels, all_concentrations = match_predictions_to_catalog(predictions_hdf5_locs, catalog, save_loc=None)
    print(len(all_labels))
    print(len(all_concentrations))
    # print(all_labels.head())

    is_retired = all_labels['smooth-or-featured_total-votes'] > 34
    retired_labels = all_labels[is_retired]
    retired_concentrations = all_concentrations[is_retired]

    print(retired_labels.columns.values)
    print(retired_concentrations.shape)

    predicted_fractions = dirichlet_stats.dirichlet_prob_of_answers(retired_concentrations, schema, temperature=None)

    print('Question & Count & Accuracy & Precision & Recall & F1 \\')  # need to add second slash back manually
    print('\hline \hline')
    for question in schema.questions:
        print_metrics(question, retired_labels, predicted_fractions, schema)

    print(r'Question & Count & Accuracy & Precision & Recall & F1 \\')
    print('\hline \hline')
    for question in schema.questions:

        answers = question.answers
        fractions = np.array([retired_labels[answer.text + '_fraction'] for answer in answers])
        high_confidence = np.any(fractions > 0.8, axis=0)
        print_metrics(question, retired_labels[high_confidence], predicted_fractions[high_confidence], schema)

    # confusion_matrices_split_by_confidence(retired_labels, predicted_fractions, schema, save_dir, normalize=normalize_cm_matrices)

    # unlike cm and regression, does not only include retired (i.e. high N) galaxies
    val_loss, loss_by_q_df = get_validation_loss(all_labels, all_concentrations, schema=schema, save_loc=os.path.join(save_dir, 'val_loss_by_q.csv'))
    print(val_loss.shape)
    print(loss_by_q_df)

    print(val_loss.sum())  # 137300.42 for dr5 only, 133340.14 for dr12 also

    get_regression_errors(
        retired=retired_labels,
        predicted_fractions=predicted_fractions,
        schema=schema,
        df_save_loc=os.path.join(save_dir, 'regression_errors.csv'),
        fig_save_loc=os.path.join(save_dir, 'regression_errors_bar_plot.pdf')
    )
