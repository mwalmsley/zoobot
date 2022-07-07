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

from zoobot.shared import schemas
from zoobot.shared import label_metadata
from zoobot.tensorflow.training import losses
from zoobot.tensorflow.stats import dirichlet_stats, vote_stats
from zoobot.tensorflow.predictions import load_predictions


def match_predictions_to_catalog(predictions_hdf5_locs: List, catalog, save_loc=None):

    # predictions will potentially not be the same galaxies as the catalog
    # need to carefully line them up 

    galaxy_id_df, concentrations, _ = load_predictions.load_hdf5s(predictions_hdf5_locs)

    assert not any(galaxy_id_df.duplicated(subset=['id_str']))
    assert not any(catalog.duplicated(subset=['id_str']))

    # which concentrations are safely in the catalog?
    catalog_locs = set(catalog['id_str'])

    loc_is_in_catalog = galaxy_id_df['id_str'].isin(catalog_locs) 
    galaxy_id_df = galaxy_id_df[loc_is_in_catalog].reset_index(drop=True)  # now safe to left join on catalog
    concentrations_in_catalog = concentrations[loc_is_in_catalog]  # drop the concentrations corresponding to the dropped rows (galaxies with preds not in the catalog)

    # join catalog labels to df of galaxy ids to create label_df

    print(catalog.iloc[0]['id_str'])
    label_df = pd.merge(galaxy_id_df, catalog, how='left', on='id_str')  # will not change the galaxy_id_df index so will still match the prediction_in_catalog index
    print(label_df['hdf5_loc'].value_counts())
    assert len(galaxy_id_df) == len(label_df)
    assert len(label_df) > 0

    print('Predictions: {}'.format(len(label_df)))
    # print('Galaxies from each hdf5:')

    if save_loc:
        label_df.to_parquet(save_loc, index=False)

    return label_df, concentrations_in_catalog


def multi_catalog_tweaks(label_df):
    smooth_featured_cols = [col for col in label_df.columns.values if ('smooth-or-featured' in col) and ('total-votes') in col]
    # print(smooth_featured_cols)
    match_with_dr = {
        'smooth-or-featured-dr12_total-votes': 'has_dr12_votes',
        'smooth-or-featured-dr5_total-votes': 'has_dr5_votes',
        'smooth-or-featured-dr8_total-votes': 'has_dr8_votes',
    }

    for col in smooth_featured_cols:
        matched_col = match_with_dr[col]
        label_df[matched_col] = label_df[col] > 0
        print('Test galaxies with {} non-zero votes: {}'.format(col, label_df[matched_col].sum()))

    return label_df


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
    assert len(label_df) == len(predictions)

    # want to filter to only galaxies with at least 1 human vote for that question
    base_q_votes = (label_df[question.text + '_total-votes'] > 0).values.astype(int)  # 0 if no votes, 1 otherwise (i.e. assume the base q has 1 vote, below)
    
    expected_votes = vote_stats.get_expected_votes_human(label_df, question, base_q_votes, schema, round_votes=False)  # human fractions imply you'd get this many votes for this question
    if not isinstance(expected_votes, np.ndarray):
        expected_votes = expected_votes.numpy()  # hack, should fix properly...

    at_least_half_of_humans_would_answer = expected_votes > 0.5
    if at_least_half_of_humans_would_answer.sum() == 0:
        logging.warning('No galaxies with odds of human being asked that q above 0.5: {}, {} candidates'.format(question, len(label_df)))

    valid_labels, valid_predictions = label_df[at_least_half_of_humans_would_answer], predictions[at_least_half_of_humans_would_answer]

    return valid_labels, valid_predictions


def print_paper_style_metric_tables(labels, pred_fractions, schema, style='human'):

    print('Metrics on all galaxies:')
    print('Question & Count & Accuracy & Precision & Recall & F1 \\')  # need to add second slash back manually
    print('\hline \hline')
    for question in schema.questions:
        print_metrics(question, labels, pred_fractions, schema, style=style)

    print('Metrics on (retired) high confidence galaxies:')
    print(r'Question & Count & Accuracy & Precision & Recall & F1 \\')
    print('\hline \hline')
    for question in schema.questions:

        answers = question.answers
        fractions = np.array([labels[answer.text + '_fraction'] for answer in answers])
        high_confidence = np.any(fractions > 0.8, axis=0)
        print('High conf galaxies: {}'.format(high_confidence.sum()))
        print_metrics(question, labels[high_confidence], pred_fractions[high_confidence], schema, style=style)



def print_metrics(question, label_df, predicted_fractions, schema, style='human'):

    y_true, y_pred = get_binary_responses(question, label_df, predicted_fractions, schema)

    # how to handle multi-class metrics - see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    # average = 'micro'  # "Calculate metrics globally by counting the total true positives, false negatives and false positives.""
    average = 'weighted'  # "Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).""
    
    if style == 'human':
        print('Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f} <- {}'.format(
            metrics.accuracy_score(y_true, y_pred),
            metrics.precision_score(y_true, y_pred, average=average),
            metrics.recall_score(y_true, y_pred, average=average),
            metrics.f1_score(y_true, y_pred, average=average),
            question.text
        ))
    elif style == 'latex':
        print('{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(
            question.text.replace('-', ' ').replace('_', ' ').title(),
            len(y_true),
            metrics.accuracy_score(y_true, y_pred),
            0,
            metrics.precision_score(y_true, y_pred, average=average),
            metrics.recall_score(y_true, y_pred, average=average),
            0,
            metrics.f1_score(y_true, y_pred, average=average)
        ))


def get_binary_responses(question, label_df, predicted_fractions, schema):

    # print(label_df['has_dr12_votes'].sum(), label_df['has_dr5_votes'].sum(), label_df['has_dr8_votes'].sum())

    # previous question should be valid
    valid_labels, valid_predictions = filter_to_sensible(label_df, predicted_fractions, question, schema)

    # print(question.text, len(valid_labels))

    cols = [answer.text + '_fraction' for answer in question.answers]
    # most likely answer, might be less than .5 though
    y_true = np.argmax(valid_labels[cols].values, axis=1)
    y_pred = np.argmax(valid_predictions[:, question.start_index:question.end_index+1], axis=1)
    return y_true, y_pred


def show_confusion_matrix(question, label_df, predicted_fractions, schema, ax=None, blank_yticks=False, add_title=False, normalize=False):
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


def confusion_matrices_split_by_confidence(retired: pd.DataFrame, predicted_fractions: np.ndarray, schema: schemas.Schema, save_dir: str, cm_name='cm_decals_dr_full_ensemble_paired', normalize=False):
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
        fig = show_confusion_matrix(question, retired, predicted_fractions, schema, ax=ax1, normalize=normalize)

        fractions = np.array([retired[answer.text + '_fraction'] for answer in question.answers])
        high_confidence = np.any(fractions > 0.8, axis=0)
        fig = show_confusion_matrix(question, retired[high_confidence], predicted_fractions[high_confidence], schema, ax=ax2, blank_yticks=False, normalize=normalize)
        
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
        plt.savefig(f'{save_dir}/{cm_name}_{norm_text}_{question.text}.png')
        # plt.clf()  
        plt.close()


def get_loss(label_df, concentrations, schema, batch_size=512, save_loc=None):

    answers = [a.text for a in schema.answers]
    labels = label_df[answers].values

    multiquestion_loss = losses.get_multiquestion_loss(schema.question_index_groups)
    loss = lambda x, y: multiquestion_loss(x, y) / batch_size  

    tf_labels = tf.constant(labels.astype(np.float32))
    tf_preds = tf.constant(concentrations.astype(np.float32).squeeze())[:, :, 0]  # TODO just picking first dropout pass for now
    tf_labels.shape, tf_preds.shape

    loss = losses.calculate_multiquestion_loss(tf_labels, tf_preds, schema.question_index_groups).numpy()  # this version doesn't reduce

    mean_loss_by_q = loss.mean(axis=0).squeeze()
    rows = [{'question': question.text, 'mean_loss': l} for question, l in zip(schema.questions, mean_loss_by_q)]
    
    loss_by_q_df = pd.DataFrame(data=rows)

    if save_loc:
        loss_by_q_df.to_csv(save_loc, index=False)

    return loss, loss_by_q_df


def get_regression_errors(retired: pd.DataFrame, predicted_fractions: np.ndarray, schema: schemas.Schema, min_total_votes=20, df_save_loc=None, fig_save_loc=None):

    errors = []
    for question_n, question in enumerate(schema.questions):
        valid_labels, valid_predictions = filter_to_sensible(retired, predicted_fractions, question, schema)
        if len(valid_labels) == 0:
            logging.warning('Skipping regression - no valid labels/predictions for answer {}, {} candidates'.format(question.text, len(retired)))
        else:
            valid_labels = valid_labels.reset_index(drop=True)
            enough_votes = valid_labels[question.text + '_total-votes'] >= min_total_votes  # not quite the same as filter_to_sensible
            # print(enough_votes.mean())
            if not any(enough_votes):
                logging.warning('Skipping regression - valid labels/predictions but none with more than {} votes - {}'.format(min_total_votes, answer.text))
            else:
                for answer in question.answers:
                    y_true = valid_labels.loc[enough_votes, answer.text + '_fraction']
                    y_pred = valid_predictions[enough_votes, answer.index]
                    assert not pd.isna(y_true).any()
                    assert not pd.isna(y_pred).any()
                    # print(len(y_true), len(y_pred))
                    absolute = metrics.mean_absolute_error(y_true, y_pred)
                    mse = metrics.mean_squared_error(y_true, y_pred)
                    errors.append({'answer': answer.text, 'rmse': np.sqrt(mse), 'mean_absolute_error': absolute, 'question_n': question_n})

    assert len(errors) > 0
    regression_errors = pd.DataFrame(errors)

    if df_save_loc:
        regression_errors.to_csv(df_save_loc, index=False)

    if fig_save_loc:
        sns.set_style('whitegrid', {'axes.edgecolor': '0.2'})
        sns.set_context('notebook')
        # sns.set_palette(repeating_palette)
        fig, ax = plt.subplots(figsize=(6, 20))
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


# def replace_dr12_preds_with_dr5_preds(predictions, schema):
#     predictions = predictions.copy()  # do not modify inplace
#     # predictions uses ortho schema

#     # shared questions are any q in the non-ortho dr5/dr12 schema that don't have "dr12" in them
#     # (decals_dr12_questions uses dr5 columns and col-dr12 for changed columns)
#     shared_questions = [q for q in label_metadata.decals_dr12_questions if "dr12" not in q.text]
#     # these will have the wrong indices for predictions/schema, which is ortho - just use for text

#     for q in shared_questions:
#         dr5_ortho_q = schema.get_question(q.text) + '-dr5' # will appear in the ortho schema (would have -dr5, but is implicit in this case)
#         dr12_ortho_q = schema.get_question(q.text + '-dr12')

#         # now replace the dr5 predictions with the dr12 values, both using the ortho schema
#         for answer_n in range(q.answers):
#             dr5_ortho_answer = dr5_ortho_q.answers[answer_n]
#             dr12_ortho_answer = dr12_ortho_q.answers[answer_n]
#             predictions[dr12_ortho_answer.index] = predictions[dr5_ortho_answer.index]

#     return predictions


def replace_dr8_cols_with_dr5_cols(predicted_fracs: np.ndarray, schema: schemas.Schema):
    predicted_fracs = predicted_fracs.copy()
    # this is easier than the above as the columns match exactly
    dr5_questions = [q for q in schema.questions if 'dr5' in q.text]
    dr8_questions = [q for q in schema.questions if 'dr8' in q.text]
    # will line up

    for dr5_q, dr8_q in zip(dr5_questions, dr8_questions):
        for dr5_a, dr8_a in zip(dr5_q.answers, dr8_q.answers):
            predicted_fracs[:, dr8_a.index] = predicted_fracs[:, dr5_a.index]

    return predicted_fracs


def main():

    question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    # logging.info('Schema: {}'.format(schema))

    """Pick which model's predictions to load"""

    shards = ['dr12', 'dr5', 'dr8']

    model_index = 'm4'

    # checkpoint = f'all_campaigns_ortho_v2_{model_index}'
    # checkpoint = f'all_campaigns_ortho_v2_train_only_dr5_{model_index}'
    checkpoint = f'all_campaigns_ortho_v2_train_only_d12_dr5_{model_index}'  # d12 typo, oops

# /home/walml/repos/gz-decals-classifiers/results/test_shard_dr8_checkpoint_all_campaigns_ortho_v2_train_only_d12_dr5_m0.hdf5 have
# /home/walml/repos/gz-decals-classifiers/results/test_shard_dr8_checkpoint_all_campaigns_ortho_v2_train_only_dr12_dr5_m0.hdf5 looking

    predictions_hdf5_locs = [f'/home/walml/repos/gz-decals-classifiers/results/test_shard_{shard}_checkpoint_{checkpoint}.hdf5' for shard in shards]
    print(predictions_hdf5_locs)
    predictions_hdf5_locs = [loc for loc in predictions_hdf5_locs if os.path.exists(loc)]
    assert len(predictions_hdf5_locs) > 0
    logging.info('Num. prediction .hdf5 to load: {}'.format(len(predictions_hdf5_locs)))


    """Specify some details for saving"""
    run_name = f'checkpoint_{checkpoint}'
    save_dir = f'/home/walml/repos/gz-decals-classifiers/results/campaign_comparison/{run_name}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # normalize_cm_matrices = 'true'
    normalize_cm_matrices = None


    """Load volunteer catalogs and match to predictions"""

    catalog_dr12 = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/catalogs/dr12_ortho_v2_labelled_catalog.parquet')
    catalog_dr5 = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/catalogs/dr5_ortho_v2_labelled_catalog.parquet')
    catalog_dr8 = pd.read_parquet(f'/home/walml/repos/gz-decals-classifiers/catalogs/dr8_ortho_v2_labelled_catalog.parquet')
    catalog = pd.concat([catalog_dr12, catalog_dr5, catalog_dr8], axis=0).reset_index()

    # possibly, catalogs don't include _fraction cols?! 
    # for question in schema.questions:
    #     for answer in question.answers:
    #         catalog[answer.text + '_fraction'] = catalog[answer.text].astype(float) / catalog[question.text + '_total-votes'].astype(float)

    all_labels, all_concentrations = match_predictions_to_catalog(predictions_hdf5_locs, catalog, save_loc=None)
    # print(len(all_labels))
    # print(len(all_concentrations))
    # print(all_labels.head())

    all_labels = multi_catalog_tweaks(all_labels)

    # not actually used currently
    all_fractions = dirichlet_stats.dirichlet_prob_of_answers(all_concentrations, schema, temperature=None)

    # plt.hist(all_labels['smooth-or-featured-dr12_total-votes'], alpha=.5, label='dr12', range=(0, 80))  # 7.5k retired
    # plt.hist(all_labels['smooth-or-featured_total-votes'], alpha=.5, label='dr5', range=(0, 80))  # 2.2k retired
    # plt.hist(all_labels['smooth-or-featured-dr8_total-votes'], alpha=.5, label='dr8', range=(0, 80))  # half have almost no votes, 4.6k retired
    # plt.show()
    # # exit()

    votes_for_retired = 34
    all_labels['is_retired_in_dr12'] = all_labels['smooth-or-featured-dr12_total-votes'] > votes_for_retired
    all_labels['is_retired_in_dr5'] = all_labels['smooth-or-featured-dr5_total-votes'] > votes_for_retired
    all_labels['is_retired_in_dr8'] = all_labels['smooth-or-featured-dr8_total-votes'] > votes_for_retired
    all_labels['is_retired_in_any_dr'] = all_labels['is_retired_in_dr12'] | all_labels['is_retired_in_dr5'] | all_labels['is_retired_in_dr8']

    retired_concentrations = all_concentrations[all_labels['is_retired_in_any_dr']]
    retired_labels = all_labels.query('is_retired_in_any_dr')
    retired_fractions = dirichlet_stats.dirichlet_prob_of_answers(retired_concentrations, schema, temperature=None)

    logging.info('All concentrations: {}'.format(all_concentrations.shape))
    logging.info('Retired concentrations: {}'.format(retired_concentrations.shape))

    # print(all_labels['is_retired_in_dr5'].sum())
    # exit()

    """Now we're ready to calculate some metrics"""

    # create_paper_metric_tables(retired_labels, retired_fractions, schema)

    # # the least interpretable but maybe most ml-meaningful metric
    # # unlike cm and regression, does not only include retired (i.e. high N) galaxies
    # val_loss, loss_by_q_df = get_loss(all_labels, all_concentrations, schema=schema, save_loc=os.path.join(save_dir, 'val_loss_by_q.csv'))
    # print('Mean val loss: {:.3f}'.format(val_loss.mean()))

    confusion_matrices_split_by_confidence(retired_labels, retired_fractions, schema, save_dir, normalize=normalize_cm_matrices, cm_name='cm')

    # print((retired_labels['smooth-or-featured-dr12_total-votes'] > 20).sum())

    get_regression_errors(
        retired=retired_labels,
        predicted_fractions=retired_fractions,
        schema=schema,
        df_save_loc=os.path.join(save_dir, 'regression_errors.csv'),
        fig_save_loc=os.path.join(save_dir, 'regression_errors_bar_plot.pdf')
    )

    """And we can repeat the process, but using the DR5 predictions for the DR8 answer columns"""

    # pick the rows with dr8 galaxies (dr8_)
    dr8_galaxies = all_labels['has_dr8_votes'].values
    dr8_labels = all_labels[dr8_galaxies]
    dr8_concentrations = all_concentrations[dr8_galaxies]
    dr8_fractions = all_fractions[dr8_galaxies]
    # convert the predictions for dr8 answers to use the dr5 answers instead
    dr8_fractions_with_dr5_head = replace_dr8_cols_with_dr5_cols(dr8_fractions, schema)
    dr8_concentrations_with_dr5_head = replace_dr8_cols_with_dr5_cols(dr8_concentrations, schema)

    # calculate loss on all dr8 galaxies, using dr5 head answers
    val_loss, loss_by_q_df = get_loss(dr8_labels, dr8_concentrations_with_dr5_head, schema=schema, save_loc=os.path.join(save_dir, 'val_loss_by_q_dr8_galaxies_with_dr5_head_for_dr8.csv'))
    logging.info('Mean val loss for DR8 galaxies using DR5 head for DR8: {:.3f}'.format(val_loss.mean()))
    
    # select the retired dr8 galaxies by row
    dr8_retired_galaxies = dr8_labels['is_retired_in_dr8'].values
    dr8_retired_labels = dr8_labels[dr8_retired_galaxies]
    dr8_retired_concentrations_with_dr5_head = dr8_concentrations_with_dr5_head[dr8_retired_galaxies]
    dr8_retired_fractions_with_dr5_head = dr8_fractions_with_dr5_head[dr8_retired_galaxies]

    # will give a bunch of warnings as we've only selected dr8 galaxies hence dr12 and dr5 will not have enough votes and give empty confusion matrices
    confusion_matrices_split_by_confidence(
        dr8_retired_labels,
        dr8_retired_fractions_with_dr5_head,
        schema,
        save_dir,
        normalize=normalize_cm_matrices,
        cm_name='cm_dr8_galaxies_with_dr5_head_for_dr8'
    )

    # similarly will throw dr12 and dr5 warnings
    get_regression_errors(
        retired=dr8_retired_labels,
        predicted_fractions=dr8_retired_fractions_with_dr5_head,
        schema=schema,
        df_save_loc=os.path.join(save_dir, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'),
        fig_save_loc=os.path.join(save_dir, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8_bar_plot.pdf')
    )


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    main()

