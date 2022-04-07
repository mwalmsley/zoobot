import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from zoobot.tensorflow.evaluation import evaluate_model

def compare_metric_by_question(eval_dirs, metric='mean_loss', save_loc=None):

    if metric == 'mean_loss':
        csv_name = 'val_loss_by_q.csv'
        x_name = 'mean_loss'
        y_name = 'question'
    elif metric == 'mae':
        csv_name = 'regression_errors.csv'
        x_name = 'mean_absolute_error'
        y_name = 'answer'
    else:
        raise ValueError(metric)

    data_by_dir = [pd.read_csv(os.path.join(d, csv_name)) for d in eval_dirs]
    for (df, d) in zip(data_by_dir, eval_dirs):
        df['eval_dir'] = os.path.basename(d).replace('_m0', '').replace('_m1', '').replace('_m2', '')  # modify by ref
        # TODO temp
        # for col in df.columns.values:
        #     if 'dr12' in col:
        #         del df[col]
    data = pd.concat(data_by_dir)
                                
    sns.set_style('whitegrid', {'axes.edgecolor': '0.2'})
    sns.set_context('notebook')
    # sns.set_palette(repeating_palette)
    fig, ax = plt.subplots(figsize=(6, 8))

    print(data)

    sns.barplot(data=data, y=y_name, x=x_name, hue='eval_dir', ax=ax)
    plt.xlabel('Mean Loss')
    plt.ylabel('')
    fig.tight_layout()
    if save_loc:
        plt.savefig(save_loc)
    else:
        plt.show()

    # TODO for paper, may do custom "percentage improvement in MAE"


# custom for paper
def compare_dr5_direct_vs_dr8_retrained():
    base_dir = '/home/walml/repos/gz-decals-classifiers/results/campaign_comparison'
    dr8_checkpoint = 'checkpoint_all_campaigns_ortho_v2_m*'
    dr5_only_checkpoint = 'checkpoint_all_campaigns_ortho_v2_train_only_dr5_m*'
    dr12_dr5_checkpoint = 'checkpoint_all_campaigns_ortho_v2_train_only_d12_dr5_m*'

    # trained on dr5 only, predicting on dr8 with dr5 head
    dr8_from_dr5_locs = glob.glob(os.path.join(base_dir, dr5_only_checkpoint, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'))
    # trained on dr12 and dr5, predicting on dr8 with dr5 head
    dr8_from_dr12_dr5_locs = glob.glob(os.path.join(base_dir, dr12_dr5_checkpoint, 'regression_errors_dr8_galaxies_with_dr5_head_for_dr8.csv'))
    # trained on all (dr12, dr5, dr8), predicting on dr8 with dr8 head
    dr8_trained_locs = glob.glob(os.path.join(base_dir, dr8_checkpoint, 'regression_errors.csv'))

    print('Locs to load: \nDR5 only: {}\nDR12 + DR5: {}\nAll: {}'.format(len(dr8_from_dr5_locs), len(dr8_from_dr12_dr5_locs), len(dr8_trained_locs)))

    dr8_from_dr5 = pd.concat([pd.read_csv(loc) for loc in dr8_from_dr5_locs], axis=0)
    dr8_from_dr12_dr5 = pd.concat([pd.read_csv(loc) for loc in dr8_from_dr12_dr5_locs], axis=0)
    dr8_trained = pd.concat([pd.read_csv(loc) for loc in dr8_trained_locs], axis=0)
   
    dr8_from_dr5 = tidy_up(dr8_from_dr5)
    dr8_from_dr12_dr5 = tidy_up(dr8_from_dr12_dr5)
    dr8_trained = tidy_up(dr8_trained)

    dr8_from_dr5['trained'] = 'dr5 only'
    dr8_from_dr12_dr5['trained'] = 'dr12 + dr5'
    dr8_trained['trained'] = 'dr12 + dr5 + dr8'

    df = pd.concat([dr8_from_dr5, dr8_from_dr12_dr5, dr8_trained], axis=0).reset_index(drop=True)

    baseline_mae_per_q_df = df.query('trained == "dr5 only"').groupby('answer').agg({'mean_absolute_error': 'mean'})
    baseline_mae_per_q = dict(zip(baseline_mae_per_q_df.index, baseline_mae_per_q_df['mean_absolute_error'].values))

    df['change_in_mae'] = df.apply(lambda x: x['mean_absolute_error'] - baseline_mae_per_q[x['answer']], axis=1)
    print(df['change_in_mae'])

    df['change_in_mae_pc'] = df['change_in_mae'] / df['mean_absolute_error']

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 10))

    colors = sns.color_palette()  # default

    sns.barplot(data=df, y='answer_clean', x='mean_absolute_error', hue='trained', ax=ax0, errwidth=0.5, palette=(colors[2], colors[0], colors[1]))
    ax0.set_xlabel('Mean Absolute Vote Frac. Error')
    ax0.set_ylabel('')

    # df[df['trained'] != 'dr5 only']
    sns.barplot(data=df, y='answer_clean', x='change_in_mae_pc', hue='trained', errwidth=0.5, palette=((1, 1, 1, 0), colors[1], colors[2]), ax=ax1)
    ax1.set_xlabel('Change in Mean Abs. Vote Frac. Error vs DR5 Only')
    ax1.set_ylabel('')
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.))

    fig.tight_layout()
    # if save_loc:
    #     plt.savefig(save_loc)
    # else:
    #     plt.show()
    plt.show()



def tidy_up(df):
    df['data_release'] = df['answer'].apply(lambda x: x.split('_')[0].split('-')[-1])
    df = df.query('data_release == "dr8"').reset_index(drop=True)
    df['answer_clean'] = df['answer'].apply(lambda x: evaluate_model.clean_text(x.replace('-dr8', ': ')))

        # only comparing the dr8 columns
    for col in df.columns.values:
        if ('dr12' in col) or ('dr5' in col):
            del df[col]

    return df




if __name__ == '__main__':

    compare_dr5_direct_vs_dr8_retrained()

    # base_dir = '/home/walml/repos/gz-decals-classifiers/results/campaign_comparison'
    # # shards = ['dr5']
    # checkpoints = [
    #     # 'all_campaigns_ortho_with_dr8',
    #     # 'all_campaigns_ortho_with_dr8_but_only_train_dr12_dr5',
    #     # 'all_campaigns_ortho_with_dr8_but_only_train_dr5',
    #     'all_campaigns_ortho_with_dr8_nl',
    #     'all_campaigns_ortho_with_dr8_nl_but_train_only_dr5'
    #     ]

    # model_indices = ['m0', 'm1', 'm2']
    # eval_dirs = []
    # # for shard in shards:
    # for checkpoint in checkpoints:
    #     for model_index in model_indices:
    #         # eval_dirs += ['shard_'+ shard + '_checkpoint_' + checkpoint + '_' + model_index]
    #         eval_dirs += ['checkpoint_' + checkpoint + '_' + model_index]

    # # print(eval_dirs)

    # eval_dirs = [os.path.join(base_dir, x) for x in eval_dirs]

    # eval_dirs = [d for d in eval_dirs if os.path.isdir(d)]
    # print(len(eval_dirs))

    # # eval_dirs = [os.path.join(base_dir, x) for x in ['dr5_pretrained_ortho', 'dr5_pretrained_ortho_dr5_only']]
    # # eval_dirs = [os.path.join(base_dir, x) for x in ['shard_dr5_checkpoint__m0', 'shard_dr5_checkpoint_all_campaigns_ortho_with_dr8_but_only_train_dr5_m0', 'shard_dr5_checkpoint_all_campaigns_ortho_with_dr8_but_only_train_dr12_dr5_m0']]
    # # compare_metric_by_question(eval_dirs, metric='mean_loss', save_loc=None)
