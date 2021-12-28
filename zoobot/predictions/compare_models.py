import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_loss_by_question(eval_dirs, save_loc=None):

    data_by_dir = [pd.read_csv(os.path.join(d, 'val_loss_by_q.csv')) for d in eval_dirs]
    for (df, d) in zip(data_by_dir, eval_dirs):
        df['eval_dir'] = os.path.basename(d)  # modify by ref
        # TODO temp
        for col in df.columns.values:
            if 'dr12' in col:
                del df[col]
    data = pd.concat(data_by_dir)
                                
    sns.set_style('whitegrid', {'axes.edgecolor': '0.2'})
    sns.set_context('notebook')
    # sns.set_palette(repeating_palette)
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(data=data, y='question', x='mean_loss', hue='eval_dir', ax=ax)
    plt.xlabel('Mean Loss')
    plt.ylabel('')
    fig.tight_layout()
    if save_loc:
        plt.savefig(save_loc)
    else:
        plt.show()


if __name__ == '__main__':

    base_dir = '/home/walml/repos/gz-decals-classifiers/results/campaign_comparison'
    eval_dirs = [os.path.join(base_dir, x) for x in ['dr5_pretrained_ortho', 'dr5_pretrained_ortho_dr5_only']]
    compare_loss_by_question(eval_dirs, save_loc=None)
