import logging

from itertools import islice
import glob

from PIL import Image  # necessary to avoid PIL.Image error assumption in web_datasets

from galaxy_datasets.shared import label_metadata
from galaxy_datasets import gz2


from zoobot.pytorch.datasets import webdataset_utils


def main():

    logging.basicConfig(level=logging.INFO)

    train_catalog, _ = gz2(root='/home/walml/repos/zoobot/only_for_me/narval/temp', download=True, train=True)

    divisor = 4096
    n_shards = len(train_catalog) // divisor
    logging.info(n_shards)

    train_catalog = train_catalog[:n_shards*divisor]
    logging.info(len(train_catalog))
    label_cols = label_metadata.gz2_ortho_label_cols

    save_loc = "/home/walml/repos/zoobot/only_for_me/narval/gz2/gz2_train.tar"
    
    webdataset_utils.df_to_wds(train_catalog, label_cols, save_loc, n_shards=n_shards)

    # webdataset_utils.load_wds_directly(save_loc)

    # webdataset_utils.load_wds_with_augmentation(save_loc)

    webdataset_utils.load_wds_with_webdatamodule(save_loc, label_cols)


if __name__ == '__main__':

    main()

