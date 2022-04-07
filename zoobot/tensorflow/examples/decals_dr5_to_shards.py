import argparse
import logging

import pandas as pd

from zoobot.tensorflow.data_utils import create_shards
from zoobot.shared import label_metadata

if __name__ == '__main__':

    """
    Some example commands:

    DECALS:

        (debugging)
        python decals_dr5_to_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2/unlabelled_catalog.csv --eval-size 100 --shard-dir=data/decals/shards/decals_debug --max-labelled 500 --max-unlabelled=300 --img-size 32

        (the actual commands used for gz decals: debug above, full below)
        python decals_dr5_to_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_retired/unlabelled_catalog.csv --eval-size 100 --shard-dir=data/decals/shards/decals_debug --max-labelled 500 --max-unlabelled=300 --img-size 32
        python decals_dr5_to_shards.py --labelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2_arc/labelled_catalog.csv --unlabelled-catalog=data/decals/prepared_catalogs/all_2p5_unfiltered_n2_arc/unlabelled_catalog.csv --eval-size 10000 --shard-dir=data/decals/shards/all_2p5_unfiltered_n2  --img-size 300

    GZ2:
        python decals_dr5_to_shards.py --labelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/labelled_catalog.csv --unlabelled-catalog=data/gz2/prepared_catalogs/all_featp5_facep5/unlabelled_catalog.csv --eval-size 1000 --shard-dir=data/gz2/shards/all_featp5_facep5_256 --img-size 256

    """

    parser = argparse.ArgumentParser(description='Make shards')

    # you should have already made these catalogs for your dataset
    parser.add_argument('--labelled-catalog', dest='labelled_catalog_loc', type=str,
                    help='Path to csv catalog of previous labels and file_loc, for shards')
    parser.add_argument('--unlabelled-catalog', dest='unlabelled_catalog_loc', type=str, default='',
                help='Path to csv catalog of previous labels and file_loc, for shards. Optional - skip (recommended) if all galaxies are labelled.')

    parser.add_argument('--eval-size', dest='eval_size', type=int,
        help='Split labelled galaxies into train/test, with this many test galaxies (e.g. 5000)')

    # Write catalog to shards (tfrecords as catalog chunks) here
    parser.add_argument('--shard-dir', dest='shard_dir', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--max-unlabelled', dest='max_unlabelled', type=int,
                    help='Max unlabelled galaxies (for debugging/speed')
    parser.add_argument('--max-labelled', dest='max_labelled', type=int,
                    help='Max labelled galaxies (for debugging/speed')
    parser.add_argument('--img-size', dest='size', type=int,
                    help='Size at which to save images (before any augmentations). 300 for DECaLS paper.')

    args = parser.parse_args()

    # log_loc = 'make_shards_{}.log'.format(time.time())
    logging.basicConfig(
        # filename=log_loc,
        # filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

    logging.info('Using GZ DECaLS label schema by default - see create_shards.py for more options, or to add your own')
    # label_cols = label_metadata.decals_partial_label_cols
    label_cols = label_metadata.decals_label_cols
    # label_cols = label_metadata.gz2_partial_label_cols
    # label_cols = label_metadata.gz2_label_cols

    # labels will always be floats, int conversion confuses tf.data
    dtypes = dict(zip(label_cols, [float for _ in label_cols]))
    dtypes['id_str'] = str
    labelled_catalog = pd.read_csv(args.labelled_catalog_loc, dtype=dtypes)
    if args.unlabelled_catalog_loc is not '':
        unlabelled_catalog = pd.read_csv(args.unlabelled_catalog_loc, dtype=dtypes)
    else:
        unlabelled_catalog = None

    # limit catalogs to random subsets
    if args.max_labelled:
        labelled_catalog = labelled_catalog.sample(len(labelled_catalog))[:args.max_labelled]
    if args.max_unlabelled and (unlabelled_catalog is not None):  
        unlabelled_catalog = unlabelled_catalog.sample(len(unlabelled_catalog))[:args.max_unlabelled]

    logging.info('Labelled catalog: {}'.format(len(labelled_catalog)))
    if unlabelled_catalog is not None:
        logging.info('Unlabelled catalog: {}'.format(len(unlabelled_catalog)))

    # in memory for now, but will be serialized for later/logs
    train_test_fraction = create_shards.get_train_test_fraction(len(labelled_catalog), args.eval_size)

    labelled_columns_to_save = ['id_str'] + label_cols
    logging.info('Saving columns for labelled galaxies: \n{}'.format(labelled_columns_to_save))

    shard_config = create_shards.ShardConfig(shard_dir=args.shard_dir, size=args.size)

    shard_config.prepare_shards(
        labelled_catalog,
        unlabelled_catalog,
        train_test_fraction=train_test_fraction,
        labelled_columns_to_save=labelled_columns_to_save
    )
