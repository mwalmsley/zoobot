import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from zoobot.shared import schemas


def get_gz_decals_dr5_benchmark_dataset(data_dir, random_state, download):
    # use the setup() methods in galaxy_datasets.prepared_datasets to get the canonical (i.e. standard) train and test catalogs

    from galaxy_datasets import gz_decals_5  # public

    canonical_train_catalog, _ = gz_decals_5(root=data_dir, train=True, download=download)
    canonical_test_catalog, _ = gz_decals_5(root=data_dir, train=False, download=download)

    train_catalog, val_catalog = train_test_split(canonical_train_catalog, test_size=0.1, random_state=random_state)
    test_catalog = canonical_test_catalog.copy()

    schema = schemas.decals_dr5_ortho_schema
    logging.info('Schema: {}'.format(schema))

    return schema, (train_catalog, val_catalog, test_catalog)


def get_gz_evo_benchmark_dataset(data_dir, random_state, download=False, debug=False, datasets=['gz_desi', 'gz_hubble', 'gz_candels', 'gz2', 'gz_rings', 'gz_cosmic_dawn']):

    from foundation.datasets import mixed  # not yet public. import will fail if you're not me.

    # temporarily, everything *but* hubble, for Ben
    # datasets = ['gz_desi', 'gz_candels', 'gz2', 'gz_rings']

    # TODO temporarily no cache, to remake
    direct_label_cols, (temp_train_catalog, temp_val_catalog, _) = mixed.everything_all_dirichlet_with_rings(data_dir, debug, download=download, use_cache=True, datasets=datasets)
    canonical_train_catalog = pd.concat([temp_train_catalog, temp_val_catalog], axis=0)

    # here I'm going to ignore the test catalog
    train_catalog, hidden_catalog = train_test_split(canonical_train_catalog, test_size=1./3., random_state=random_state)
    val_catalog, test_catalog = train_test_split(hidden_catalog, test_size=2./3., random_state=random_state)

    schema = mixed.mixed_schema()
    assert len(direct_label_cols) == len(schema.label_cols), ValueError((len(direct_label_cols), len(schema)))
    logging.info('Schema: {}'.format(schema))
    return schema, (train_catalog, val_catalog,test_catalog)
