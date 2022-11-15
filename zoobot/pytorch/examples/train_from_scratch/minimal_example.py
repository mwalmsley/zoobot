import logging

from sklearn.model_selection import train_test_split

from galaxy_datasets import gz_decals_5

from zoobot.shared import schemas
from zoobot.pytorch.training import train_with_pytorch_lightning


if __name__ == '__main__':

    """
    This is a minimal example showing how to train Zoobot from scratch using PyTorch.
    No arguments are required. 
    
    Data is downloaded automatically - GZ DECaLS DR5, which is ~220k galaxies and ~11GB.

    - See zoobot/pytorch/examples/train_model_on_catalog for a version training on a catalog without prespecifing the splits.
    That uses command-line args for flexibility.
    - See benchmarks/pytorch/decals_dr5.sh/.py for the scripts used to create the "official" pretrained checkpoints

    """
    # you need to set these paths
    data_dir = 'TODO' # e.g. /home/data/decals_dr5
    save_dir = 'TODO' # e.g. results/minimal_model

    # only train on the first 5000 galaxies, to check if the code runs. Set False to train on all galaxies.
    debug = True

    # I picked these to run on most machines, not to train the most accurate model.
    # See benchmarks/ for real examples of args needed for most accurate model
    gpus = 1
    batch_size = 64
    resize_size = 64  # set to 64 to train quickly, ideally 224 (or more) for good performance
    num_workers = 4  # set <= num cpus

    schema = schemas.decals_dr5_ortho_schema
    logging.info('Schema: {}'.format(schema))

    # use the setup() methods in pytorch_galaxy_datasets.prepared_datasets to get the canonical (i.e. standard) train and test catalogs
    canonical_train_catalog, _ = gz_decals_5(root=data_dir, train=True, download=True)
    canonical_test_catalog, _ = gz_decals_5(root=data_dir, train=False, download=True)

    train_catalog, val_catalog = train_test_split(canonical_train_catalog, test_size=0.1)
    test_catalog = canonical_test_catalog.copy()

    # debug mode
    if debug:
        logging.warning(
            'Using debug mode: cutting catalogs down to 5k galaxies each')
        train_catalog = train_catalog.sample(5000).reset_index(drop=True)
        val_catalog = val_catalog.sample(5000).reset_index(drop=True)
        test_catalog = test_catalog.sample(5000).reset_index(drop=True)

    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir=save_dir,
        schema=schema,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        epochs=1000,  # rely on early stopping
        # augmentation parameters
        resize_size=resize_size,
        # hardware parameters
        gpus=gpus,
        prefetch_factor=4,
        num_workers=num_workers
    )
