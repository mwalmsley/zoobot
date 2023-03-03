import pytest

from zoobot.pytorch.training import train_with_pytorch_lightning
from zoobot.shared import schemas

from galaxy_datasets.shared.demo_gz_candels import demo_gz_candels

@pytest.fixture
def schema():
    return schemas.gz_candels_ortho_schema


@pytest.fixture
def train_catalog(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    train_catalog, _ = demo_gz_candels(root=data_dir, download=True, train=True)
    return train_catalog


def test_train_rings(tmp_path, schema, train_catalog):

    save_dir = tmp_path / "save_dir"
    save_dir.mkdir()

    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir=save_dir,
        schema=schema,
        catalog=train_catalog,
        epochs=1,
        gpus=0,
        batch_size=32,
        wandb_logger=None
    )
