import pytest

from zoobot.pytorch.estimators import define_model
from zoobot.shared import schemas

@pytest.fixture
def schema():
    return schemas.decals_dr8_ortho_schema

def test_ZoobotTree_init(schema):
    model = define_model.ZoobotTree(
        output_dim=12,
        question_answer_pairs=schema.question_answer_pairs,
        dependencies=schema.dependencies
    )

