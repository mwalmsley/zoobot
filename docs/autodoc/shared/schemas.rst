.. _schemas:

schemas
===================

This module contains classes to handle the Galaxy Zoo decision tree - Schema, Question, and Answer - and functions to help link them.
This is crucial for both calculating the custom loss and analysing the predictions in practice.

See :ref:`training_on_vote_counts` for full explanation.


.. autoclass:: zoobot.shared.schemas.Question

|

.. autoclass:: zoobot.shared.schemas.Answer

|

.. autofunction:: zoobot.shared.schemas.create_answers

|

.. autofunction:: zoobot.shared.schemas.set_dependencies

|

.. autoclass:: zoobot.shared.schemas.Schema
    :members:

|