.. _representations_guide:

Representations
===============

Representations are vectors that describe your input data, usually much lower-dimensional than the original input data.
You might like to extract the representations learned by the model. Here's how.

If you would like the representations of the trained GZ DECaLS model on the DECaLS DR5 galaxies, you can find them here (TODO Zenodo link).
These were used for the morphology tools paper. If you need your own model and representations, read on.

## Training a New Model

Do this exactly like you normally would. It will work better on a broad multi-question task, like answering the GZ decision tree.

See :ref:`reproducing_decals` for a guide to training a new model.

## Extracting the representation

Extracting the representation is really just making a prediction, but without the final layers of the model.
Run ``make_predictions_loop.py``, configuring the model (by commenting) like so:

- The base model should have ``include_top=False`` (we don't want the final dense layer)
- Add a new top with just the global pooling
- Group the base model and new top with ``tf.keras.Sequential``
- Set the ``label_cols`` to be as long as the dimension of your representation (e.g. 1280) rather than the usual answers (e.g. 34)

As always, remember to check ``run_name`` and any file paths.

``make_predictions_loop.py`` will then save the representations for each galaxy to files like {run_name}_{index}.csv.
These files are a bit awkward as they include lots of numbers like ``[[0.4, ...]]``.
Remove the brackets with ``predictions/reformat_predictions.py``.

Finally, compress the 1280-dim representation using PCA with ``representations/compress_representations.py``.
