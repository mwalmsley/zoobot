# Zoobot

*Classify detailed galaxy morphology with Bayesian CNN.*

This repo contains all the code needed to train Bayesian CNN to predict galaxy morphology.

We hope this is useful to:

- Reproduce and improve upon the automated classifications published as Galaxy Zoo DECaLS
- Finetune the CNN for new surveys and/or for other morphology-related tasks such as identifying tidal features or segmenting spiral arms.

If you use this repo for your research, please cite the paper.

---

## Reproducing the DECaLS Classifications

This code was used to create the automated classifications for GZ DECaLS.
It can be re-used for new Galaxy Zoo projects or as a baseline or starting point to improve on our performance.

You will need galaxy images and volunteer classifications.
For GZD-5, these are available at [https://zenodo.org/record/4196267].
You will also need a fairly good GPU - we used an NVIDIA V100. You might get away with a somewhat worse GPU by lowering the batch size (we used 128) or the image size, but this may reduce performance.

To create a CNN:

- Define the decision tree asked of volunteers in `schemas.py`. *Skip if using GZD-5.*
- Prepare a catalog with your images and labels (matching the decision tree)
- Create TFRecord shards (groups of images encoded for fast reading) from your catalog with `create_shards.py`
- Train the CNN on those shards with `train_model.py`.

Galaxy Zoo uses a decision tree where the questions asked depend upon the previous answers.
The decision tree is defined under `schemas.py` and `label_metadata.py`.
The GZ2 and GZ DECaLS decision trees are already defined for you; for other projects, you'll need to define your own (it's easy, just follow the same pattern).

Create a catalog with all your labelled galaxies.
The catalog should be a csv file with rows of (unique) galaxies and columns including:

- id_str, a string that uniquely identifies each galaxy (e.g. the iauname)
- file_loc, the absolute location of the galaxy image on disk. This is expected to be a .png of any size, but you could easily extend it for other filetypes if needed.
- a column with the total votes for each label you want to predict, matching the schema (above).  For GZD-5, this is e.g. smooth-or-featured_smooth, smooth-or-featured_featured-or-disk, etc.

It's quite slow to train a model using normal images, and so we first encode them as several TFRecords, a format which is much faster to read.
Make these with create_shards.py, passing in your catalog location and where the TFRecords should be placed e.g.

    python create_shards.py --labelled-catalog path/to/my_catalog.csv --shard-dir folder/for/shards --img-size 300  --eval-size 5000

More options are available; see `data_utils/create_shards.py`.

Now you can train a CNN using those shards. `training/training_config.py` has the code to do this. Use it in your own code like so:

    from zoobot.training import training_config

    run_config = training_config.get_run_config(
      initial_size=shard_img_size,
      final_size=final_size,  # size after augmentations
      crop_size=int(shard_img_size * 0.75),  # 75% zoom
      log_dir=save_dir,
      train_records=train_records,  # shards to train on
      eval_records=eval_records,  # shards to validate on
      epochs=epochs,
      schema=schema,  # decision tree schema from schemas.py
      batch_size=batch_size
    )

    trained_model = run_config.run_estimator()  # train!
    trained_model.save_weights(save_loc)

There is a complete working example at `train_model.py` which you can copy and adapt.

Once trained, the model can be used to make new predictions on either folders of images (png, jpeg) or TFRecords. For example:

    folder_to_predict = '/media/walml/beta/decals/png_native/dr5/J000'
    file_format = 'png'  # jpg or png supported. FITS is NOT supported (PRs welcome)
    predict_on_images.predict(
        schema=schema,
        file_format=file_format,
        folder_to_predict=folder_to_predict,
        checkpoint_dir=checkpoint_dir,
        save_loc=save_loc,
        n_samples=n_samples,  # number of dropout forward passes
        batch_size=batch_size,
        initial_size=initial_size,
        crop_size=crop_size,
        final_size=final_size
    )

There is a complete working example at `make_predictions.py`.

Note that in the DECaLS paper, we only used galaxies classified in GZD-5 even for questions which did not change between GZD-1/2 and GZD-5.
It would be straightforward (and appreciated) to retrain the models using GZD-1/2 classifications as well, to improve performance.

## Fine-Tuning our Pre-Trained CNN

Fine-tuning, aka transfer learning, is when a model trained on one task is partially retrained for use on another related task (or similar data).
Hopefully, the model will have learned an effective representation of the data from the first task, and so might perform better on the related task than a model trained from scratch.
For a general introduction, see the excellent blog post at [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html].

The CNN have been trained to solve a variety of morphology tasks (i.e. to simultaneously answer all of the Galaxy Zoo questions) and so is likely to have learned a useful general representation of galaxy morphology.
You can benefit from this general representation by fine-tuning the CNN for your task.

The general approach is:
1. Load the pretrained model, replacing the head (output layers) to match your task
2. Train *only* the new head, leaving the rest of the model frozen
3. Optionally, once the new head is trained, unfreeze the rest of the model and train with a low learning rate

We include some utilities to make fine-tuning easy.

https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

The CNN architecture we use, EfficientNet, is a bit complicated


The Dirichlet-Multinomial loss function is designed for Galaxy Zoo and may not be appropriate for your task. 


We provide pretrained CNN
, as reported by Galaxy Zoo volunteers

Th


