

import os
import logging
import argparse
import time
import json

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from zoobot.tensorflow.estimators import preprocess, define_model
from zoobot.tensorflow.training import training_config
from zoobot.tensorflow.transfer_learning import utils
from zoobot.tensorflow.datasets import rings

    
def main(batch_size, requested_img_size, train_dataset_size, max_galaxies_to_show=5000000, greyscale=True):
    """

    This example is the research-grade version of finetune_minimal.py. Start there first.
    I used this code for the morphology tools paper.

    You can use this example to do various finetuning tasks by commenting/uncommenting the blocks you need:
    - Train a new head on a frozen model
    - Finetune a new head on a trained head/frozen model pair
    - Initialise the frozen model with either GZ DECaLS weights (note: channels=1!) or ImageNet weights (note: channels=3!) or randomly
    """

    """  
    Set up your finetuning dataset.
    
    Here, I'm using galaxies tagged or not tagged as "ring" by Galaxy Zoo volunteers.

    This time, I'm using the "advanced" rings dataset. 
    Instead of labelling galaxies as 1 if tagged ring or 0 otherwise, 
    this calculates labels based on the GZ DECaLS "Are there any of these rare features?" "Ring" answer vote fraction.

    Also, instead of balancing the classes by dropping most of the non-rings, I'm repeating the rings by a factor of ~5
    This includes more information but needs some footwork to make sure that no repeated ring ends up in both the train and test sets.py

    Lastly, the dataset can optionally be restricted to have a smaller train and validation dataset size.
    I used this in a paper to test how the number of available labels changed the performance of the finetuned models.

    """

    if greyscale:
      channels = 1
    else:
      logging.warning('greyscale = False, hopefully you expect to use imagenet or a color-trained model')
      channels = 3
      logging.warning('Using color images, channels=3')


    raw_train_dataset, raw_val_dataset, raw_test_dataset = rings.get_advanced_ring_image_dataset(
      batch_size=batch_size, requested_img_size=requested_img_size, train_dataset_size=train_dataset_size)

    # small datasets that fit in memory can be cached before augmentations
    # this speeds up training
    use_cache = False  # sequential if's for awkward None/int type
    if train_dataset_size is not None:  # when None, load all -> very many galaxies
      if train_dataset_size < 10000:
        use_cache = True

    if use_cache:
      raw_train_dataset = raw_train_dataset.cache()
      raw_val_dataset = raw_val_dataset.cache()
      raw_test_dataset = raw_test_dataset.cache()
      # read once (and don't use) to trigger the cache
      _ = [x for x in raw_train_dataset.as_numpy_iterator()]
      _ = [x for x in raw_val_dataset.as_numpy_iterator()]
      _ = [x for x in raw_test_dataset.as_numpy_iterator()]
      logging.info('Cache complete')
    else:
      logging.warning('Requested {} training images (if None, using all available). Skipping cache.'.format(train_dataset_size))
    
    preprocess_config = preprocess.PreprocessingConfig(
        label_cols=['label'],  # image_datasets.get_image_dataset will put the labels arg under the 'label' key for each batch
        input_size=requested_img_size,
        normalise_from_uint8=True,  # divide by 255
        make_greyscale=greyscale,  # take the mean over RGB channels
        permute_channels=False  # swap channels around randomly (no need when making greyscale anwyay)
    )
    train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
    val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)
    test_dataset = preprocess.preprocess_dataset(raw_test_dataset, preprocess_config)

    # Must match how the model was trained - do not change.
    crop_size = int(requested_img_size * 0.75)  # implies cropping the 300 pixel images to 225 pixels. However, the code will ignore this and crop directly to resize_size (below) as they are very similar
    resize_size = 224  # code will skip resizing and crop straight to 224. Don't change - must be 224 for both DECaLS and ImageNet

    run_name = 'example_run_{}'.format(time.time())  # scratch was very likely trained in colour
    log_dir = os.path.join('results/example_run', run_name)
    log_dir_head = os.path.join(log_dir, 'head_only')
    for d in [log_dir, log_dir_head]:
      if not os.path.isdir(d):
        os.mkdir(d)

    """
    In the next few code blocks, we will load a base model (without the "head" output layer), freeze it, and add a new head.
    """

    """Pick a base model"""

    # get base model from pretrained *DECaLS* checkpoint (includes augmentations)
    pretrained_checkpoint = 'data/pretrained_models/decals_dr_trained_on_all_labelled_m0/in_progress'
    # pretrained_checkpoint = 'data/pretrained_models/decals_dr_train_set_only_replicated/checkpoint'
    ## a few other checkpoints used in the representations paper, trained on single questions - happy to share on request, but lower performance than the above
    # pretrained_checkpoint = '/share/nas2/walml/repos/gz-decals-classifiers/results/replicated_train_only_smooth_only/checkpoint'  # single task smooth
    # pretrained_checkpoint = '/share/nas2/walml/repos/gz-decals-classifiers/results/replicated_train_only_bar_only/checkpoint'
    # pretrained_checkpoint = '/share/nas2/walml/repos/gz-decals-classifiers/results/replicated_train_only_bulge_size_only/checkpoint'
    # pretrained_checkpoint = '/share/nas2/walml/repos/gz-decals-classifiers/results/replicated_train_only_spiral_yn_only/checkpoint'

    base_model = get_headless_model(
      pretrained_checkpoint,
      requested_img_size,
      crop_size,
      resize_size,
      channels,  # careful, 1 for decals checkpoint or 3 for imagenet checkpoint
      expect_partial=True
    ) 
    base_model.trainable = False # freeze the headless model (no training allowed)

    # # OR get base model from pretrained *ImageNet* checkpoint (includes augmentations)
    # base_model = define_model.get_model(
    #   output_dim=None,
    #   input_size=requested_img_size,
    #   crop_size=crop_size,
    #   resize_size=resize_size,
    #   weights_loc=None,
    #   include_top=False,
    #   channels=channels,
    #   use_imagenet_weights=True
    # )
    # base_model.trainable = False  # freeze the headless model (no training allowed)

    # # OR create a blank base model from scratch (it will need training)
    # base_model = define_model.get_model(
    #   output_dim=None,
    #   input_size=requested_img_size,
    #   crop_size=crop_size,
    #   resize_size=resize_size,
    #   weights_loc=None,
    #   include_top=False,
    #   channels=channels,
    #   use_imagenet_weights=False
    # )
    # base_model.trainable = True

    
    """Add the small (trainable) dense head"""
    # I am not using test-time dropout (MC Dropout) on the head as 0.75 would be way too aggressive and reduce performance
    new_head = tf.keras.Sequential([
      layers.InputLayer(input_shape=(7,7,1280)),  # base model dim before GlobalAveragePooling (ignoring batch)
      layers.GlobalAveragePooling2D(),  # quirk of code that this is included - could have been input_shape=(1280) and skipped GAP2D
      # TODO the following layers will likely need some experimentation to find a good combination for your problem
      # layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.75),
      layers.Dense(1, activation="sigmoid", name='sigmoid_output')  # output should be one neuron w/ sigmoid for binary classification...
      # layers.Dense(3, activation="softmax", name="softmax_output")  # ...or N neurons w/ softmax for N-class classification
    ])

    # stick the new head on the pretrained base model
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(requested_img_size, requested_img_size, channels)),
      base_model,
      new_head
    ])


    # Retrain the model. If you froze the base model, only the new head will train (fast). Otherwise, the whole model will train (slow).

    epochs = max(int(max_galaxies_to_show / train_dataset_size), 1)
    patience = min(max(10, int(epochs/6)), 30)  # between 5 and 30 epochs, sliding depending on num. epochs (TODO may just set at 30, we'll see)
    # patience = 1  # TODO
    logging.info('Epochs: {}'.format(epochs))
    logging.info('Early stopping patience: {}'.format(patience))

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )
    model.summary(print_fn=logging.info)

    train_config = training_config.TrainConfig(
      log_dir=log_dir_head,
      epochs=epochs,
      patience=patience  # early stopping: if val loss does not improve for this many epochs in a row, end training
    )

    training_config.train_estimator(
      model,
      train_config,  # e.g. how to train epochs, patience
      train_dataset,
      val_dataset
    )

    evaluate_performance(
      model=model,test_dataset=test_dataset, run_name=run_name + '_transfer', log_dir=log_dir, batch_size=batch_size, train_dataset_size=train_dataset_size
    )

    """
    If you trained the whole model (i..e base_model.trainable=True), you can stop here. Otherwise, carry on if you want to finetune.

    The head has been retrained.
    It may be possible to further improve performance by unfreezing the layers just before the head in the base model,
    and training with a very low learning rate (to avoid overfitting).

    If you want to focus on this step, you can comment out the training above and instead simply load the previous model (including the finetuned head).
    """
    # define_model.load_weights(model=model, checkpoint_loc=os.path.join(log_dir_head, 'checkpoint'), expect_partial=True)
    
    logging.info('Unfreezing layers')
    # you can unfreeze layers like so:
    utils.unfreeze_model(model, unfreeze_names=['top'])
    # or more...
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6', 'block5'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6', 'block5', 'block4'])
    # utils.unfreeze_model(model, unfreeze_names=[], unfreeze_all=True)
    # note that the number of free parameters increases very quickly!

    logging.info('Recompiling with lower learning rate and trainable upper layers')
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 10x lower initial learning rate (adam will adapt anyway)
        metrics=['accuracy']
    )

    model.summary(print_fn=logging.info)

    log_dir_full = os.path.join(log_dir, 'full')
    train_config_full = training_config.TrainConfig(
      log_dir=log_dir_full,
      epochs=epochs,
      patience=patience
    )
  
    training_config.train_estimator(
      model,  # inplace
      train_config_full,
      train_dataset,
      val_dataset
    )

    logging.info('Finetuning complete')

    evaluate_performance(
      model=model, test_dataset=test_dataset,run_name=run_name + '_finetuned',log_dir=log_dir,batch_size=batch_size, train_dataset_size=train_dataset_size
    )

  

def get_headless_model(pretrained_checkpoint, requested_img_size, crop_size, resize_size, channels, expect_partial=False):
  # get headless model (inc. augmentations)
  logging.info('Loading pretrained model from {}'.format(pretrained_checkpoint))
  return define_model.load_model(
    pretrained_checkpoint,
    include_top=False,  # do not include the head used for GZ DECaLS - we will add our own head
    input_size=requested_img_size,  # the preprocessing above did not change size
    crop_size=crop_size,  # model augmentation layers apply a crop...
    resize_size=resize_size,  # ...and then apply a resize
    output_dim=None , # headless so no effect
    channels=channels,
    expect_partial=expect_partial
  )


def evaluate_performance(model, test_dataset, run_name, log_dir, batch_size, train_dataset_size):
    # evaluate performance on test set, repeating to marginalise over any test-time augmentations or dropout
    losses = []
    accuracies = []
    for _ in range(5):
      test_metrics = model.evaluate(test_dataset.repeat(3), verbose=0)
      losses.append(test_metrics[0])
      accuracies.append(test_metrics[1])
    logging.info('Mean test loss: {:.3f} (var {:.4f})'.format(np.mean(losses), np.var(losses)))
    logging.info('Mean test accuracy: {:.3f} (var {:.4f})'.format(np.mean(accuracies), np.var(accuracies)))

    predictions = model.predict(test_dataset).astype(float).squeeze()  # only works for 1D prediction array
    logging.info(predictions)
    labels = np.concatenate([label_batch.numpy().astype(float) for _, label_batch in test_dataset]).squeeze()
    logging.info(labels)
    results = {
      'batch_size': int(batch_size),
      'mean_loss': float(np.mean(losses)),
      'mean_acc': float(np.mean(accuracies)),
      'predictions': predictions.tolist(),
      'labels': labels.tolist(),
      'train_dataset_size': int(train_dataset_size),
      'log_dir': log_dir,
      'run_name': str(os.path.basename(log_dir))
    }
    json_name = '{}_result_timestamped_{}_{}.json'.format(run_name, train_dataset_size, np.random.randint(10000))
    json_loc = os.path.join(log_dir, json_name)

    with open(json_loc, 'w') as f:
      json.dump(results, f)

    logging.info(f'Results saved to {json_loc}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    # useful to avoid errors on small GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      
    parser = argparse.ArgumentParser(description='Transfer learning and finetuning from pretrained model on ring dataset')
    parser.add_argument('--dataset-size', dest='train_dataset_size', default=None, type=int,
                        help='Max labelled galaxies to use (including resampling)')
    parser.add_argument('--batch-size', dest='batch_size', default=256, type=int,
                        help='Batch size to use for train/val/test of model')
    parser.add_argument('--img-size', dest='requested_img_size', default=300, type=int,
                        help='Image size before conv layers i.e. after loading (from 424, by default) and cropping (to 300, by default).')

    args = parser.parse_args()

    main(
      batch_size=args.batch_size,
      requested_img_size=args.requested_img_size,
      train_dataset_size=args.train_dataset_size,
      greyscale=True
    )


      

"""
---
I also tried with this small CNN rather than EfficientNet. It performed very poorly.
"""

# model = get_small_cnn(
#   input_size=requested_img_size,
#   crop_size=crop_size,
#   resize_size=resize_size,
#   size_after_preprocessing=resize_size,  # code will crop straight to resize_size when crop_size ~= resize_size
#   channels=channels
# )

# def get_small_cnn(input_size, crop_size, resize_size, size_after_preprocessing, channels):
#   # tf2 version of
#   # https://github.com/mwalmsley/tidalclassifier/blob/master/tidalclassifier/cnn/individual_cnn/meta_CNN.py#L33 

#     model = tf.keras.Sequential()

#     input_shape = (input_size, input_size, channels)
#     model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

#     define_model.add_augmentation_layers(
#       model, crop_size=crop_size, resize_size=resize_size)  # inplace

#     small_cnn = tf.keras.Sequential([
#       tf.keras.layers.InputLayer((size_after_preprocessing, size_after_preprocessing, channels)),

#       tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

#       tf.keras.layers.Conv2D(32, (3, 3)),
#       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

#       tf.keras.layers.Conv2D(64, (3, 3)),
#       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(64, activation='relu'),  # original did not have relu
#       tf.keras.layers.Dropout(0.5),
#       tf.keras.layers.Dense(1, activation="sigmoid")
#     ])

#     model.add(small_cnn)

#     model.summary(print_fn=logging.info)

#     return model
