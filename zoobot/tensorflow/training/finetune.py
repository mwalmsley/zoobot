import logging
import os

import tensorflow as tf
from tensorflow.keras import layers

from zoobot.tensorflow.training import training_config


def run_finetuning(config, encoder, train_dataset, val_dataset, test_dataset, save_dir):

    new_head = linear_classifier(config['finetune']['encoder_dim'], config['finetune']['label_dim'])

    img_size = config['finetune']['img_size']

    """
    Retrain the model. Only the new head will train as the rest is frozen.
    """


    encoder.trainable = False

    # stick the new head on the pretrained base model
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 1)),
      encoder,
      new_head
    ])

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal learning rate is okay
        metrics=['accuracy']
    )
    model.summary()

    trainer = training_config.Trainer(
        # parameters for how to train e.g. epochs, patience
        log_dir=os.path.join(save_dir, 'head_only'),
        epochs=config['finetune']['n_epochs'],
        patience=config['finetune']['patience']
    )

    model_with_trained_head = trainer.fit(
        model,
        train_dataset,
        val_dataset,
        eager=False
    )

    logging.info('Head finetuning complete')

    if config['finetune']['n_layers'] == 0:
        logging.info('n_layers = 0: not finetuning lower layers')
        return model_with_trained_head  

    logging.info('Unfreezing layers')
    # you can unfreeze layers like so:
    unfreeze_model(model, unfreeze_names=['top'])
    # or more...
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6', 'block5'])
    # utils.unfreeze_model(model, unfreeze_names=['top', 'block7', 'block6', 'block5', 'block4'])
    # utils.unfreeze_model(model, unfreeze_names=[], unfreeze_all=True)


    logging.info('Recompiling with lower learning rate and trainable upper layers')
    model_with_trained_head.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 10x lower initial learning rate (adam will adapt anyway)
        metrics=['accuracy']
    )

    model_with_trained_head.summary(print_fn=logging.info)

    trainer = training_config.Trainer(
        # parameters for how to train e.g. epochs, patience
        log_dir=os.path.join(save_dir, 'full'),
        epochs=config['finetune']['n_epochs'],
        patience=config['finetune']['patience']
    )

    model_with_trained_lower_layers = trainer.fit(
        model_with_trained_head,
        train_dataset,
        val_dataset,
        eager=False
    )
    
    logging.info('Finetuning complete')
    return model_with_trained_lower_layers
    

    
    """
    Well done!
    
    You can now use your finetuned models to make predictions on new data..
    See make_predictions.py for a self-contained example.
    """


def linear_classifier(input_dim, output_dim):
    return tf.keras.Sequential([
      # TODO move pooling
      tf.keras.layers.InputLayer(input_shape=(input_dim)),  # base model dim after GlobalAveragePooling (ignoring batch)
      tf.keras.layers.Dense(output_dim, name='logits')  # output should be N neurons w/ softmax for N-class classification
      # layers.Dense(3, activation="softmax", name="softmax_output")  # ...or 
    ])



def freeze_model(model):
    # Freeze the pretrained weights
    # inplace
    model.trainable = False


def unfreeze_model(model, unfreeze_names=['block7', 'top'], unfreeze_all=False):
    if unfreeze_all and (len(unfreeze_names) > 0):
        logging.warning('unfreeze_all is True; ignoring unfreeze_names and unfreezing all layers')
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    # required for any layer to be trainable.
    # however, setting to True sets *every* layer trainable (why, tf, why...)
    # so need to then set each layer individually trainable or not trainable below
    model.trainable = True  # everything trainable, recursively.

    for layer in model.layers:
        # recursive
        # if isinstance(layer, tf.keras.Sequential) or isinstance(layer, tf.python.keras.engine.functional.Functional):  # layer is itself a model (effnet is functional due to residual connections)
        if isinstance(layer, tf.keras.Model):  # includes subclasses Sequential and Functional
            unfreeze_model(layer, unfreeze_names=unfreeze_names, unfreeze_all=unfreeze_all)  # recursive

        elif any([layer.name.startswith(name) for name in unfreeze_names]) or unfreeze_all:
            if isinstance(layer, layers.BatchNormalization):
                logging.debug('freezing batch norm layer {}'.format(layer.name))
                layer.trainable = False
            else:
                logging.debug('unfreezing {}'.format(layer.name))
                layer.trainable = True
                # print('Freezing batch norm layer')
                # https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization?version=stable#note_that_2
                # this will also switch layer to inference mode from tf2, no need to separately pass training=False
        else:
            logging.warning('Layer {} ({}) not in unfreeze list - freezing by default'.format(layer.name, layer))
            layer.trainable = False  # not a recursive call, and not with a name to unfreeze

    # model will be trainable next time it is compiled

def check_batchnorm_frozen(model):
    for layer in model.layers:
        print(layer)
        if isinstance(layer, tf.keras.Model):
            check_batchnorm_frozen(layer)
        elif isinstance(layer, layers.BatchNormalization):
            assert not layer.trainable
            print('checks out')



    # import numpy as np
    # import pandas as pd


    # paths_pred = paths_val  # TODO for simplicitly I'll just make more predictions on the validation images, but you'll want to change this
    # raw_pred_dataset = image_datasets.get_image_dataset(paths_pred, file_format=file_format, requested_img_size=requested_img_size, batch_size=batch_size)

    # ordered_paths = [x.numpy().decode('utf8') for batch in raw_pred_dataset for x in batch['id_str']]

    # # must exactly match the preprocessing you used for training
    # pred_config = preprocess.PreprocessingConfig(
    #   label_cols=[],  # image_datasets.get_image_dataset will put the labels arg under 'label' key for each batch
    #   input_size=requested_img_size,
    #   make_greyscale=True,
    #   # normalise_from_uint8=True,
    #   permute_channels=False
    # )
    # pred_dataset = preprocess.preprocess_dataset(raw_pred_dataset, pred_config)

    # predictions = model.predict(pred_dataset)

    # data = [{'prediction': float(prediction), 'image_loc': local_png_loc} for prediction, local_png_loc in zip(predictions, ordered_paths)]
    # pred_df = pd.DataFrame(data=data)
  
    # example_predictions_loc = 'results/finetune_minimal/example_predictions.csv'
    # pred_df.to_csv(example_predictions_loc, index=False)
    # logging.info(f'Example predictions saved to {example_predictions_loc}')