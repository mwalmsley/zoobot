import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from zoobot.tensorflow.training import training_config
from zoobot.tensorflow.transfer_learning import utils  # TODO move


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
    utils.unfreeze_model(model, unfreeze_names=['top'])
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
      layers.InputLayer(input_shape=(input_dim)),  # base model dim after GlobalAveragePooling (ignoring batch)
      layers.Dense(output_dim, name='logits')  # output should be N neurons w/ softmax for N-class classification
      # layers.Dense(3, activation="softmax", name="softmax_output")  # ...or 
    ])


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