# import os
# import glob
# import json
# import logging

# import numpy as np
# import pandas as pd
# import tensorflow as tf

# from zoobot import label_metadata
# from zoobot.data_utils import tfrecord_datasets
# from zoobot.training import losses, training_config
# from zoobot.estimators import preprocess, define_model


# def prediction_to_row(prediction, id_str, label_cols):
#     assert len(label_cols) > 0 # use [] to load tfrecord without label cols, but need to know label cols when saving predictions
#     row = {
#         'id_str': id_str
#     }
#     for n in range(len(label_cols)):
#         answer = label_cols[n]
#         row[answer + '_concentration'] = json.dumps(list(prediction[n].astype(float)))
#         # row[answer + '_concentration_mean'] = float(prediction[n].mean())  # not meaningful, use DirichletMixture instead
#     return row


# def predict(label_cols, tfrecord_locs, checkpoint_dir, save_loc, n_samples, batch_size, initial_size, crop_size, resize_size, channels=3):

#     raw_dataset = tfrecord_datasets.get_dataset(
#         tfrecord_locs,
#         label_cols=[],
#         batch_size=batch_size,
#         shuffle=False,
#         repeat=False,
#         drop_remainder=False
#     )
#     id_strs_batched = [batch['id_str'] for batch in raw_dataset]
#     id_strs = [id_str.numpy().decode('utf-8') for batch in id_strs_batched for id_str in batch]

#     input_config = preprocess.PreprocessingConfig(
#         label_cols=[],  # no labels
#         input_size=initial_size,
#         greyscale=True,
#         channels=3
#     )
#     dataset = preprocess.preprocess_dataset(raw_dataset, input_config)

#     model = define_model.load_model(
#         checkpoint_dir=checkpoint_dir,
#         include_top=True,
#         input_size=initial_size,
#         crop_size=crop_size,
#         resize_size=resize_size,
#         expect_partial=True
#     )

#     logging.info('Beginning predictions')
#     # predictions must fit in memory
#     predictions = np.stack([model.predict(dataset) for n in range(n_samples)], axis=-1)
#     logging.info('Predictions complete - {}'.format(predictions.shape))

#     data = [prediction_to_row(predictions[n], id_strs[n], label_cols) for n in range(len(predictions))]
#     predictions_df = pd.DataFrame(data)

#     predictions_df.to_csv(save_loc, index=False)
#     logging.info(f'Predictions saved to {save_loc}')
