# import json
# import logging
# from typing import List

# import numpy as np
# import pandas as pd
# import tensorflow as tf


# def predict(tfrecord_ds: tf.data.Dataset, model: tf.keras.Model, n_samples: int, label_cols: List, save_loc: str):

#     logging.info('Reading id_str from records')
#     id_strs = []
#     for _, id_str_batch in tfrecord_ds:
#         id_strs += id_str_batch
#     logging.info(id_strs)

#     logging.info('Beginning predictions')
#     # predictions must fit in memory
#     predictions = np.stack([model.predict(tfrecord_ds) for n in range(n_samples)], axis=-1)
#     logging.info('Predictions complete - {}'.format(predictions.shape))

#     data = [prediction_to_row(predictions[n], id_strs[n], label_cols) for n in range(len(predictions))]
#     predictions_df = pd.DataFrame(data)

#     predictions_df.to_csv(save_loc, index=False)
#     logging.info(f'Predictions saved to {save_loc}')

#     end = datetime.datetime.fromtimestamp(time.time())
#     logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
#     logging.info('Time elapsed: {}'.format(end - start))


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


"""Now refactored into predict_on_dataset (both images and tfrecord - any dataset)"""