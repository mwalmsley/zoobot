# import logging
# from typing import Dict

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt




# # currently only used in tests
# def load_examples_from_tfrecord(tfrecord_locs, feature_spec, n_examples=None, max_examples=1e8):
#     dataset = load_dataset(tfrecord_locs, feature_spec)
#     dataset = dataset.batch(1)  # 1 image per batch
#     dataset = dataset.prefetch(1)

#     if n_examples is None:  # load full record
#         data = []
#         for batch in dataset:
#             data.append(batch)
#     else:  # load exactly n examples, or throw an error
#         logging.debug('Loading the first {} examples from {}'.format(n_examples, tfrecord_locs))
#         data = [batch for batch in dataset.take(n_examples)]

#     return data


# def matrix_feature_spec(size, channels):  # used for predict mode
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')

# def matrix_label_feature_spec(size, channels, float_label=True):
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')




# def matrix_label_counts_feature_spec():
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')


# def id_label_counts_feature_spec():
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')


# def matrix_id_feature_spec(size, channels):
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')


# def matrix_label_id_feature_spec(size, channels):
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')


# def id_feature_spec():
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')


# def id_label_feature_spec():
#     raise NotImplementedError('This has been deprecated - use get_feature_spec instead')


# not required, use tf.parse_single_example directly
# def parse_example(example, size, channels):
#     features = {
#         'matrix': tf.FixedLenFeature((size * size * channels), tf.float32),
#         'label': tf.FixedLenFeature([], tf.int64),
#         }

#     return tf.parse_single_example(example, features=features)


# # these are actually not related to reading a tfrecord, they are very general
# def show_examples(examples, size, channels):
#     # simple wrapper for pretty example plotting
#     # TODO make plots in a grid rather than vertical column
#     fig, axes = plt.subplots(nrows=len(examples), figsize=(4, len(examples) * 3))
#     for n, example in enumerate(examples):
#         show_example(example, size, channels, ax=axes[n])
#     fig.tight_layout()
#     return fig, axes


# def show_example(example, size, channels, ax, show_label=False):  # modifies ax inplace
#     # saved as floats but truly int, show as int
#     im = example['matrix'].astype(np.uint8).reshape(size, size, channels)
#     ax.axis('off')
#     if show_label:
#         label = example['label']
#         if isinstance(label, int):
#             name_mapping = {
#                 0: 'Feat.',
#                 1: 'Smooth'
#             }
#             label_str = name_mapping[label]
#         else:
#             label_str = '{:.2}'.format(label)
#         ax.text(60, 110, label_str, fontsize=16, color='r')
