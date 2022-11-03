import tensorflow as tf
# import wandb

# TODO now deprecated
class UpdateStepCallback(tf.keras.callbacks.Callback):

    def __init__(self, batch_size):
        """
        I couldn't work out the "proper" way to track steps within a Sequential model for tensorboard, so this is a workaround.py
        Use a custom callback using keras.backend.set_value to set self.epoch=epoch, and then read that on each summary call
        Model must already have self.epoch = tf.Variable(0).
        https://www.tensorflow.org/guide/keras/custom_callback

        Args:
            batch_size (int): batch size, used to calculate steps as batch_size * epoch
        """
        super(UpdateStepCallback, self).__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        # print('\n\nStarting epoch', epoch, '\n\n')
        # # access model with self.model, tf.ketas.backend.get/set_value
        # # e.g. lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        # print('\n epoch ', epoch, type(epoch))
        step = epoch * self.batch_size
        # # self.model.step = step
        # # self.model.step.assign(step)
        tf.keras.backend.set_value(self.model.step, step)
        # print('\n Ending step: ', float(tf.keras.backend.get_value(self.model.step)))
        # # print(f'Step {step}')


# class VisualizeImages(tf.keras.callbacks.Callback):

#     # no args needed yet
#     # def __init__(self):
#     #     super(VisualizeImages, self).__init__()

#     def on_train_begin(self, logs=None):
#         tf.summary.image(name='images_before_augmentation', data, max_outputs=3, description='Images passed to Zoobot')


# callbacks can use self.model
# https://www.tensorflow.org/guide/keras/custom_callback#usage_of_selfmodel_attribute