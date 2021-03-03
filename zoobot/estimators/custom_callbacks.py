import tensorflow as tf

# use any custom callback to keras.backend.set_value self.epoch=epoch, and then read that on each summary call
# if this doesn't get train/test right, could similarly use the callbacks to set self.mode
# https://www.tensorflow.org/guide/keras/custom_callback
class UpdateStepCallback(tf.keras.callbacks.Callback):

    def __init__(self, batch_size):
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
        print('\n Ending step: ', float(tf.keras.backend.get_value(self.model.step)))
        # # print(f'Step {step}')
