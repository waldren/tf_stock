import tensorflow as tf


class ModelRunner():
    def __init__(self, max_epochs=20):
        self.MAX_EPOCHS = max_epochs
    
    def compile_and_fit(self, model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=self.MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
        return history

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]