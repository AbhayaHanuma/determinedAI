"""
This example shows how to use Determined to implement an image
classification model for the Fashion-MNIST dataset using tf.keras.
Based on: https://www.tensorflow.org/tutorials/keras/classification.
After about 5 training epochs, accuracy should be around > 85%.
This mimics theoriginal implementation. Continue training or increase
the number of epochs to increase accuracy.
"""
import fetch
import tensorflow as tf
from tensorflow import keras

from determined.keras import InputData, TFKerasTrial, TFKerasTrialContext

from keras_multi_head import MultiHead

class ApparelTrail(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context
        self.scaler_obj = None

    def build_model(self):
        LR = keras.layers.LeakyReLU()

        window_ip = keras.layers.Input(batch_shape=(None,self.context.get_hparam("window"),3,1))
        multihead = MultiHead([
            keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=self.context.get_hparam("conv1D_a_filter"),
                kernel_size=self.context.get_hparam("conv1D_a_kernel"),
                activation=LR,
                padding='causal',
                strides=1,
                kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                dilation_rate=self.context.get_hparam("conv1D_a_dilation"))
            ),
            keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=self.context.get_hparam("conv1D_b_filter"),
                kernel_size=self.context.get_hparam("conv1D_b_kernel"),
                activation='relu',
                padding='causal',
                strides=1,
                kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                dilation_rate=self.context.get_hparam("conv1D_b_dilation"))
            ),
            keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=self.context.get_hparam("conv1D_c_filter"),
                kernel_size=self.context.get_hparam("conv1D_c_kernel"),
                activation='tanh',
                padding='causal',
                strides=1,
                kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                dilation_rate=self.context.get_hparam("conv1D_c_dilation"))
            )], name='Multi_CNN')(window_ip)
        
        bnorm = keras.layers.BatchNormalization()(multihead)
        dropout1 = keras.layers.Dropout(self.context.get_hparam("dropout1"),seed=0)(bnorm)
        tdf = keras.layers.TimeDistributed(keras.layers.Flatten())(dropout1)
        lstm1 = keras.layers.LSTM(self.context.get_hparam("lstm1"),
                                activation='tanh',
                                return_sequences=True,
                                kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                                recurrent_initializer=keras.initializers.orthogonal(seed=0),
                                unroll=True)(tdf)
        dropout2 = keras.layers.Dropout(self.context.get_hparam("dropout2"),seed=0)(lstm1)
        lstm2 = keras.layers.LSTM(self.context.get_hparam("lstm2"),
                                activation='tanh',
                                return_sequences=False,
                                kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                                recurrent_initializer=keras.initializers.orthogonal(seed=0),
                                unroll=True)(dropout2)
        
        dense = keras.layers.Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(lstm2)
        model = keras.models.Model(inputs=window_ip, outputs=dense)

        model = self.context.wrap_model(model)

        optimizer = tf.keras.optimizers.Adam()
        optimizer = self.context.wrap_optimizer(optimizer)

        model.compile(
            optimizer=optimizer,
            # metrics=['accuracy'],
            loss='mean_absolute_error',)
        
        return model
    
    def build_training_data_loader(self) -> InputData:
        train_images, train_labels, scaler, data, _ = fetch.load_training_data(LAG=1,window=self.context.get_hparam('window'))
        self.scaler_obj = scaler
        self.data = data
        return train_images, train_labels

    def build_validation_data_loader(self) -> InputData:
        test_images, test_labels = fetch.load_validation_data(data=self.data,scaler=self.scaler_obj,window=self.context.get_hparam('window'))
        return test_images, test_labels

if __name__=='__main__':
    pass