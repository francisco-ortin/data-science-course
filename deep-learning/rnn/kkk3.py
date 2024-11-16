from keras import Input
from sklearn.model_selection import train_test_split


full_text = """The movie was not good. The animation and the graphics were terrible. I would not recommend this movie."""

texts = full_text.tolist()
texts = [' '.join(t.split()[:max_len]) for t in texts]
texts = np.array(texts, dtype=object)[:, np.newaxis]

x_train, x_val, y_train, y_val = train_test_split(texts, y, random_state=1992, test_size=0.2)


import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import tensorflow_hub as hub


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            K.squeeze(K.cast(x, tf.string), axis=1),
            as_dict=True,
            signature='default',
            )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout
)


num_classes = 1
batch_size = 512
epochs = 200
learnRate = 0.001

input_text = Input(shape=(1,), dtype="string", name='input_0')
x = ElmoEmbeddingLayer(trainable=False)(input_text)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation="sigmoid")(x)

model = Model(inputs=[input_text], outputs=x)

model.summary()

# 'binary_crossentropy'
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())