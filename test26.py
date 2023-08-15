# 데이터 교육 복습 26

#85. 딥러닝 연습 6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras

def build_model(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer != None:
        model.add(a_layer)
        model.add(keras.layers.Dense(10, activation='softmax'))
    return model


def keras_train_2():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    (train_input, train_target) , (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

    from sklearn.model_selection import train_test_split

    train_scaled = train_input/255
    train_scaled = train_scaled.reshape(-1, 784)
    train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

    dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
    dense2 = keras.layers.Dense(10, activation='softmax')
    model = keras.Sequential([dense1, dense2])
    print(model.summary())

    model = keras.Sequential([
        keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
        keras.layers.Dense(10, activation='softmax', name='output')], name='패션 MNIST model')
    print(model.summary())

    model = keras.Sequential()
    model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784, )))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(train_scaled, train_target, epochs=5)
    print(model.evaluate(val_scaled, val_target))

    #Relu, Flatten
    train_scaled = train_input/255
    #train_scaled = train_scaled.reshape(-1, 784)

    train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(train_scaled, train_target, epochs=5)
    print(model.evaluate(val_scaled, val_target))

    #옵티마이저 사용
    sgd = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    '''
    케라스 기본 옵티마이저 = RMSprop
    SGD = sgd + momentum + nesterov
    learning_rate 기본값 = 0.01
    momentum 기본값 = 0
    nesterov 기본값 = False
    Adagrad, RMSprop, Adam의 기본 학습률 = 0.001
    keras.optimizers.Adagrad()
    keras.optimizers.RMSprop()
    keras.optimizers.Adam()
    '''
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(train_scaled, train_target, epochs=5)
    print(model.evaluate(val_scaled, val_target))

    #콜백 사용
    (train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
    train_scaled = train_input/255.0
    train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model = build_model()
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
    print(history.history)
    print(history.history.keys())
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.show()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=125, verbose=0)

    import history_plot as hp

    hp.history_plot(history)
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=40, verbose=0, validation_data=(val_scaled, val_target))
    print(history.history)
    print(history.history.keys())
    hp.history_plot(history)
    model = build_model()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=40, verbose=0, validation_data=(val_scaled, val_target))
    hp.history_plot(history)
    model = build_model()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=40, verbose=0, validation_data=(val_scaled, val_target))
    hp.history_plot(history)
    model = build_model()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=100, verbose=0, validation_data=(val_scaled, val_target))
    hp.history_plot(history)
    model = build_model(keras.layers.Dropout(0.3))
    print(model.summary())


def keras_train_3():
    #https://keras.io/ko/callbacks/ 콜백 사이트
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    (train_input, train_target) , (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

    import history_plot as hp
    from sklearn.model_selection import train_test_split

    train_scaled = train_input/255
    train_scaled = train_scaled.reshape(-1, 784)
    train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
    hp.history_plot(history)

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
    hp.history_plot(history)
    print(history.history.keys())
    print(history.history['val_loss'])
    print(history.history['val_accuracy'])

    #최적화된 모델 저장 후 사용
    model = build_model(keras.layers.Dropout(0.3))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    history = model.fit(train_scaled, train_target, epochs=8, validation_data=(val_scaled, val_target))
    model.save_weights('model_weights.h5')
    model.save('model_whole.h5')
    model = build_model(keras.layers.Dropout(0.3))
    model.load_weights('model_weights.h5')

    x = model.predict(val_scaled)
    y = np.argmax(x, axis=1)
    accuracy = np.mean(y ==val_target)
    print(accuracy)

    model = keras.models.load_model('model_whole.h5')
    print(model.evaluate(val_scaled, val_target))

    #콜백
    model = build_model(keras.layers.Dropout(0.3))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    checkpoint_cb = keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

    model = keras.models.load_model('best_model.h5')
    print(model.evaluate(val_scaled, val_target))

    model = build_model(keras.layers.Dropout(0.3))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    checkpoint_cb = keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    es_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
    print(es_cb.stopped_epoch)
    hp.history_plot(history)
    print(model.evaluate(val_scaled, val_target))

    #Keras Dense 층 예제
    from keras.utils import np_utils
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Activation

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(-1, 784).astype('float32')/255.0
    x_test = x_test.reshape(-1, 784).astype('float32')/255.0
    y_train = np.utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    x_val = x_train[50000:]
    y_val = y_train[50000:]
    x_train = x_train[:50000]
    y_train = y_train[:50000]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
    hp.history_plot(history)
    print(model.evaluate(x_test, y_test))

    xhat = x_test[:2]
    yhat = model.predict(xhat)
    print(np.argmax(yhat, axis=1))
    print(y_test[:2])
