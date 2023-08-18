# 데이터 교육 복습 29

# tensorflow 연습
def set_rnn(model, train_input, train_target, val_input, val_target, cb_name=None):
    model = model
    rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    if cb_name != None:
        checkpoint_cb = keras.callbacks.ModelCheckpoint(cb_name, save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        history = model.fit(train_input, train_target, epochs=100, batch_size=64, 
                validation_data=(val_input, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
    else:
        history = model.fit(train_input, train_target, epochs=100, batch_size=64, 
                validation_data=(val_input, val_target))
    
    return history


def rnn_set_model(layer1, layer2=None):
    model = keras.Sequential()
    model.add(layer1)
    if layer2 != None:
        model.add(layer2)
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


def set_padseq(data, option):
    from tensorflow.preprocessing.sequence import pad_sequences
    seq = pad_sequences(data, maxlen=option)
    return seq


def set_to_cate(data):
    cate = keras.utils.to_categorical(data)
    return cate


def rnn_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()


def set_lstm(layer, loss, metric):
    model = keras.Sequential()
    model.add(layer)
    model.add(keras.layers.Dense(1))
    model.compile(loss=loss, optimizer='adam', metrics=[metric])
    return model


def lstm_model_test(model, x_test, y_test):
    #LSTM 모델 평가
    ev = model.evaluate(x_test, y_test, verbose=0)
    print("손실 함수:", ev[0], "MAE:", ev[1])
    pred = model.predict(x_test)
    print("평균절댓값백분율오차(MAPE)",sum(abs(y_test-pred)/y_test)/len(y_test))


def lstm_plot(history):
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model mae')
    plt.ylabel('mae')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid()
    plt.show()


def lstm_plot_2(y_test, pred):
    #예측 결과 시각화
    x_range=range(len(y_test))
    plt.plot(x_range,y_test[x_range], color='red')
    plt.plot(x_range,pred[x_range], color='blue')
    plt.legend(['True prices','Predicted prices'], loc='best')
    plt.grid()
    plt.show()


def tf_train_1():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    from tensorflow.keras.datasets import imdb

    (train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

    from sklearn.model.selection import train_test_split

    train_input, val_input, train_target, val_target = train_test_split(train_input,
                                                                        train_target, test_size=0.2, random_state=42)

    lengths = np.array([len(x) for x in train_input])

    from tensorflow.preprocessing.sequence import pad_sequences

    train_seq = pad_sequences(train_input, maxlen=100)
    val_seq = pad_sequences(val_input, maxlen=100)
    print(train_input.shape, val_input.shape, train_target.shape, val_target.shape, test_input.shape, test_target.shape)
    print(train_seq.shape, val_seq.shape)

    from tensorflow import keras

    train_oh = keras.utils.to_categorical(train_seq)
    val_oh = keras.utils.to_categorical(val_seq)
    print(train_oh.shape, val_oh.shape)

    layer1 = keras.layers.SimpleRNN(8, input_shape=(100, 500))
    model = rnn_set_model(layer1)
    history = set_rnn(model, train_oh, train_target, val_oh, val_target, cb_name='best_simplernn_model.h5')
    rnn_plot(history)

    layer2 = keras.layers.Embedding(500, 16, input_length=100)
    layer3 = keras.layers.SimpleRNN(8)
    model2 = rnn_set_model(layer2, layer3)
    history = set_rnn(model2, train_seq, train_target, val_seq, val_target, cb_name='best_embedding_model.h5')
    rnn_plot(history)

    layer4 = keras.layers.LSTM(8)
    model3 = rnn_set_model(layer2, layer4)
    history = set_rnn(model3, train_seq, train_target, val_seq, val_target, cb_name='best_lstm_model.h5')
    rnn_plot(history)

    layer5 = keras.layers.LSTM(8, dropout=0.3)
    model4 = rnn_set_model(layer2, layer5)
    history = set_rnn(model4, train_seq, train_target, val_seq, val_target, cb_name='best_dropout_model.h5')
    rnn_plot(history)

    data = pd.read_csv('C:/Users/eleun/Downloads/BTC_USD_2019-02-28_2020-02-27-CoinDesk.csv')
    print(data.head())
    seq = data[['Closing Price (USD)']].to_numpy()
    print(seq.shape, seq[:5])
    print(data.info())

    plt.plot(seq, color='red')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()

    w= 7
    h= 1

    X, Y = seq2dataset(seq, w, h)
    print(X.shape, Y.shape)
    print(X[:2], Y[0])
    split = int(len(X)*0.7)
    x_train = X[:split]
    y_train = Y[:split]
    x_test = X[split:]
    y_test = Y[split:]
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    layer6 = keras.layers.LSTM(128, activation='relu', input_shape=x_train[0].shape)
    model = set_lstm(layer6, loss='mae', metric='mae')
    hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test))
    lstm_model_test(model, x_test, y_test)
    pred = model.predict(x_test)
    lstm_plot(hist)
    lstm_plot(y_test, pred)

    layer7 = keras.layers.LSTM(128, input_shape=x_train[0].shape)
    model = set_lstm(layer7, loss='mae', metric='mae')
    hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test))
    lstm_model_test(model, x_test, y_test)
    pred = model.predict(x_test)
    lstm_plot(hist)
    lstm_plot(y_test, pred)


def set_split(X, Y, split):
    x_train=X[:split]
    y_train=Y[:split]
    x_test=X[split:]
    y_test=Y[split:]
    return x_train, y_train, x_test, y_test


def tf_train_2():
    names=['site', 'year', 'mean_temp','temp','max_temp','max_date','average_min_temp','min_temp','min_date']
    data = pd.read_csv('C:/Users/eleun/Downloads/seoul_100.csv', names=names)
    data.dropna(inplace=True)
    seq=data[['mean_temp']].to_numpy()
    plt.figure(figsize=(8,6))
    plt.plot(seq)
    plt.title('Seoul annual mean temperature')
    plt.ylabel('Air temperature (C)')
    plt.xlabel('year')
    plt.grid()
    plt.show()

    #w=3, h=1
    w=3
    h=1
    X, Y = seq2dataset(seq, w, h)
    print(X.shape, Y.shape)
    split = int(len(X)*0.7)
    print(split)
    x_train, y_train, x_test, y_test = set_split(X, Y, split)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    from tensorflow import keras

    layer = keras.layers.LSTM(128, activation='relu', input_shape=x_train[0].shape)
    model = set_lstm(layer, loss='mse', metric='mae')
    hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test))
    pred = model.predict(x_test)
    lstm_plot(hist)
    lstm_plot(y_test, pred)

    #w=5, h=1
    w=5
    h=1
    X, Y = seq2dataset(seq, w, h)
    print(X.shape, Y.shape)
    split = int(len(X)*0.7)
    print(split)
    x_train, y_train, x_test, y_test = set_split(X, Y, split)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    layer = keras.layers.LSTM(128, activation='relu', input_shape=x_train[0].shape)
    model = set_lstm(layer, loss='mse', metric='mae')
    hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test))
    pred = model.predict(x_test)
    lstm_plot(hist)
    lstm_plot(y_test, pred)

    #w=7, h=1
    w=7
    h=1
    X, Y = seq2dataset(seq, w, h)
    print(X.shape, Y.shape)
    split = int(len(X)*0.7)
    print(split)
    x_train, y_train, x_test, y_test = set_split(X, Y, split)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    layer = keras.layers.LSTM(128, activation='relu', input_shape=x_train[0].shape)
    model = set_lstm(layer, loss='mse', metric='mae')
    hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test))
    pred = model.predict(x_test)
    lstm_plot(hist)
    lstm_plot(y_test, pred)
