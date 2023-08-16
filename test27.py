# 데이터 교육 복습 27

#86. 딥러닝 연습 7
# 라이브러리 생략

def keras_train_4():
    from tensorflow.keras.utils import to_categorical
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    import history_plot as hp
    #import keras_plot as kp # history_plot.py 파일 수정 후 이름 변경
    

    def model_fn(a_layer=None):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(784,)))
        if a_layer != None:
            model.add(a_layer)
            model.add(Dense(10, activation='softmax'))
        return model


    def model_test(optimizers, epoch=None, callback=None):
        loss_acc_opti = {}
        for opt in optimizers:
            model = model_fn()
            model.compile(otimimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
            print(opt)
            if callback ==None:
                history = model.fit(x_train, y_train, epochs=epoch, batch_size=32, validation_data=(x_val, y_val), verbose=0)
            else:
                history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val), verbose=0, callbacks=[callback])
            loss_acc = model.evaluate(x_test, y_test)
            loss_acc_opti[opt] = (loss_acc, history)
        return loss_acc_opti


    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1

        return results


    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    x_train = x_train.reshape(-1, 784).astype('float32')/255.0
    x_test = x_test.reshape(-1, 784).astype('float32')/255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_val = x_train[50000:]
    y_val = y_train[50000:]
    x_train = x_train[:50000]
    y_train = y_train[:50000]

    opti = ['sgd', 'adagrad', 'rmsprop', 'adam']
    opt_test_1 = model_test(opti)
    print(opt_test_1)
    for i in opti:
        hp.history_plot(i, opt_test_1['opti'][1])
        #kp.keras_plot(i, opt_test_1['opti'][1], height=6, width=6)

    #데이터 일부 추출
    idx_train = np.random.choice(50000, 700)
    idx_val = np.random.choice(10000, 300)
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    x_val = x_val[idx_val]
    y_val = y_val[idx_val]

    opt_test_2 = model_test(opti)
    print(opt_test_2)
    for i in opti:
        hp.history_plot(i, opt_test_2['opti'][1])
        #kp.keras_plot(i, opt_test_2['opti'][1], height=6, width=6)

    plt.show()

    #에포크 증가
    opt_test_3 = model_test(opti, epoch=1000)
    print(opt_test_3)
    hp.history_plot(opt_test_3[opti][1])
    #kp.keras_plot(opt_test_3[opti][1])

    #콜백
    import keras
    class CustomHistory(keras.callbacks.Callback):
        def __init__(self):
            self.losses = []
            self.val_losses = []
            self.accs = []
            self.val_accs = []

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.accs.append(logs.get('accuracy'))
            self.val_accs.append(logs.get('val_accuracy'))

    opti = 'adam'
    model = model_fn()
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics='accuracy')
    custom_hist = CustomHistory()
    for epoch_idx in range(100):
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val),
                            verbose=0, callbacks=[custom_hist])
    print(model.evaluate(x_test, y_test))
    #kp.keras_plot(history)

    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(custom_hist.losses, 'y', label='train loss')
    loss_ax.plot(custom_hist.val_losses, 'r', label='val loss')

    acc_ax.plot(custom_hist.accs, 'b', label='train acc')
    acc_ax.plot(custom_hist.val_accs, 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    loss_ax.legend(loc='lower left')

    from keras.callbacks import EarlyStopping

    early_stop = [EarlyStopping(), EarlyStopping(patience=2), EarlyStopping(patience=10)]
    opt_test_5 = model_test(opti, epoch=1000, callback=[early_stop[0]])
    opt_test_6 = model_test(opti, epoch=1000, callback=[early_stop[1]])
    opt_test_7 = model_test(opti, epoch=1000, callback=[early_stop[2]])
    opt_list = [opt_test_5, opt_test_6, opt_test_7]
    for i in opt_list:
        hp.history_plot(i['opti'][1])
        #kp.keras_plot(i['opti'][1])

    #이진 분류
    from keras.datasets import imdb

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    print(type(train_data))
    print(train_data[:2])
    for i in train_data[:3]:
        print(len(i))
    print(train_labels[:2])
    print(max([len(seq) for seq in train_data]))
    word_index = imdb.get_word_index()
    print(train_data[1][:3])
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    print(' '.join([reverse_word_index.get(i-3, '!') for i in train_data[0]]))
    print(reverse_word_index[3])
    for i in train_data[:3]:
        print(len(i))

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    print(len(train_data))
    print(x_train.shape, x_test.shape)
    x_val = x_train[:10000]
    p_x_train = x_train[10000:]
    y_val = y_train[:10000]
    p_y_train = y_train[10000:]

    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(10000,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    history = model.fit(p_x_train, p_y_train, epochs=20, batch_size=512, validation_data=(x_val,y_val))
    hp.history_plot(history)
    #kp.keras_plot(history)

    model.fit(p_x_train, p_y_train, epochs=3, batch_size=512, validation_data=(x_val,y_val))
    print(model.evaluate(x_test, y_test))

    import random

    def generate_array(seeds=None, width=16, height=16):
        random.seed(seeds)
        resolution = range(width*height) #resolution = 256
        number_of_points=random.choice(resolution) #number_of_points = 10
        idx = random.sample(resolution, number_of_points) #idx : 먼지 10개의 위치(=인덱스 번호) [12, 56, 1, 2, 0...]
        row_list = [1 if (i in idx) else 0 for i in resolution] #row_list = [1, 1, 1, 0, 0, .., , 1, ..]
        arrays = np.array(row_list).reshape(width, height)
        nop=np.array([number_of_points]) #sklearn 라이브러리에 사용될때는 2차원 배열이 기본!
        return (arrays, nop)

    plt_row=5
    plt_col=5
    fig, ax = plt.subplots(plt_row, plt_col, figsize=(10, 10))
    for i in range(plt_row*plt_col):
        sub_plt=ax[i//plt_row, i%plt_col]
        arr_result=generate_array()
        sub_plt.imshow(arr_result[0])
        sub_plt.set_title('label = '+str(arr_result[1]))
        sub_plt.axis('off')
    plt.show()

    def data_set(samples):
        ds_x = []
        ds_y = []
        for i in range(samples):
            x, y = generate_array(i)
            ds_x.append(x)
            ds_y.append(y)
        ds_x = np.array(ds_x)
        ds_y = np.array(ds_y)
        return (ds_x, ds_y)

    x_train, y_train = data_set(1500)
    print(x_train.shape, y_train.shape)
    print(x_train[:3])

    x_train = x_train.reshape(x_train.shape[0], 256)
    x_val = x_val.reshape(x_val.shape[0], 256)
    x_test = x_test.reshape(x_test.shape[0], 256)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    model = Sequential()
    model.add(Dense(256, input_shape=(256, ), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics='mae')
    hist = model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val))
    print(model.evaluate(x_test, y_test, batch_size=32))
    yhat_test = model.predict(x_test, batch_size=32)
    plt_row = 5
    plt_col = 5

    plt.rcParams["figure.figsize"] = (10,10)

    f, axarr = plt.subplots(plt_row, plt_col)

    for i in range(plt_row*plt_col):
        sub_plt = axarr[i//plt_row, i%plt_col]
        sub_plt.axis('off')
        sub_plt.imshow(x_test[i].reshape(16, 16))
        sub_plt.set_title(f'R {y_test[i][0]} P {round(yhat_test[i][0])}')
    plt.show()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    from tensorflow import keras
    from sklearn.model_selection import train_test_split

    (train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()

    train_scaled= train_input.reshape(-1, 28, 28, 1)/255.0

    train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

    #cnn
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                padding='same', input_shape=(28, 28, 1))) # 32개 필터, 커널 사이즈 (3, 3, 1)
                                # 출력값 (28, 28, 32)
    model.add(keras.layers.MaxPooling2D(2)) #(2, 2) 사용  -> (14, 14, 32)
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                padding='same')) # (14, 14, 64)

    model.add(keras.layers.MaxPooling2D(2)) #(7, 7, 64)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    print(model.summary())

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    cp_cb = keras.callbacks.ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
    es_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

    history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target),
                        callbacks=[cp_cb, es_cb])

    print(model.evaluate(val_scaled, val_target))
    hp.history_plot(history)
    #kp.keras_plot(history)
    preds = model.predict(val_scaled[0:1])
    print(preds)

import random

def generate_image(a=None, width=16, height=16): #points는 찍혀야 할 점의 개수
    '''
        array : 특정 픽셀에 1 이 찍혀 있는 배열, 나머지는 0
        nop : array 에 있는 1의 개수, 정답 역활을 한다. 
    '''
    random.seed(a)
    resolution = range(width*height)
    number_of_points = random.choice(resolution)
    idx = random.sample(resolution, number_of_points)
    row_list = [1 if (i in idx) else 0 for i in resolution]
    array = np.array(row_list).reshape(width, height)
    nop = np.array([number_of_points])
    return (array, nop)


#data 만들기
def data_set(samples):
    ds_x = []
    ds_y = []
    for i in range(samples):
        x, y = generate_image(i)
        ds_x.append(x)
        ds_y.append(y)
    ds_x = np.array(ds_x)
    ds_y = np.array(ds_y)
    return (ds_x, ds_y)


def seq2dataset(seq, window, horizon):
    X = []
    Y = []
    for i in range(len(seq)-(window+horizon)+1):
        idx = i+window
        x = seq[i:idx]
        y = (seq[idx+horizon-1])
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


def hist_plot(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylim(0.0, 100.0)
    plt.xlim(0.0, 200.0)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def yhat_plot(x_test, y_test, yhat):
    width = 16
    height = 16
    plt_row = 5
    plt_col = 5
    plt.rcParams['figure.figsize']=(10, 10)
    f, axarr = plt.subplots(plt_row, plt_col)

    for i in range(plt_row*plt_col):
        sub_plt = axarr[i//plt_row, i%plt_col]
        sub_plt.axis('off')
        sub_plt.imshow(x_test[i].reshape(width, height))
        sub_plt.set_title('R %d P %.1f' % (y_test[i][0], yhat[i][0]))
    plt.show()
