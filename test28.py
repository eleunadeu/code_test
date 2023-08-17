# 데이터 교육 복습 28

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

def rnn_train_1():
    x_train, y_train = data_set(1500)
    x_val, y_val = data_set(300)
    x_test, y_test = data_set(100)
    print(x_train.shape, y_train.shape)
    x_tc = x_train.reshape(-1, 16, 16, 1)
    x_vc = x_val.reshape(-1, 16, 16, 1)
    print(x_tc.shape, x_vc.shape)
    x_testc = x_test.reshape(-1, 16, 16, 1)
    print(x_testc.shape)

    from keras.models import Sequential
    from keras.layers import Dense, Flatten
    from keras.layers import Conv2D, MaxPooling2D

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    hist = model.fit(x_tc, y_train, batch_size=32, epochs=1000, validation_data=(x_vc, y_val))
    hist_plot(hist)
    score = model.evaluate(x_testc, y_test, batch_size=32)
    print(score)

    yhat_test = model.predict(x_testc, batch_size=32)
    yhat_plot(x_test, y_test, yhat_test)


def rnn_train_2():
    train_data = pd.read_table('C:/Users/eleun/Downloads/test_data/ratings_train.txt')
    print(train_data.isnull().sum())
    train_data = train_data.dropna(how='any')
    print(train_data.isnull().sum())
    print(len(train_data))
    train_data['documents']=train_data['documents'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

    from konlpy.tag import Okt

    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    print(train_data.info())
    okt = Okt()
    tokenized_data = []
    for sentence in train_data['documents']:
        tokenized_sentence = okt.morphs(sentence, stem=True)
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
        tokenized_data.append(stopwords_removed_sentence)

    print(tokenized_data[1:3])
    print(sum(max(len, tokenized_data))/len(tokenized_data)) # 리뷰 평균 길이

    from gensim.models import Word2Vec

    model = Word2Vec(sentences = tokenized_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)
    print(model.wv.vectors.shape)
    print(model.wv.most_similar('영웅'))

    names=['site', 'year', 'mean_temp','temp','max_temp','max_date','average_min_temp','min_temp','min_date']
    data = pd.read_csv('C:/Users/eleun/Downloads/seoul_100.csv', names=names)
    print(data.head())
    data.dropna(inplace=True)
    seq = data[['temp']].to_numpy()
    print(seq.shape)
    w=10
    h=1
    X, Y = seq2dataset(seq, w, h)
    print(X.shape, Y.shape)
    print(X[:2])
    print(Y[:2])

    from tensorflow.keras.datasets import imdb

    (train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
    print(train_input.shape, test_input.shape)
    print(len(train_input[0]))
    print(len(train_input[1]))
    print(train_input[0])
    print(train_target[:10])

    from sklearn.model_selection import train_test_split

    train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
    lengths = np.array([len(x) for x in train_input])
    print(lengths)
    print(np.mean(lengths), np.median(lengths))

    plt.hist(lengths)
    plt.xlabel('lenghts')
    plt.ylabel('frequency')
    plt.show()
