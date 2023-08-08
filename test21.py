# 데이터 교육 복습 21

#80. 딥러닝, 자연어 처리 연습
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def text_train_5():
    review_df = pd.read_csv('C:/Users/eleun/Downloads/labeledTrainData.tsv', header=0, sep='\t', quoting=3)
    print(review_df.head())
    review_df['review'] = review_df['review'].str.replace('<br />', ' ')
    
    import re

    review_df['review']= review_df['review'].apply(lambda x: re.sub('[^a-zA-Z]',' ', x))
    
    import nltk
    nltk.download('all')

    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import sentiwordnet as swn
    from nltk import sent_tokenize, word_tokenize, pos_tag

    def penn_to_wn(tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return
    
    def swn_polarity(text):
        sentiment = 0.0
        tokens_count = 0

        lemmatizer = WordNetLemmatizer()
        raw_sentences = sent_tokenize(text)
        for raw_sentence in raw_sentences:
            tagged_sentence = pos_tag(word_tokenize(raw_sentence))
            for word, tag in tagged_sentence:
                wn_tag = penn_to_wn(tag)
                if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB): #함수화 필요
                    continue
                lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                synsets = wn.synsets(lemma, pos=wn_tag)
                if not synsets: #함수화 필요
                    continue
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                sentiment += (swn_synset.pos_score()-swn_synset.neg_score())
                tokens_count +=1
        if sentiment >= 0:
            return 1
        return 0
    
    review_df['preds'] = review_df['review'].apply(lambda x: swn_polarity(x))
    print(review_df.head())

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
    from sklearn.metrics import recall_score, f1_score
    import numpy as np

    y_target = review_df['sentiment'].values
    preds = review_df['preds'].values

    print(confusion_matrix(y_target, preds))
    print(accuracy_score(y_target, preds))
    print(precision_score(y_target, preds))
    print(recall_score(y_target, preds))
    print(f1_score(y_target, preds))

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    senti_analyzer = SentimentIntensityAnalyzer()
    senti_scores = senti_analyzer.polarity_scores(review_df['review'][10])
    print(senti_scores)

    def vader_polarity(review, threshold=0.1):
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(review)
        agg_score = scores['compound']
        final_sentiment = 1 if agg_score >= threshold else 0
        return final_sentiment
    
    review_df['vader'] = review_df['review'].apply(lambda x: vader_polarity(x, 0.1))

    y_target = review_df['sentiment'].values
    preds = review_df['vader'].values
    print(confusion_matrix(y_target, preds))
    print(accuracy_score(y_target, preds))
    print(precision_score(y_target, preds))
    print(recall_score(y_target, preds))
    print(f1_score(y_target, preds))

    import glob, os
    import warnings

    warnings.filterwarnings('ignore')
    pd.set_option('display.max_colwidth', 700)

    path = 'C:/Users/eleun/Downloads/topics'
    all_files = glob.glob(os.path.join(path, "*.data"))
    filename_list = []
    opinion_text = []

    for file_ in all_files:
        filename_list.append(file_.split('/')[-1].split('.')[0])
        df = pd.DataFrame(file_, index_col=None, header=0, encoding='latin1')

    print(len(filename_list))
    print(len(opinion_text))
    document_df = pd.DataFrame({'filename':filename_list, 'opinion_text':opinion_text})
    print(document_df.head())

    import string
    print(string.punctuation)

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    lemmar = WordNetLemmatizer()

    def LemTokens(tokens):
        return [lemmar.lemmatize(token) for token in tokens]

    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    # nltk.word_tokenize(text.lower().translate(remove_punct_dict)) : 문장부호 삭제 한 후, 단어로 토큰화한 리스트 결과

    print(remove_punct_dict)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1, 2), min_df=0.05, max_df=0.85)
    feature_vect = tfidf_vect.fit_transform(document_df['opinion_text'])
    print(feature_vect.toarray())
    #함수화
    km_cluster = KMeans(n_clusters=5, max_iter=10000, random_state=0)
    km_cluster.fit(feature_vect)
    cluster_label = km_cluster.labels_
    cluster_centers = km_cluster.cluster_centers_
    print(cluster_label)
    document_df['cluster_label'] = cluster_label
    print(document_df.head())
    print(document_df[document_df['cluster_label']==4].sort_values(by='filename'))
    #함수화
    km_cluster = KMeans(n_clusters=3, max_iter=10000, random_state=0)
    km_cluster.fit(feature_vect)
    cluster_label = km_cluster.labels_
    cluster_centers = km_cluster.cluster_centers_
    print(cluster_label)
    document_df['cluster_label'] = cluster_label
    print(document_df[document_df['cluster_label']==2].sort_values(by='filename'))

    doc1 = np.array([0, 1, 1, 1]) # 저는 사과 좋아요
    doc2 = np.array([1, 0, 1, 1]) # 저는 바나나 좋아요
    doc3 = np.array([2, 0, 2, 2]) # 저는 바나나 좋아요 저는 바나나 좋아요

    print(feature_vect.toarray())
    print(np.dot(doc1, doc2))

    from numpy.linalg import norm

    def cos_sim(a, b):
        return np.dot(a, b)/(norm(a)*norm(b))
    
    print(cos_sim(doc1, doc2))
    print(cos_sim(doc1, doc3))
    print(cos_sim(doc2, doc3))

    doc_list = ['if you take the blue pill, the story ends' ,
            'if you take the red pill, you stay in Wonderland',
            'if you take the red pill, I show you how deep the rabbit hole goes']
    
    td_ = TfidfVectorizer()
    fv_ = td_.fit_transform(doc_list)

    print(fv_.shape)
    print(fv_.toarray())
    print(fv_)
    print(fv_.todense())

    fv_array = fv_.toarray()
    doc1 = fv_array[0]
    doc2 = fv_array[1]
    doc3 = fv_array[2]
    cos_sim(doc1, doc2)
    cos_sim(doc1, doc3)
    cos_sim(doc2, doc3)

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_similarity(fv_[0], fv_)
    print(document_df[document_df['cluster_label']==1].sort_values(by='filename'))
    hotel_indexes = document_df[document_df['cluster_label']==1].index
    print(hotel_indexes, len(hotel_indexes))
    print(feature_vect[0])
    print(feature_vect[hotel_indexes])
    sim_pair = cosine_similarity(feature_vect[hotel_indexes[0]], feature_vect[hotel_indexes])
    print(sim_pair)
    print(sim_pair.shape)
    low_sort_index = sim_pair.argsort()
    print(low_sort_index[0])
    print(sim_pair[:, low_sort_index[0]])
    sort_index = sim_pair.argsort()[:, ::-1]
    print(sort_index)
    sorted_index = sort_index[:, 1:].reshape(-1)
    print(sorted_index)
    hs_index = hotel_indexes[sorted_index]
    print(hotel_indexes[sorted_index])
    hs_sim_values = sim_pair[0, sorted_index]
    print(hs_sim_values)
    print(document_df.iloc[hs_index].filename)
    df = pd.DataFrame()
    df['fn'] = document_df.iloc[hs_index].filename
    df['sn'] = hs_sim_values
    print(df)
    sns.barplot(x='sn', y='fn', data=df)
    plt.show()
    print(document_df.iloc[hotel_indexes[0]])


def text_train_6():
    from konlpy.tag import Okt
    from sklearn.feature_extraction.text import TfidfVectorizer
    import re
    
    train_df = pd.read_csv('C:/Users/eleun/Downloads/ratings_train.txt', sep='\t')
    test_df = pd.read_csv('C:/Users/eleun/Downloads/ratings_test.txt', sep='\t')
    print(train_df.head())
    print(test_df.head())
    print(train_df.info())
    print(test_df.info())
    print(train_df['label'].value_counts())
    train = train_df.dropna()
    train_df = train.fillna(' ')
    test = test_df.dropna()
    test_df = test.fillna(' ')
    train_df['document'] = train_df['document'].apply(lambda x: re.sub(r"\d+"," ", x))
    test_df['document'] = test_df['document'].apply(lambda x: re.sub(r"\d+"," ", x))
    train_df.drop('id', axis=1, inplace=True)
    test_df.drop('id', axis=1, inplace=True)

    okt = Okt()
    print(okt.morphs('지루하지는 않은데 완전 막장임... 돈주고 보기에는'))
    print(okt.pos('지루하지는 않은데 완전 막장임... 돈주고 보기에는'))
    print(okt.nouns('지루하지는 않은데 완전 막장임... 돈주고 보기에는'))

    okt = Okt()
    def tw_tokenizer(text):
        tokens = okt.morphs(text)
        return tokens
    
    print(tw_tokenizer('지루하지는 않은데 완전 막장임... 돈주고 보기에는'))
    tfidf_vect = TfidfVectorizer(tokenizer=tw_tokenizer, ngram_range=(1, 2), min_df=3, max_df=0.9)
    tfidf_vect.fit(train_df['document'])
    tfidf_matrix_train = tfidf_vect.transform(train_df['document'])

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    lg_clf = LogisticRegression(solver='liblinear', random_state=0)
    params = {'C':[1, 3.5, 4.5, 5.5, 10]}
    grid_cv = GridSearchCV(lg_clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
    grid_cv.fit(tfidf_matrix_train, train_df['label'])
    print(grid_cv.best_params_, round(grid_cv.best_score_, 4))

    from sklearn.metrics import accuracy_score

    tfidf_matrix_test = tfidf_vect.transform(test_df['document'])
    best_estimator = grid_cv.best_estimator_
    preds = best_estimator.predict(tfidf_matrix_test)

    print('Logistic Regression 정확도 :' , accuracy_score(test_df['label'], preds))

class SoftMax:
    def __init__(self, x, a):
        self.x = x
        self.a = a

    def identity_function(self):
        return self.x

    def relu(self):
        return np.maximum(0, self.x)

    def step_function(self):
        return np.array(self.x > 0, dtype=np.int)

    def sigmoid(self):
        return 1 / (1+np.exp(-self.x))

    def softmax(self):
        c = np.max(self.a)
        exp_a = np.exp(self.a-c) #오버 플로 대책
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

def dml_train_1():
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    x = np.arange(-15.0, 15.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()

    X = np.array([1.0, 1.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    print(X.shape, W1.shape)
    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    print(Z1)

    def identity_function(x):
        return x

    def relu(x):
        return np.maximum(0, x)

    def step_function(x):
        return np.array(x > 0, dtype=np.int)

    def sigmoid(x):
        return 1 / (1+np.exp(-x))

    def softmax(a):
        c = np.max(a)
        exp_a = np.exp(a-c) #오버플로 대책
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
    
    x = np.arange(-15, 15, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.show()

    X=np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    print(A1)
    print(Z1)
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(A2)
    print(Z2)
    print(X.shape, W1.shape, Z1.shape, W2.shape, Z2.shape)
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)
    print(A3)
    print(Y)

    def init_network():
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network['B1'] = np.array([0.1, 0.2, 0.3])
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network['B2'] = np.array([0.1, 0.2])
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network['B3'] = np.array([0.1, 0.2])
        return network
    
    def forward(network, X):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        B1, B2, B3 = network['B1'], network['B2'], network['B3']

        A1 = np.dot(X, W1) + B1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W2) + B2
        Z2 = sigmoid(A2)
        A3 = np.dot(Z2, W3) + B3
        Y = identity_function(A3)
        return Y
    
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

    a = np.array([0.3, 2.9, 4.0])

    exp_a = np.exp(a) #분자 계산
    print(exp_a)

    sum_exp_a = np.sum(exp_a) #분모 계산
    print(sum_exp_a)

    y = exp_a / sum_exp_a
    print(y)

    a = np.array([1010, 1000, 990])
    print(np.exp(a)/np.sum(np.exp(a)))

    a - np.max(a)
    print(np.exp(a-np.max(a))/np.sum(np.exp(a-np.max(a))))
    
    x = np.array([2, 3, 4])
    print(np.exp(x)/np.sum(np.exp(x)))

    print(np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x))))

    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))

    x = np.arange(-1, 5, 0.1)
    y = softmax(x)
    plt.plot(x, y)
    plt.show()

def identity_function(x):
    return x

def relu(x):
     return np.maximum(0, x)

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
