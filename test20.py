#데이터 교육 복습 20

#79. ml 텍스트, 자연어 처리 연습
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def text_train_1():
    from nltk import sent_tokenize, word_tokenize
    import nltk

    nltk.download('punkt')
    text_sample = '''When I choose to see the good side of things, I’m not being naive. It is strategic and necessary. It’s how I’ve learned to survive through everything.'''
    print(text_sample)
    sentences = sent_tokenize(text=text_sample)
    print(sentences)
    print(len(sentences))
    print(type(sentences))

    words = word_tokenize(sentences[0])
    print(words)

    def tokenize_text(text):
        words = []
        sent = sent_tokenize(text)
        for i in sent:
            words.append(word_tokenize(i))
        return words
    
    tokenize_text(text_sample)
    
    nltk.download('stopwords')
    print(nltk.corpus.stopwords.words('english'))
    print(len(nltk.corpus.stopwords.words('english')))
    stopwords_list = nltk.corpus.stopwords.words('english')

    import string

    print(string.punctuation)
    trans_dic = {101:None, 102:None, 103:105}
    str_ = 'abcdefg_hi'
    print(str_.translate(trans_dic))
    for i in str_:
        print(i, ord(i))
    print(ord('가'))
    

    def rm_punc(snts):
        rm_punc_dict = {}
        for punc in string.punctuation:
            rm_punc_dict[ord(punc)]=None
        return snts.translate(rm_punc_dict)
    
    print(rm_punc(text_sample))

    def tokenize_text(text, stopwords):
        word_list = []
        sentences_list = sent_tokenize(text)
        for snts in sentences_list:
            snts = rm_punc(snts)
            words = word_tokenize(snts)
            token_list = []
            for word in words:
                word = word.lower()
                if word not in stopwords_list and len(word) > 1:
                    token_list.append(word)
                word_list.append(token_list)
        return word_list
    
    result = tokenize_text(text_sample, stopwords_list)
    print(result)

    for i in result[0]:
        print(i, len(i))

    from nltk.stem import LancasterStemmer

    stemmer = LancasterStemmer()
    print(stemmer.stem('working'), stemmer.stem('works'), stemmer.stem('worked'))
    print(stemmer.stem('amusing'), stemmer.stem('amuses'), stemmer.stem('amused'))
    print(stemmer.stem('happier'), stemmer.stem('happiest'))
    print(stemmer.stem('fancier'), stemmer.stem('fanciest'))

    nltk.download('all')

    from nltk.stem import WordNetLemmatizer

    lemma = WordNetLemmatizer()
    print(lemma.lemmatize('amusing','v'),lemma.lemmatize('amuses','v'),lemma.lemmatize('amused','v'))
    print(lemma.lemmatize('happier','a'),lemma.lemmatize('happiest','a'))
    print(lemma.lemmatize('fancier','a'),lemma.lemmatize('fanciest','a'))

    ['사과 딸기', '딸기 바나나', '수박' , '수박 수박']

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    tfidf = TfidfVectorizer()
    contv = CountVectorizer()

    from scipy import sparse

    dense2 = np.array([[0,0,1,0,0,5],
                [1,4,0,3,2,5],
                [0,6,0,3,0,0],
                [2,0,0,0,0,0],
                [0,0,0,7,0,8],
                [1,0,0,0,0,0]])
    data2 = np.array([1, 5, 1, 4, 3, 2, 5, 6, 3, 2, 7, 8, 1])
    print(dense2)
    print(data2)
    row_pos = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5])
    col_pos = np.array([2, 5, 0, 1, 3, 4, 5, 1, 3, 0, 3, 5, 0])

    # COO 형식으로 변환 
    sparse_coo = sparse.coo_matrix((data2, (row_pos,col_pos)))

    # 행 위치 배열의 고유한 값들의 시작 위치 인덱스를 배열로 생성
    row_pos_ind = np.array([0, 2, 7, 9, 10, 12, 13])

    # CSR 형식으로 변환 
    sparse_csr = sparse.csr_matrix((data2, col_pos, row_pos_ind))

    print('COO 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인')
    print(sparse_coo.toarray())
    print('CSR 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인')
    print(sparse_csr.toarray())


def text_train_2():

    from sklearn.datasets import fetch_20newsgroups

    news_data = fetch_20newsgroups(subset='all', random_state=156)
    print(news_data.keys())
    print(news_data.target)
    print(news_data.target)
    print(pd.Series(news_data.target).value_counts().sort_index())
    print(news_data.data[0])
    train_news = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=156)
    X_train = train_news.data
    y_train = train_news.target
    test_news = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), random_state=156)
    X_test = test_news.data
    y_test = test_news.target
    cnt_vect = CountVectorizer()
    cnt_vect.fit(X_train)
    X_train_cnt_vect = cnt_vect.transform(X_train)
    X_test_cnt_vect = cnt_vect.transform(X_test)

    X_train_cnt_vect.shape, X_test_cnt_vect.shape
    print(type(X_test))

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import warnings

    warnings.filterwarnings('ignore')
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train_cnt_vect, y_train)
    pred = lr_clf.predict(X_test_cnt_vect)
    print(accuracy_score(y_test, pred))
    print(X_test[:2])
    print(y_test)

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(X_train)
    X_train_tfidf = tfidf_vect.transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train_tfidf, y_train)
    pred = lr_clf.predict(X_test_tfidf)
    print(accuracy_score(y_test, pred))

    tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)
    tfidf_vect.fit(X_train)
    X_train_tfidf = tfidf_vect.transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train_tfidf, y_train)
    pred = lr_clf.predict(X_test_tfidf)
    print(accuracy_score(y_test, pred))

    from sklearn.model_selection import GridSearchCV
    params = {'C':[0.01, 0.1, 1, 5, 10]}
    grid = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=3)
    grid.fit(X_train_tfidf, y_train)
    print(grid.best_params_, grid.best_score_)

    pred = grid.predict(X_test_tfidf)
    print(accuracy_score(y_test, pred))

    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)),
        ('lr_clf', LogisticRegression(solver='liblinear', C=10))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(accuracy_score(y_test, pred))

    pipeline = Pipeline([
        ('tfidf_vect', TfidfVectorizer(stop_words='english')),
        ('lr_clf', LogisticRegression(solver='liblinear'))
    ])
    params = {'tfidf_vect__ngram_range':[(1,1,), (1,2), (1,3)],
              'tfidf_vect__max_df':[100,300,700],
              'lr_clf__C':[1, 5, 10]}
    grid = GridSearchCV(pipeline, param_grid=params, scoring='accuracy', cv=3)
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_score_)

def text_train_3():
    review_df = pd.read_csv('C:/Users/eleun/Downloads/labeledTrainData.tsv', header=0, sep='\t', quoting=3)
    print(review_df.head(3))
    print(review_df['review'][0])
    review_df['review'] = review_df['review'].str.replace('<br />',' ')
    print(review_df['review'][0])
    print(review_df['review'][11])

    import re
    
    review_df['review'] = review_df['review'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    print(review_df['review'][11])

    from sklearn.model_selection import train_test_split

    class_df = review_df['sentiment']
    feature_df = review_df.drop(['id', 'sentiment'], axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(feature_df, class_df, test_size=0.3, random_state=156)
    print(X_train.shape, X_test.shape)

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('cnt_vect', CountVectorizer(stop_words='english', ngram_range=(1,2))),
        ('lr_clf', LogisticRegression(solver='liblinear', C=10))
    ])
    pipeline.fit(X_train['review'], y_train)
    pred = pipeline.predict(X_test['review'])
    print(accuracy_score(y_test, pred))
    pred_probas = pipeline.predict_proba(X_test['review'])[:, 1]
    print(roc_auc_score(y_test, pred_probas))

    pipeline = Pipeline([
        ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('lr_clf', LogisticRegression(solver='liblinear', C=10))
    ])
    pipeline.fit(X_train['review'], y_train)
    pred = pipeline.predict(X_test['review'])
    pred_probas = pipeline.predict_proba(X_test['review'])[:, 1]

    print('예측 정확도는 {0:.4f}, ROC-AUC는 {1:.4f}'.format(accuracy_score(y_test, pred), roc_auc_score(y_test, pred_probas)))

    import nltk
    from nltk.corpus import wordnet as wn

    nltk.downlaod('all')
    term = 'present'
    synsets = wn.synsets(term)
    print(synsets)
    print(type(synsets))
    print(len(synsets))
    synsets2 = wn.synsets(term, pos=wn.NOUN)
    print(synsets2)

    for synset in synsets:
        print(synset.name())
        print('\n')
        print(synset.definition())
        print('\n')
        print(synset.lexname())
        print('\n')
        print(synset.lemma_names())

    tree = wn.synset('tree.n.01')
    lion = wn.synset('lion.n.01')
    tiger = wn.synset('tiger.n.02')
    cat = wn.synset('cat.n.01')
    dog = wn.synset('dog.n.01')

    entities = [tree, lion, tiger, cat, dog]
    similarities = []
    entity_names = [ entity.name().split('.')[0] for entity in entities ]
    print(entity_names)

    for entity in entities:
        similarity = [round(entity.path_similarity(compared_entity), 2) for compared_entity in entities]
        similarities.append(similarity)
    
    similarity_df = pd.DataFrame(similarities, columns=entity_names, index=entity_names)
    print(similarity_df)


def text_train_4():
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet as wn

    senti_synsets = list(swn.senti_synsets('slow'))
    print('senti_synsets() 반환 type :', type(senti_synsets))
    print('senti_synsets() 반환 값 갯수 :', len(senti_synsets))
    print('senti_synsets() 반환 값 :', senti_synsets)

    print(senti_synsets[0].synset.definition())
    father = swn.senti_synset('father.n.01')
    print(father.pos_score())
    print(father.neg_score())
    print(father.obj_score())

    fb = swn.senti_synset('fabulous.a.01')
    print(fb.pos_score())
    print(fb.neg_score())
    print(fb.obj_score())

    sample = [['choose', 'see', 'good', 'side', 'things', 'naive'],
            ['strategic', 'necessary'],
            ['learned', 'survive', 'everything']]
    
    print(nltk.pos_tag(sample[0]))

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
    
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import sentiwordnet as swn
    from nltk import sent_tokenize, word_tokenize, pos_tag
