# 데이터 교육 복습 6

#50. sklearn LabelEncoder, OneHotEncoder 연습
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

def encoder_test(encoder, itmes):
  '''
  encoder : Label, OneHot Encoder List, items : LabelEncoding 할 항목 리스트
  '''
  print(encoder[0].fit(items))
  labels = encoder[0].transform(items)
  print(labels)
  print(labels.classes_)
  print(labels.inverse_transform([4, 3, 3, 2]))
  labels = labels.reshape(-1, 1)
  print(encoder[1].fit(labels))
  oh_labels = encoder[1].transform(labels)
  print(oh_labels)
  oh_labels.toarray()
  df = pd.DataFrame({'items':itmes})
  print(df)
  print(pd.get_dummies(df))

#51. pandas 활용 연습
def pd_cats():
    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
    bins= [18, 25, 35, 60, 100]
    cats = pd.cut(ages, bins)
    print(cats)
    print(cats.codes)
    print(cats.categories)
    print(pd.value_counts(cats))
    group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
    p_cats =pd.cut(ages, bins, labels=group_names)
    print(p_cats)
    an_ages=[18, 25, 42, 73]
    an_cats = pd.cut(an_ages, bins)
    print(an_cats)
    an_ages=[18, 25, 42, 73]
    an_cats = pd.cut(an_ages, bins, right=False)
    print(an_cats)
    data = np.random.randn(1000)
    cats = pd.qcut(data, 4)
    print(cats)
    print(pd.value_counts(cats))

def stock_data():
    df = pd.read_csv('C:/Users/eleun/downloads/stock-data.csv')
    print(df.head())
    print('\n')
    print(df.info())
    df['new_Date'] = pd.to_datetime(df['Date'])
    print(df.head())
    print('\n')
    print(df.info())
    print('\n')
    print(type(df['new_Date'][0]))
    df.set_index('new_Date', inplace=True)
    print(df)
    dates = ['2019-01-01', '2020-03-01', '2021-06-01']
    ts_dates = pd.to_datetime(dates)
    print(ts_dates)
    print('\n')
    pr_day = ts_dates.to_period(freq='D')
    print(pr_day)
    pr_month = ts_dates.to_period(freq='M')
    print(pr_month)
    pr_year = ts_dates.to_period(freq='A')
    print(pr_year)
    ts_ms = pd.date_range(start='2019-01-01',
                        end=None,
                        periods=6,
                        freq='MS',
                        tz='Asia/Seoul')
    print(ts_ms)
    ts_me = pd.date_range('2019-01-01', periods=6,
                        freq='M',
                        tz='Asia/Seoul')
    print(ts_me)
    print('\n')
    ts_3m = pd.date_range('2019-01-01', periods=6,
                        freq='3M',
                        tz='Asia/Seoul')
    print(ts_3m)
    pr_m = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='M')
    print(pr_m)
    pr_h = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='H')
    print(pr_h)
    print('\n')
    pr_2h = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='2H')
    print(pr_2h)
    df2 = df.reset_index()
    df2['Year'] = df2['new_Date'].dt.year
    df2['Month'] = df2['new_Date'].dt.month
    df2['Day'] = df2['new_Date'].dt.day
    print(df2.head())
    df2['Date_yr'] = df2['new_Date'].dt.to_period(freq='A')
    df2['Date_m'] = df2['new_Date'].dt.to_period(freq='M')
    print(df2.head())
    df2.set_index('Date_m', inplace=True)
    print(df2.head())
    print(df.head())
    df_y = df['2018']
    print(df_y.head())
    df_ym = df['2018-07']
    print(df_ym)
    df_ym_cols = df.loc['2018-07', 'Start':'High']
    print(df_ym_cols)
    df_ymd = df['2018-07-02']
    print(df_ymd)
    df_ymd_range = df['2018-06-20':'2018-06-25']
    print(df_ymd_range)
    today = pd.to_datetime('2018-12-25')
    df['time_delta'] = today - df.index
    df.set_index('time_delta', inplace=True)
    df_180 = df['180 days':'189 days']
    print(df_180)

#52. sklearn vectorizer 연습
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def vect():
    x=[{'city':'seoul','temp':10.0}, {'city':'Dubai', 'temp':33.5}, {'city':'LA','temp':20.0}]
    print(x)
    vec=DictVectorizer(sparse=False)
    print(vec.fit_transform(x))
    print(vec)
    vec_spars_true = DictVectorizer()
    x_sparse = vec_spars_true.fit_transform(x)
    x_sparse
    print(x_sparse)
    print(x_sparse.toarray())
    v = DictVectorizer(sparse=False)
    D = [{'foo':1, 'bar':2},{'foo':3, 'baz':1}]
    X = v.fit_transform(D)
    print(X)
    print(v.get_feature_names_out())
    v=DictVectorizer()
    X=v.fit_transform(D)
    X
    print(X)
    text=['떴다 떴다 비행기 날아라 날아라',
        '높이 높이 날아라 우리 비행기',
        '내가 만든 비행기 날아라 날아라',
        '멀리 멀리 날아라 우리 비행기']
    vec2 = CountVectorizer()
    t = vec2.fit_transform(text).toarray()
    print(t)
    print(vec2.get_feature_names_out())
    t1 = pd.DataFrame(t, columns=vec2.get_feature_names_out())
    print(t1)
    tfidf = TfidfVectorizer()
    x2 = tfidf.fit_transform(text)
    x2
    print(x2.toarray())
    x3 = pd.DataFrame(x2.toarray(), columns=tfidf.get_feature_names_out())
    print(x3)
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age','fare']]
    df['ten'] = 10
    print(df.head())
    def add_10(n):
        return n +10
    def add_two_obj(a, b):
        return a+ b
    print(add_10(10))
    print(add_two_obj(10, 10))
    sr1 = df['age'].apply(add_10)
    print(sr1.head())
    print('\n')
    sr2 = df['age'].apply(add_two_obj, b=10)
    print(sr2.head())
    print('\n')
    sr3 = df['age'].apply(lambda x: add_10(x))
    print(sr3.head())

#53.pandas, seaborn dataset 활용 연습
def apl():
    df = pd.DataFrame([[4,9]] *3 , columns=['A','B'])
    print(df)
    print(df.apply(np.sqrt))
    print(df.apply(np.sum, axis=0))
    print(df.apply(lambda x:x.A*10, axis=1))
    print(df.apply(lambda x: x.B*10, axis=1))
    print(df.apply(lambda x:(x.A*10 + x.B*10), axis=1))

def titanic_1():
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age', 'fare']]
    print(df.head())
    def missing_value(series):
        return series.isnull()
    result = df.apply(missing_value, axis= 0)
    print(result.head())
    print(type(result))
    def min_max(x):
        return x.max() - x.min()
    result = df.apply(min_max)
    print(result)
    print(type(result))

def titanic_2():
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age', 'fare']]
    df['ten'] = 10
    print(df.head())
    def add_two_obj(a, b):
        return a+ b
    df['add'] = df.apply(lambda x: add_two_obj(x['age'], x['ten']), axis=1)
    print(df.head())

def titanic_3():
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[0:4, 'survived':'age']
    print(df)
    columns = list(df.columns.values)
    print(columns)
    columns_sorted = sorted(columns)
    df_sorted = df[columns_sorted]
    print(df_sorted)
    columns_reversed = list(reversed(columns))
    df_reversed = df[columns_reversed]
    print(df_reversed)
    columns_customed = ['pclass', 'sex', 'age', 'survived']
    df_customed = df[columns_customed]
    print(df_customed)

def str_prac():
    df = pd.read_excel('C:/Users/eleun/Downloads/주가데이터.xlsx', engine='openpyxl')
    print(df.head())
    print(df.dtypes)
    df['연월일'] = df['연월일'].astype('str')
    dates = df['연월일'].str.split('-')
    print(dates.head())
    df['연'] = dates.str.get(0)
    df['월'] = dates.str.get(1)
    df['일'] = dates.str.get(2)
    print(df.head())

def movies_str():
    movies = pd.read_table('C:/Users/eleun/Downloads/movies.txt', sep='::', header=None, names=['movie_id', 'title', 'genres'], encoding='latin-1')
    movies.genres.str.split('|')
    all_genres= []
    for x in movies.genres:
        all_genres.extend(x.split('|'))
    genres = pd.unique(all_genres)
    print(genres)
    print(len(genres))
    print(len(movies))
    zero_matrix = np.zeros((len(movies), len(genres)))
    print(zero_matrix)
    print(zero_matrix.shape)
    df_movies = pd.DataFrame(zero_matrix, columns=genres)
    print(df_movies.head())
    print(movies.head())
    print(movies.genres.str.split('|')[3])
    print(df_movies.columns)
    print(df_movies.columns.get_indexer(movies.genres.str.split('|')[3]))
    for i, gen in enumerate(movies.genres):
        df_movies.iloc[i, df_movies.columns.get_indexer(gen.split('|'))]=1
    print(df_movies.head())
    print(df_movies.loc[0, ['Comedy', 'Fantasy']])

def mask_():
    titanic = sns.load_dataset('titanic')
    mask1 = (titanic.age >= 10) & (titanic.age < 20)
    df_teenage = titanic.loc[mask1, :]
    print(df_teenage.head())
    mask2 = (titanic.age < 10) & (titanic.sex == 'female')
    df_female_under10 = titanic.loc[mask2, :]
    print(df_female_under10.head())
    mask3 = (titanic.age < 10) | (titanic.age >=  60)
    df_under10_morethan60 = titanic.loc[mask3, ['age', 'sex', 'alone']]
    print(df_under10_morethan60.head())
    
def isin_():
    titanic = sns.load_dataset('titanic')
    pd.set_option('display.max_columns', 10)
    mask3 = titanic['sibsp'] == 3
    mask4 = titanic['sibsp'] == 4
    mask5 = titanic['sibsp'] == 5
    df_boolean = titanic[mask3|mask4|mask5]
    print(df_boolean.head())
    isin_filter = titanic['sibsp'].isin([3,4,5])
    df_isin = titanic[isin_filter]
    print(df_isin.head())

def concat_():
    df1 = pd.DataFrame({'a':['a0','a1','a2','a3'],
                    'b':['b0','b1','b2','b3'],
                    'c':['c0','c1','c2', 'c3']},index=[0,1,2,3])
    df2 = pd.DataFrame({'a':['a2','a3','a4','a5'],
                        'b':['b2','b3','b4','b5'],
                        'c':['c2','c3','c4','c5'],
                        'd':['d2','d3','d4','d5']}, index=[2,3,4,5])
    print(df1, '\n')
    print(df2, '\n')

    result1 = pd.concat([df1, df2])
    print(result1, '\n')

    result2 = pd.concat([df1, df2], ignore_index=True)
    print(result2, '\n')

    result3 = pd.concat([df1, df2], axis=1)
    print(result3, '\n')

    result3_in = pd.concat([df1, df2], axis=1, join='inner')
    print(result3_in, '\n')

    sr1 = pd.Series(['e0','e1','e2','e3'], name='e')
    sr2 = pd.Series(['f0','f1','f2'], name='f', index=[3,4,5])
    sr3 = pd.Series(['g0','g1','g2','g3'], name='g')

    result4 = pd.concat([df1, sr1], axis=1)
    print(result4, '\n')

    result5 = pd.concat([df2, sr2], axis=1, sort=True)
    print(result5, '\n')

    result6 = pd.concat([sr1, sr3], axis=1)
    print(result6, '\n')

    result7 = pd.concat([sr1, sr3], axis=0)
    print(result7, '\n')

def any_all():
    bool_list = [True, True, False]
    print(all(bool_list))
    print(any(bool_list))
    list_data = [1,2,3,4,5]
    print(all(i<10 for i in list_data))
    print(any(i<10 for i in list_data))
    print(any(i > 3 for i in list_data))
    print(all(i > 3 for i in list_data))
    np.random.seed(12)
    data_ = pd.DataFrame(np.random.randn(1000, 4))
    print(data_.head())
    col = data_[2]
    print(col[np.abs(col)>3])
    print((np.abs(data_)>3).any(0))
    print((np.abs(data_)>3).any(1))
    print(data_[(np.abs(data_)>3).any(1)])
    print(data_[[0, 1]])
    data_= pd.DataFrame(np.arange(12).reshape((3,4)), index=['a', 'b', 'c'], columns=['k', 'l', 'm', 'n'])
    data_
    print((data_>9).any(0))
    print(data_[(data_>9).any(1)])
    print(data_[[False, False, True]])
    print(data_[['m', 'n']])
    print(data_.loc[:, (data_>9).any(0)])

def merge_():
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.unicode.east_asian_width', True)
    df1 = pd.read_excel('C:/Users/eleun/Downloads/stock price.xlsx', engine='openpyxl')
    df2= pd.read_excel('C:/Users/eleun/Downloads/stock valuation.xlsx', engine='openpyxl')

    print(df1)
    print('\n')
    print(df2)

    merge_inner = pd.merge(df1, df2)
    print(merge_inner)
    print('\n')
    merge_outer = pd.merge(df1, df2, how='outer', on='id')
    print(merge_outer)
    print('\n')
    merge_left = pd.merge(df1, df2, how='left', left_on='stock_name', right_on='name')
    print(merge_left)
    print('\n')
    merge_right = pd.merge(df1, df2, how='right', left_on='stock_name', right_on='name')
    print(merge_right)
    print('\n')
    price = df1[df1['price'] < 50000]
    print(price.head())
    print('\n')

    value = pd.merge(price, df2)
    print(value)

def merge_2():
    left = pd.DataFrame({'key':['K0', 'K4', 'K2', 'K3'],
                     'A':['A0','A1', 'A2','A3'],
                     'B':['B0', 'B1', 'B2','B3']})

    right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                        'C':['C0','C1','C2','C3'],
                        'D':['D0','D1','D2','D3']})
    print(left)
    print(right)
    print(pd.merge(left, right, on='key'))
    print(pd.merge(left, right, how='left', on='key'))
    print(pd.merge(left, right, how='right', on='key'))
    print(pd.merge(left, right, how='outer', on='key'))
    print(pd.merge(left, right, how='inner', on='key'))
