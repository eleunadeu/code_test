# 데이터 교육 복습

#54. matplotlib 폰트 한글 지정
import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
plt.rcParams['axes.uunicode_minus'] = False

if platform.system() == 'Darwin':
  rc('font', family = 'AppleGothic')
elif platform.system() == 'Window':
  path = "c:/Windows/Fonts/malgun.ttf"
  font_name = font_manager.FontProperties(fname=path).get_name()
  rc('font', family=font_name)
else:
  font_path = 'IropkeBatangM.woff'
  fontprop = font_manager.FontProperties(fname=path, size=10)
  font_name = fontprop.get_name()
  rc('font', family=font_name)

#55. csv 파일 읽어서 pandas로 데이터 활용
import pandas as pd
import numpy as np
def seoul_cctv_pop():
    cctv_seoul = pd.read_csv('C:/Users/eleun/Downloads/seoul_cctv.csv', encoding='cp949', skiprows=[0,2], header=[0], thousands=',')
    POP_seoul = pd.read_table('C:/Users/eleun/Downloads/report.txt', skiprows=[0,1,3], thousands=',')
    print(cctv_seoul.head())
    print(cctv_seoul.info())
    print(POP_seoul.head())
    print(POP_seoul.info())
    cctv_seoul.rename(columns={cctv_seoul.columns[0]:'구별'}, inplace=True)
    pop_seoul = POP_seoul.loc[:, ['자치구','계', '계.1', '계.2', '65세이상고령자']]
    print(pop_seoul.head())
    pop_seoul.columns=['구별', '인구수', '한국인', '외국인', '고령자']
    print(pop_seoul.head())
    cctv_seoul.sort_values(by='총계', ascending=False).head()
    cctv_seoul.dropna(axis=1, inplace=True)
    cctv_seoul['최근증가율'] = \
    (cctv_seoul['2021년']+cctv_seoul['2020년']+cctv_seoul['2019년']+cctv_seoul['2018년']+cctv_seoul['2017년'])/(cctv_seoul['2016년']+cctv_seoul['2015년']+cctv_seoul['2014년']+cctv_seoul['2012년 이전'])*100
    print(cctv_seoul.head())
    print(cctv_seoul.sort_values(by='최근증가율', ascending=False).head())
    pop_seoul['외국인비율'] = (pop_seoul['외국인']/pop_seoul['인구수'])*100
    pop_seoul['고령자비율'] = (pop_seoul['고령자']/pop_seoul['인구수'])*100
    print(pop_seoul.head())
    print(pop_seoul.sort_values(by='고령자비율', ascending=False).head(5))
    print(pop_seoul.sort_values(by='외국인비율', ascending=False).head(5))
    print(pop_seoul.info())
    print(cctv_seoul.info())
    data = pd.merge(cctv_seoul, pop_seoul, on='구별')
    print(data.head())
    print(data.info())
    gu_list = list(data.구별)
    for i in pop_seoul.구별:
        if i in gu_list:
            pass
        else:
            print(i)
    for i in cctv_seoul.구별:
        if i in gu_list:
            pass
        else:
            print(i)
    cctv_seoul.loc[1, '구별'] = '중구'
    data = pd.merge(cctv_seoul, pop_seoul, on='구별')
    print(data.head())
    print(data.info())
    data_result = data.drop(['2012년 이전', '2014년', '2015년', '2016년', '2017년', '2018년', '2019년', '2020년', '2021년'], axis=1)
    print(data_result.head())
    print(data_result.sort_values(by='총계', ascending=False).head())
    data_result.set_index('구별', inplace=True)
    print(data_result.sort_values(by='총계', ascending=False).head())
    data_result.to_csv('C:/Users/eleun/.jupyter/cctv_pop.csv')

#56. 정리한 데이터 matplotlib 사용 시각화하기
def cctv_pop_graph():
    data_result = pd.read_csv('./cctv_pop.csv')
    data_result.set_index('구별', inplace=True)
    x = data_result.index.to_numpy()
    y = data_result.총계.to_numpy()
    plt.rc('font', family=font_name)
    plt.figure(figsize=(10,10))
    plt.yticks()
    plt.barh(x,y)
    plt.show()
    
    plt.figure(figsize=(10,10))
    data_result['총계'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
    plt.yticks()
    plt.ylabel('구별')
    plt.show()

    data_result['CCTV비율'] = data_result['총계']/data_result['인구수'] * 100
    data_result['CCTV비율'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
    plt.yticks()
    plt.ylabel('구별')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(data_result['인구수'], data_result['총계'], s=50)
    plt.xlabel('인구수')
    plt.ylabel('CCTV')
    plt.grid()
    plt.show()

    df_sort = data_result.sort_values(by='총계', ascending=False)
    print(df_sort.head(10))
    plt.figure(figsize=(6,6))
    plt.scatter(data_result['인구수'], data_result['총계'], s=50)
    for n in range(3):
        plt.text(df_sort['인구수'][n]*1.02, df_sort['총계'][n]*0.98, df_sort.index[n], fontsize=15)
    plt.xlabel('인구수')
    plt.ylabel('CCTV')
    plt.grid()
    plt.show()

    fp1 = np.polyfit(data_result.인구수, data_result.총계, 1)
    func = np.poly1d(fp1)
    data_result['predict'] = func(data_result.인구수)
    plt.figure(figsize=(6,6))
    plt.scatter(data_result['인구수'], data_result['총계'], s=50)
    for n in range(3):
        plt.text(df_sort['인구수'][n]*1.02, df_sort['총계'][n]*0.98, df_sort.index[n], fontsize=15)
    plt.plot(data_result.인구수, data_result.predict, ls='-', lw=1, color='r')
    plt.xlabel('인구수')
    plt.ylabel('CCTV')
    plt.grid()
    plt.show()

def cctv_pop_graph2():
    from sklearn.linear_model import LinearRegression
    data_result = pd.read_csv('./cctv_pop.csv')
    data_result.set_index('구별', inplace=True)
    print(data_result.총계.shape, data_result.인구수.shape)
    x = data_result.인구수[:, np.newaxis]
    x.shape
    print(x.shape)
    y = data_result.총계.values
    print(y)
    model = LinearRegression()
    model.fit(x, y)
    print(model.intercept_, model.coef_)
    fp1 = np.polyfit(data_result.인구수, data_result.총계, 1)
    func = np.poly1d(fp1)
    data_result['predict'] = func(data_result.인구수)
    data_result['predict_sklearn'] = model.predict(x)
    print(data_result.head())
    data_result['res'] = data_result.총계 - data_result.predict_sklearn
    print(data_result.head())
    data_sort = data_result.sort_values(by='res', ascending=False)
    print(data_sort.head(2))
    plt.figure(figsize=(7, 5))
    plt.scatter(data_sort.인구수, data_sort.총계, c = data_sort.res, s=50)
    for n in range(3):
        plt.text(data_sort['인구수'][n]*1.02, data_sort['총계'][n]*0.96, data_sort.index[n], fontsize=15)
        plt.text(data_sort['인구수'][24-n]*1.02, data_sort['총계'][24-n]*0.96, data_sort.index[24-n], fontsize=15)
    plt.plot(data_sort.인구수, data_sort.predict_sklearn, ls='-', lw=1, color='r')
    plt.xlabel('인구수')
    plt.ylabel('CCTV')
    plt.colorbar()
    plt.grid()
    plt.show()

#57. Group 기능 연습
import seaborn as sns
def group_make(df, group_opt):
  '''
  df : 데이터 프레임
  group_opt : group으로 바꿀 column name
  '''
  grouped = df.groupby([group_opt])
  return grouped

def group_tr():
  titanic = sns.load_dataset('titanic')
  df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]
  print('승객 수 :', len(df))
  print(df.head())
  print('\n')
  grouped = group_make(df, 'class')
  print(grouped)
  for key, group in grouped:
        print('* key :', key)
        print('* numner :', len(group))
        print(group.head())
        print('\n')
  average = grouped.mean()
  print(average)
  group3 = grouped.get_group('Third')
  print(group3.head())
  grouped_two = df.groupby(['class', 'sex'])
  for key, group in grouped_two:
      print('* key :', key)
      print('* number :', len(group))
      print(group.head())
      print('\n')
  average_two = grouped_two.mean()
  print(average_two)
  print('\n')
  print(type(average_two))
  group3f = grouped_two.get_group(('Third','female'))
  print(group3f.head())

def group_tr2():
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]

    grouped = group_make(df, 'class')
    std_all = grouped.std()
    print(std_all)
    print('\n')
    print(type(std_all))
    print('\n')

    std_fare = grouped.fare.std()
    print(std_fare)
    print('\n')
    print(type(std_fare))
    print('\n')

    def min_max(x):
        return x.max() - x.min()
    agg_minmax = grouped.agg(min_max)
    print(agg_minmax.head())

    agg_all = grouped.agg(['min', 'max'])
    print(agg_all.head())
    print('\n')

    agg_sep = grouped.agg({'fare':['min','max'], 'age':'mean'})
    print(agg_sep.head())

def group_tr3():
    df = pd.DataFrame({'Animal':['Falcon', 'Falcon',
                             'Parrot', 'Parrot'],
                   'Max_Speed':[380., 370., 24., 26.]})
    print(df.groupby(['Animal']).mean)
    print(df.Max_Speed.groupby(df['Animal']).mean())

def group_tr4():
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]
    grouped = group_make(df, 'class')
    age_mean = grouped.age.mean()
    print(age_mean)
    print('\n')
    age_std = grouped.age.std()
    print(age_std)
    print('\n')
    for key, group in grouped.age:
        group_zscore = (group - age_mean.loc[key])/age_std.loc[key]
        print('* origin :', key)
        print(group_zscore.head())
        print('\n')
    def z_score(x):
        return (x- x.mean())/x.std()
    age_zscore = grouped.age.transform(z_score)
    print(age_zscore.loc[[1,9,0]])
    print('\n')
    print(len(age_zscore))
    print('\n')
    print(age_zscore.loc[0:9])
    print('\n')
    print(type(age_zscore))

def group_tr5():
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]
    grouped = group_make(df, 'class')
    grouped_filter = grouped.filter(lambda x: len(x) >= 200)
    print(grouped_filter.head())
    print('\n')
    print(type(grouped_filter))
    age_filter = grouped.filter(lambda x: x.age.mean() < 30)
    print(age_filter.tail())
    print('\n')
    print(type(age_filter))

def group_tr6():
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]
    grouped = group_make(df, 'class')
    agg_grouped = grouped.apply(lambda x: x.describe())
    print(agg_grouped)

    def z_score(x):
        return (x - x.mean())/x.std()
    
    age_zscore = grouped.age.apply(z_score)
    print(age_zscore.head())

    age_filter = grouped.apply(lambda x: x.age.mean() < 30)
    print(age_filter)
    print('\n')
    for x in age_filter.index:
        if age_filter[x] == True:
            age_filter_df = grouped.get_group(x)
            print(age_filter_df.head())
            print('\n')
    grouped = df.groupby(['class', 'sex'])
    gdf = grouped.mean()
    print(gdf)
    print('\n')
    print(type(gdf))
    print(gdf.loc['First'])
    print(gdf.loc[('First', 'female')])
    print(gdf.xs('male', level='sex'))

def group_tr7():
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 20)
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]
    print(df.head())
    print('\n')

    pdf1 = pd.pivot_table(df, index='class', columns='sex', values='age', aggfunc='mean')
    print(pdf1.head())

    pdf2 = pd.pivot_table(df, index='class', columns='sex', values='survived', aggfunc=['mean', 'sum'])
    print(pdf2.head())

    pdf3 = pd.pivot_table(df, index=['class', 'sex'], columns='survived', values=['age', 'fare'], aggfunc=['mean', 'max'])

    pd.set_option('display.max_columns', 10)
    print(pdf3.head())
    print('\n')
    print(pdf3.index)
    print(pdf3.columns)

    print(pdf3.xs('First'))
    print(pdf3.xs(('First', 'female')))
    print(pdf3.xs('male', level='sex'))
    print(pdf3.xs(('Second', 'male'), level=[0, 'sex']))
    print(pdf3.xs('mean', axis=1))
    print(pdf3.xs(('mean', 'age'), axis=1))
    print(pdf3.xs('mean', axis=1))
    print(pdf3.xs(('mean', 'age'), axis=1))
    print(pdf3.xs(1, level='survived', axis=1))
    print(pdf3.xs(('max', 'fare', 0), level=[0, 1, 2], axis=1))

def group_tr8():
    df = pd.DataFrame({'Animal':['Falcon', 'Falcon',
                             'Parrot', 'Parrot'],
                   'Max_Speed':[380., 370., 24., 26.]})
    print(df)
    arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
          ['Captive', 'Wild', 'Captive', 'Wild']]
    index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
    print(index)
    df = pd.DataFrame({'Max_Speed':[390., 350., 30., 20.]}, index=index)
    print(df)
    print(df.groupby(level=0).mean())
    grouped = df.Max_Speed.groupby(level=0)
    print(grouped.mean())
    print(df.groupby(level='Animal').mean())
    print(df.groupby(level=1).mean())

def group_tr9():
    df = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
                   'key2':['one', 'two', 'one', 'two', 'one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
    print(np.random.randint(1, 6))
    print(np.random.rand(6))
    print(np.random.rand(2,3))
    print(np.random.randn(3, 2))
    print(df['data1'].groupby(df['key1']).mean())
    means = df['data1'].groupby([df['key1'], df['key2']]).mean()
    print(means)
    print(means.unstack)
    states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
    years = np.array([2005,2005,2006,2005,2006])
    print(df['data1'].groupby([states, years]).mean())
    print(df.groupby(['key1', 'key2']).mean())
    print(df.groupby(['key1', 'key2']).size())
    for (k1, k2), group in df.groupby(['key1', 'key2']):
        print((k1, k2))
        print(group)
    k = list(df.groupby('key1'))
    print(k)
    print(k[0][1])
    print(type(k[0][1]))
    pieces = dict(list(df.groupby('key1')))
    print(pieces)
    print(df.groupby('key1')['data1'])
    print(df.groupby('key1')[['data1']])
    people = pd.DataFrame(np.random.randn(5,5),
                      columns=['a', 'b', 'c', 'd', 'e'],
                      index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
    people.iloc[2:3, [1, 2]] = np.nan
    print(people)
    mapping = {'a':'red', 'b':'red', 'c':'blue',
           'd':'blue', 'e':'red', 'f':'orange'}
    print(people.groupby(mapping, axis=1).sum())
    map_series = pd.Series(mapping)
    print(map_series)
    print(people.groupby(map_series, axis=1).count())
    print(people.groupby(len).sum())
    key_list = ['one', 'one', 'one', 'two', 'two']
    people.groupby([len, key_list]).mean()

def group_tr10():
    df = pd.DataFrame({
        'city':['부산', '부산', '부산', '부산', '서울', '서울', '서울'],
        'fruits':['apple', 'orange', 'banana', 'banana', 'apple', 'apple', 'banana'],
        'price':[100, 200, 250, 300, 150, 200, 400],
        'quantity':[1, 2, 3, 4, 5, 6, 7]})
    print(df)
    print(df.groupby(['city']).mean())
    print(df.groupby(['city', 'fruits']).mean())
    print(df.groupby(['city', 'fruits'], as_index=False).mean())
    print(df.groupby(['city']).get_group('부산'))
    print(df.groupby(['city']).size())
    print(type(df.groupby(['city']).size()))
    def my_mean(s):
        return np.mean(s)

    print(df.groupby(['city']).agg({'price':my_mean, 'quantity':np.sum}))
    def total_series(d):
        return d.price * d.quantity
    
    print(df.groupby(['city', 'fruits']).apply(total_series))
    print(df.groupby(['city', 'fruits'], as_index=False).apply(lambda d: (d.price * d.quantity).sum()))

    df = pd.DataFrame({'상품번호':['P1', 'P1', 'P2', 'P2'],
                  '수량':[2, 3, 5, 10]})
    print(df)
    print(df.groupby(by=['상품번호'], as_index=False).min())
    print(df.groupby(by=['상품번호'], as_index=False).max())
    print(df.groupby(by=['상품번호'], as_index=False).mean())
    print(df.groupby(by=['상품번호'], as_index=False).sum())
    print(df.groupby(by=['상품번호'], as_index=False).count())

#58. datetime, timedelta, dateutil.parser 연습
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
def time_series():
    now = datetime.now()
    print(now)
    print(now.year, now.month, now.day)
    delta = datetime(2022, 7, 1) - datetime(2023, 1, 11)
    print(delta)
    print(delta.days, delta.seconds)
    start = datetime(2022, 7, 1)
    print(start + timedelta(12))
    print(start - 2 * timedelta(12))
    print(start.strftime('%Y-%m-%d'))
    value = '2011-01-03'
    print(datetime.strptime(value, '%Y-%m-%d'))
    print(parse('2023-01-03'))
    print(parse(value))
    print(parse('Jan 31, 1997 10:45 PM'))
    date_strs = ['2011-07-06 12:00:00', '2011-08-06 12:00:00']
    result = pd.to_datetime(date_strs)
    print(type(result))
    print(result)
    idx = pd.to_datetime(date_strs + [None])
    print(idx)
    print(idx[2])
    print(pd.isnull(idx))
    dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
    datetime(2011, 1, 7), datetime(2011, 1, 8), 
    datetime(2011, 1, 10), datetime(2011, 1, 12)]
    ts = pd.Series(np.random.randn(6), index=dates)
    print(ts)
    print(ts.index)
    print(ts + ts[::2])
    print(ts.index[0]) 
    stamp = ts.index[2]
    print(stamp)
    print(ts['1/10/2011'])
    print(ts['20110110'])
    longer_ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    print(pd.date_range('1/1/2000', periods=1000))
    print(longer_ts)
    print(longer_ts['2002'])
    print(longer_ts['2002-05'])
    print(longer_ts['2002-05':])
    print(longer_ts['2002-05':'2002-06'])
    print(longer_ts.truncate(after='1/9/2002'))
    dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
    long_df = pd.DataFrame(np.random.randn(100, 4), index=dates, columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print(long_df.loc['5-2001'])
    dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000', '1/3/2000'])
    dup_ts = pd.Series(np.arange(5), index=dates)
    print(dup_ts)
    print(dup_ts.index.is_unique)
    print(dup_ts['1/3/2000'])
    print(dup_ts['1/2/2000'])
    grouped = dup_ts.groupby(level=0)
    print(grouped.count())
    print(grouped.mean())
    print(ts)
    index = pd.date_range('2012-04-01', '2012-06-01')
    print(index)
    print(pd.date_range('2012-04-01', periods=20))
    print(pd.date_range(end= '2012-04-01', periods=20))
    print(pd.date_range('2000-01-01', '2000-12-01', freq='BM'))
    print(pd.date_range('2012-05-02 12:56:31', periods=5))
    print(pd.date_range('2012-05-02 12:56:31', periods=5, normalize=True))
    print(pd.date_range('2023-01-01', '2023-01-07 23:59', freq='3h'))
    print(pd.date_range('2023-01-01', periods=10, freq='1h30min'))
    ts = pd.Series(np.random.randn(4), index=pd.date_range('1/1/2000', periods=4, freq='M'))
    print(ts)
    print(ts.shift(2))
    print(ts.shift(-2))
    print(ts.shift(1))
    print(ts / ts.shift(1) - 1)
    print(ts.shift(2, freq='M'))
    print(ts.shift(2))
    print(ts.shift(3, freq='D'))
    rng = pd.date_range('2000-01-01', periods=100, freq='D')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)
    print(len(rng))
    print(ts)
    print(ts.resample('M').mean())
    print(ts.resample('M', kind='periods').mean())
    rng = pd.date_range('2000-01-01', periods=12, freq='T')
    ts = pd.Series(np.arange(12), index=rng)
    print(ts)
    print(ts.resample('5min').sum())
    print(ts.resample('5min', closed='left').sum())
    print(ts.resample('5min', closed='right').sum())
    print(ts.resample('5min', closed='right', label='right').sum())
    print(ts.resample('5min', closed='right', label='right', loffset='-1s').sum())
    frame = pd.DataFrame(np.random.randn(2, 4),
    index=pd.date_range('1/1/2000', periods=2, freq='W-WED'), 
    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print(frame)
    df_daily = frame.resample('D').asfreq()
    print(df_daily)
    print(frame.resample('D').ffill())
    print(frame.resample('D').ffill(limit=2))
    df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/timeTest.csv')
    print(df.head())
    print(df.info())
    df['times'] = pd.to_datetime(df.Yr_Mo_Dy)
    print(df.head())
    print(df.info())
    print(df.times.dt.year.unique())
    print(df.loc[3, 'times'].year)
    print(df.loc[3, 'times'].month)
    print(df.loc[3, 'times'].day)

def date_time():
    df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/timeTest.csv')
    df['times'] = pd.to_datetime(df.Yr_Mo_Dy)
    from datetime import datetime
    from dateutil.parser import parse
    def notimemuchine(x):
        if x > datetime(2000, 1, 1):
            temp_year = x.year - 100
            temp_month = x.month
            temp_day = x.day
            return parse(str(temp_year)+'/'+str(temp_month)+'/'+str(temp_day))
        else:
            return x
    df['CorrectTime'] = df['times'].apply(notimemuchine)
    df['YMD'] = df.loc[:, 'times'].apply(lambda x: x.replace(year=x.year -100) if x.year > 2000 else x)
    import datetime
    todays_date = datetime.date.today()
    print(todays_date)
    print(todays_date.replace(year=2000))
    from datetime import datetime
    def fix_year(x):
        year = x.year - 100 if x.year > 1989 else x.year
        return datetime(year, x.month, x.day)
    df['YMD-2'] = df['times'].apply(fix_year)
    print(df.head())
    print(df.RPT.groupby(df.YMD.dt.year).mean())
    print(df['RPT'].groupby(df.YMD.dt.year).mean())
    print(df.resample('Y', on='YMD').agg({'RPT':['size', 'mean']}))
    print(df.resample('Y', on='YMD').agg({'RPT':['size','mean']}).droplevel(level=0, axis=1))
    print(df.groupby(df.YMD.dt.month).mean)
    print(df.groupby(df.YMD.dt.to_period('M')).mean())

#59. group 활용 연습2
def gr_tr():
    data = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
    idx = [['idx1', 'idx1', 'idx2', 'idx2'], ['row1', 'row2', 'row3', 'row3']]
    col = [['col1', 'col1', 'col2', 'col2'], ['val1', 'val2', 'val3', 'val4']]
    df1 = pd.DataFrame(data=data, index=idx, columns=col)
    print(df1)
    print(df1.droplevel(axis=0, level=0))
    print(df1.droplevel(axis=0, level=1))
    print(df1.droplevel(axis=1, level=0))
    print(df1.droplevel(axis=1, level=1))
    df2 = pd.DataFrame({'key':['a', 'b', 'c']*4, 'value':np.arange(12)})
    print(df2)
    print(df2.groupby('key').mean())
    bill_tip = pd.read_csv('C:/Users/eleun/Downloads/bill_tips.csv')
    print(bill_tip)
    bill_tip['pct'] = bill_tip['tip']/bill_tip['total_bill']
    print(bill_tip.head())
    print(bill_tip.info())
    grouped = bill_tip.groupby(['day', 'smoker'])
    print(grouped['pct'].agg(['mean', 'std']))
    print(grouped['pct'].agg([('평균', 'mean'), ('표준편차','std')]))
    print(grouped['pct'].agg([('평균', np.mean), ('표준편차',np.std)]))
    func= ['count', 'mean', 'max']
    print(grouped['pct', 'total_bill'].agg(func))
    func_tuple = [('표준편차', np.std), ('분산', np.var)]
    print(grouped['pct', 'total_bill'].agg(func_tuple))
    print(grouped.agg({'tip':np.max, 'size':np.sum}))
    print(grouped.agg({'pct':[np.min, np.max, np.mean, np.std], 'size':np.sum}))
    grouped = bill_tip.groupby('day', as_index=False).mean()
    print(grouped)

#60. seaborn 활용 연습
def sb_tr():
    import seaborn as sns
    titanic = sns.load_dataset('titanic')
    print(titanic.head())
    print('\n')
    print(titanic.info())
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sns.regplot(x='age', y='fare', data=titanic, ax=ax1)
    sns.regplot(x='age', y='fare', data=titanic, ax=ax2, fit_reg=False)
    plt.show()
    fig = plt.figure(figsize=(15, 5))
    ax3 = fig.add_subplot(1, 3, 1)
    ax4 = fig.add_subplot(1, 3, 2)
    ax5 = fig.add_subplot(1, 3, 3)
    sns.distplot(titanic['fare'], ax=ax3)
    sns.kdeplot(x='fare', data=titanic, ax=ax4)
    sns.histplot(x='fare', data=titanic, ax=ax5)
    ax3.set_title('titanic fare - hist/kde')
    ax4.set_title('titanic fare - kde')
    ax5.set_title('titanic fare - hist')
    plt.show()
    table = titanic.pivot_table(index=['sex'], columns=['class'], aggfunc='size')
    sns.heatmap(table, annot=True, fmt='d', cmap='YlGnBu', linewidth=.5, cbar=False)
    plt.show()
    recipes = pd.read_csv('C:/Users/eleun/Downloads/recipes_muffins_cupcakes.csv')
    print(recipes)
    figure = plt.figure()
    axes1 = figure.add_subplot(1, 2, 1)
    axes2 = figure.add_subplot(1, 2, 2)
    axes1.plot(recipes.Flour)
    axes2.plot(recipes.Milk)
    axes1.set_title('Flour')
    axes2.set_title('Milk')
    figure.suptitle('recipes')
    figure
    plt.show()
    sns.heatmap(recipes.corr(), linewidth=.1, vmax=.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
    plt.show()
    sns.pairplot(recipes, hue='Type')
    plt.show()
    sns.countplot(data=recipes, x='Egg')
    plt.show()
    recipes.plot('Milk', 'Sugar', kind='scatter', c='blue')
    plt.show()
    recipes.Sugar.plot(kind='hist', bins=10, color='blue')
    plt.show()
    sns.set_style('whitegrid')
    fig= plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sns.stripplot(x='class',
                y='age',
                data=titanic,
                ax=ax1)
    sns.swarmplot(x='class',
                y='age',
                data=titanic,
                ax=ax2)
    ax1.set_title('Strip Plot')
    ax2.set_title('Swarm Plot')
    plt.show()
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    sns.barplot(x='sex', y='survived', data=titanic, ax=ax1)
    sns.barplot(x='sex', y='survived', hue='class', data=titanic, ax=ax2)
    sns.barplot(x='sex', y='survived', hue='class', dodge=False, data=titanic, ax=ax3)
    ax1.set_title('titanic survived - sex')
    ax2.set_title('titanic survived - sex/class')
    ax3.set_title('titanic survived - sex/class(stacked)')
    plt.show()
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    sns.countplot(x='class', palette='Set1', data=titanic, ax=ax1)
    sns.countplot(x='class', hue='who',palette='Set2', data=titanic, ax=ax2)
    sns.countplot(x='class', hue='who', palette='Set1', dodge=False, data=titanic, ax=ax3)
    ax1.set_title('titanic class')
    ax2.set_title('titanic class - who')
    ax3.set_title('titanic class - who(stacked)')
    plt.show()
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    sns.boxplot(x='alive', y='age', data=titanic, ax=ax1)
    sns.boxplot(x='alive', y='age', hue='sex', data=titanic, ax=ax2)
    sns.violinplot(x='alive', y='age', data=titanic, ax=ax3)
    sns.violinplot(x='alive', y='age', hue='sex', data=titanic, ax=ax4)
    plt.show()
