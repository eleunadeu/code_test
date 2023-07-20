# 데이터 교육 복습 8

#61. pandas, matplotlib 활용 연습
def gh_tr():
    df = pd.read_excel('C:/Users/eleun/Downloads/남북한발전전력량.xlsx', engine='openpyxl')
    df_ns = df.iloc[[0,5], 3:]
    df_ns.index = ['South', 'North']
    df_ns.columns = df_ns.columns.map(int)
    print(df_ns.head())
    df_ns.plot()
    plt.show()
    tdf_ns = df_ns.T
    print(tdf_ns.head())
    tdf_ns.plot()
    plt.show()
    df_ns.plot(kind='bar')
    plt.show()
    tdf_ns.plot(kind='bar')
    plt.show()
    tdf_ns = tdf_ns.apply(pd.to_numeric)
    tdf_ns.plot(kind='hist')
    plt.show()
    df = pd.read_csv('C:/Users/eleun/Downloads/auto-mpg.csv', header=None)
    df.columns = ['mpg', 'cylinder', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
    df.plot(x='weight', y='mpg', kind='scatter')
    plt.show()
    df[['mpg', 'cylinder']].plot(kind='box')
    plt.show()
    df = pd.read_excel('C:/Users/eleun/Downloads/시도별 전출입 인구수.xlsx', header=0)#engine='openpyxl'은 엑셀 파일 안 열릴 때만 사용
    df = df.fillna(method='ffill')
    mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
    df_seoul = df[mask]
    df_seoul = df_seoul.drop(['전출지별'], axis=1)
    df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
    df_seoul.set_index('전입지', inplace=True)
    sr_one = df_seoul.loc['경기도']
    plt.plot(sr_one.index, sr_one.values)
    plt.plot(sr_one)
    plt.plot(sr_one.index, sr_one.values, label='인구수')
    plt.title('서울 -> 경기 인구 이동', size=15)
    plt.xlabel('기간', size=10)
    plt.ylabel('이동 인구수', size=10)
    plt.legend()

#62. matplotlib graph 활용 연습
def mat_plt():
    df = pd.read_excel('C:/Users/eleun/Downloads/시도별 전출입 인구수.xlsx', header=0)
    df = df.fillna(method='ffill')
    mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
    df_seoul = df[mask]
    df_seoul = df_seoul.drop(['전출지별'], axis=1)
    df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
    df_seoul.set_index('전입지', inplace=True)
    col_years = list(map(str, range(1970, 2018)))
    df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
    print(df_4.head())
    df_4 = df_4.transpose()
    print(df_4.head())
    plt.style.use('ggplot')
    df_4.index = df_4.index.map(int)
    df_4.plot(kind='area', stacked=False, alpha=0.2, figsize=(20, 10))
    plt.title('서울 -> 타시도 인구 이동', size=30)
    plt.ylabel('이동 인구 수', size=20)
    plt.xlabel('기간', size=20)
    plt.legend(loc='best', fontsize=15)
    plt.show()
    df_4.plot(kind='area', stacked=True, alpha=0.2, figsize=(20, 10))
    plt.title('서울 -> 타시도 인구 이동', size=30)
    plt.ylabel('이동 인구 수', size=20)
    plt.xlabel('기간', size=20)
    plt.legend(loc='best', fontsize=15)
    plt.show()
    df_4.plot(kind='bar', figsize=(20,10), width=0.7, color=['orange', 'green', 'skyblue', 'blue'])
    plt.show()
    col_years = list(map(str, range(2010, 2018)))
    df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
    df_4['합계'] = df_4.sum(axis=1)
    df_total = df_4[['합계']].sort_values(by='합계', ascending=True)
    plt.style.use('ggplot')
    df_total.plot(kind='barh', color='cornflowerblue', width=0.5, figsize=(10, 5))
    plt.title('서울 -> 타시도 인구 이동')
    plt.ylabel('전입지')
    plt.xlabel('이동 인구 수')
    plt.show()

def mat_plt_bar():
    plt.style.use('ggplot')
    plt.rcParams['axes.unicode_minus']=False
    df = pd.read_excel('C:/Users/eleun/Downloads/남북한발전전력량.xlsx', engine='openpyxl', convert_float=True)
    df = df.loc[5:9]
    df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
    df.set_index('발전 전력별', inplace=True)
    df = df.T
    df = df.rename(columns={'합계':'총발전량'})
    df['총발전량 - 1년'] = df['총발전량'].shift(1)
    df['증감률'] = ((df['총발전량']/df['총발전량 - 1년'])-1)*100
    ax1 = df[['수력', '화력']].plot(kind='bar', figsize=(20,10), width=0.7, stacked=True)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df.증감률, ls='--', marker='o', markersize=20, color='red', label='전년대비 증감률')
    ax1.set_ylim(0, 500)
    ax2.set_ylim(-50, 50)
    ax1.set_xlabel('연도', size=20)
    ax1.set_ylabel('발전량(억 KWh)')
    ax2.set_ylabel('전년 대비 증감률(%)')
    plt.title('북한 전력 발전량 (1990~2016)', size=20)
    ax1.legend(loc='upper right')
    plt.show()

def mat_plt_scatter_pie():
    plt.style.use('default')
    df = pd.read_csv('C:/Users/eleun/Downloads/auto-mpg.csv', header=None)
    df.columns= ['mpg', 'cylinders', 'displacement', 'horespower', 'weight', 'acceleration', 'model year', 'origin', 'name']
    df.plot(kind='scatter', x='weight', y='mpg', c='coral', s=10, figsize=(10,5))
    plt.title('Scatter Plot - mpg vs weight')
    plt.show
    plt.style.use('default')
    cylinders_size = df.cylinders/df.cylinders.max() *300
    df.plot(kind='scatter', x='weight', y='mpg', c='coral', figsize=(10,5), s=cylinders_size, alpha=0.3)
    plt.title('Scatter Plot: mpg-weight-cylinders')
    plt.show()
    plt.style.use('default')
    cylinders_size = df.cylinders/df.cylinders.max() *300
    df.plot(kind='scatter', x='weight', y='mpg', marker='+', cmap='viridis', figsize=(10,5), c=cylinders_size, s=50, alpha=0.3)
    plt.title('Scatter Plot: mpg-weight-cylinders')
    plt.savefig('./scatter.png')
    plt.savefig('./scatter_transparent.png', transparent=True)
    plt.show()
    df['count'] = 1
    df_origin = df.groupby('origin').sum()
    print(df_origin.head())
    df_origin.index = ['USA', 'EU', 'JAPAN']
    df_origin['count'].plot(kind='pie', figsize=(7,5), autopct='%1.1f%%', startangle=10, colors=['chocolate', 'bisque', 'cadetblue'])
    plt.title('Model Origin', size=20)
    plt.axis('equal')
    plt.legend(labels=df_origin.index, loc='upper right')
    plt.show()

#63. query 활용 연습
def train_query():
    data = {'age':[10, 10, 21, 22], 'weight':[20, 30, 60, 70]}
    df = pd.DataFrame(data)
    print(df.query('age == 10'))
    expr_str = 'age == 10'
    print(df.query(expr_str))
    expr_str = 'age != 10'
    print(df.query(expr_str))
    expr_str = 'age == [21, 22]'
    print(df.query(expr_str))
    expr_str = 'age in [21, 22]'
    print(df.query(expr_str))
    expr_str = 'age not in [21, 22]'
    print(df.query(expr_str))
    expr_str = '(age ==10) and (weight >= 30)'
    print(df.query(expr_str))
    num_age = 10
    num_weight = 30
    expr_str = '(age == @num_age) and (weight >= @num_weight)'
    print(df.query(expr_str))
    expr_str = f'(age == {num_age}) and (weight >= {num_weight})'
    print(df.query(expr_str))
    def max_user(x, y):
        return max(x, y)
    print(max_user(3, 123))
    expr_str = 'age >= @max_user(1, 22)'
    print(df.query(expr_str))
    expr_str = 'index > 1'
    print(df.query(expr_str))
    data = {'name':['White tiger', 'Tiger black', 'Red tiger'], 'age':[5, 7, 9]}
    df = pd.DataFrame(data)
    print(df)
    expr_str = "name.str.contains('tiger')"
    print(df.query(expr_str, engine='python'))
    expr_str = "name.str.contains('tiger', case=False)"
    print(df.query(expr_str, engine='python'))
    expr_str = "name.str.startswith('Tiger')"
    print(df.query(expr_str, engine='python'))
    expr_str = "name.str.endswith('tiger')"
    print(df.query(expr_str, engine='python'))

#64. file concat 활용 연습
def web_file_concat():
    from glob import glob
    file_list = glob('C:/Users/eleun/Downloads/gas_station/*.xls')
    print(file_list)
    print(len(file_list))
    df1 = pd.read_excel(file_list[0], header=2)
    print(df1.head())
    data_list = []
    for i in range(len(file_list)):
        temp = pd.read_excel(file_list[i], header=2)
        data_list.append(temp)
    total = pd.concat(data_list, ignore_index=True)
    print(total)
    print(total.info())
    data = total[['상호', '주소', '휘발유', '셀프여부', '상표']]
    print(data.info())
    data.columns = ['store', 'address', 'gasoline', 'self', 'brand']
    print(data)
    gu_dummy_list = []
    for i in data.address:
        gu_dummy_list.append(i.split()[1])
    print(gu_dummy_list[3])
    df2 = data.copy()
    for i, j in enumerate(df2.address):
        df2.loc[i, 'gu'] = j.split()[1]
    print(df2.head())
    print(df2.gu.unique())
    df2.boxplot(column='gasoline', by='self', figsize=(12, 8))
    plt.show()
    plt.figure(figsize=(10,6))
    sns.boxplot(x='brand', y='gasoline', hue='self', data=df2)
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='brand', y='gasoline', data=df2)
    sns.swarmplot(x='brand', y='gasoline', data=df2, color='0.4')
    plt.show()
    df3 = df2.sort_values(by='gasoline', ascending=False).head(10)
    print(df3)
    df4 = df2.sort_values(by='gasoline').head(10)
    print(df4)
    print(df2.gasoline.groupby(df2.gu).mean())
    from geopy.geocoders import Nominatim
    geolocoder = Nominatim(user_agent = 'South Korea')
    def geocoding(address): 
        geo = geolocoder.geocode(address)
        crd = (geo.latitude, geo.longitude)
        return crd
    import re
    address_list = []
    for i in df4.address:
        address_list.append(re.sub(r'\([^)]*\)', '', i))
    geo_data = [geocoding(i) for i in address_list]
    print(geo_data)
    df4['lat'] = [i[0] for i in geo_data]
    df4['long'] = [i[1] for i in geo_data]
    print(df4)
    map = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='Stamen Toner')
    for i in df4.index:
        lats = df4.loc[i, 'lat']
        longs = df4.loc[i, 'long']
        if pd.notnull(df4.loc[i,'lat']):
            print(lats)
            folium.CircleMarker([lats, longs], radius=10, color='brown', fill_color='coral').add_to(map)
    map.save('./Cheap_GasStation.html')
    
    df5= df2.gasoline.groupby(df2.gu).mean()
    import json
    geo_path = 'C:/Users/eleun/Downloads/seoul_gu_geo.json'
    geo_datas = json.load(open(geo_path, encoding='utf-8'))
    map = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='Stamen Toner')
    map.choropleth(geo_data=geo_datas, data=df5, columns=[df5.index, 'gasoline'], key_on='feature.id', fill_color='PuRd')
    map.save('./Seoul_GasStation.html')

    address_list2 = []
    for i in df3.address:
        address_list2.append(re.sub(r'\([^)]*\)', '', i))
    geo_data2 = [geocoding(i) for i in address_list2]
    print(geo_data2)
    df3['lat'] = [i[0] for i in geo_data2]
    df3['long'] = [i[1] for i in geo_data2]
    print(df3)
    map = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='Stamen Toner')
    for i in df3.index:
        lats = df3.loc[i, 'lat']
        longs = df3.loc[i, 'long']
        if pd.notnull(df3.loc[i,'lat']):
            print(lats)
            folium.CircleMarker([lats, longs], radius=10, color='brown', fill_color='coral').add_to(map)
    map.save('./Expensive_GasStation.html')

#65. resample, rolling 활용 연습
def train_rolling():
    rng = pd.date_range('2022-01-01', periods=100, freq='D')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)
    print(ts)
    print(ts.resample('M').mean())
    print(ts.resample('M', kind='period').mean())
    rng = pd.date_range('2023-01-01', periods=1440, freq='T')
    ts = pd.Series(np.arange(1, len(rng)+1), index=rng)
    print(ts)
    print(ts.resample('3min').mean())
    print(sum(range(1, 61))/60)
    print(ts.resample('60min').mean())
    print(ts.resample('H').agg(['mean', 'size']))
    print(ts.resample('3H').agg(['mean', 'size']))
    print(ts.resample('D').agg(['mean', 'size']))
    titanic = sns.load_dataset('titanic')
    df = titanic.loc[:, ['age','fare']]
    print(df.head())
    print('\n')

    # 사용자 함수 정의
    def min_max(x):    # 최대값 - 최소값
        return x.max() - x.min()
        
    # 데이터프레임의 각 열을 인수로 전달하면 시리즈를 반환
    result = df.apply(min_max)   #기본값 axis=0 
    print(result)
    print('\n')
    print(type(result))
    print(ts.resample('D').mean())
    close_px = pd.read_csv('C:/Users/eleun/Downloads/stock_px_2.csv', parse_dates=True, index_col=0)
    print(close_px)
    df = close_px[['AAPL', 'MSFT']]
    df2 = df.resample('B').ffill()
    print(df2.head())
    print(df2.rolling(250).mean())
    df2.rolling(250).mean().plot()
    plt.show()
    df.AAPL.plot()
    df2.AAPL.rolling(250).mean().plot()
    plt.show()
    close_px.rolling(250).mean().plot()
    plt.show()
    df = df2.AAPL.rolling(10).mean()
    print(df.head(12))
    df = df2.AAPL.rolling(10, min_periods=4).mean()
    print(df.head(12))
    rng = pd.date_range('2023-01-01', periods=1440, freq='T')
    ts = pd.DataFrame({'number':np.arange(1, len(rng)+1), 'dt':rng})
    print(ts)
    df = ts.rolling(10).sum()
    print(df.head(12))
    df = ts.number.rolling(10, min_periods=4).sum()
    print(df.head(12))
    rng = pd.date_range('2023-01-01', periods=1440, freq='T')
    ts = pd.Series(np.arange(1, len(rng)+1), index=rng)
    print(ts)
    print(ts.rolling('4min').sum().head(10))
    print(ts.rolling('4min').mean().head(10))
    ts[::2] = np.nan
    print(ts.head(10))
    print(ts.rolling('4min').sum().head(10))
    print(ts.rolling(4).sum().head(10))

#66. scipy, numpy 기능 활용 연습
def train_numpys():
    from scipy.stats import norm
    x = 80
    m = 60
    sigma = 10
    z = (x-m)/sigma
    print(z)
    x = np.linspace(-5, 5, 101)
    print(x)
    y_pdf = norm(0, 1).pdf(x)
    f,axes = plt.subplots()
    axes.plot(x, y_pdf)
    axes.grid()
    axes.set_xlabel('Z', fontdict={'weight':'bold'})
    axes.set_ylabel('Probability Density Finction', fontdict={'weight':'bold'})
    axes.axvline(x=z, color='r')
    plt.show()
    print(norm(0,1).pdf(z))
    print(norm(0,1).cdf(z))
    print(1-norm(0,1).cdf(z))
    z_math = (90-80)/20
    print(z_math)
    print(1-norm(0,1).cdf(z_math))
    print(np.random.rand(5,2))
    e = np.random.rand(5,2)
    print(type(e))
    print(np.arange(6))
    print(np.arange(12).reshape(4,3))
    print(np.arange(24).reshape(2,3,4))
    a = np.array([1,2,3])
    print(a)
    a = [(1,2,3), ('a','b','c'), ("가","나","다")]
    b = np.array(a)
    print(b)
    print(b.ndim, b.shape)
    ar = np.array([(1,2,3), (4,5,6), (7,8,9)], dtype=float)
    print(ar)
    print(np.zeros((3,4)))
    print(np.ones((3,4)))
    print(np.full((3,4), 12))
    print(np.empty((3,4)))
    print(np.eye(4))
    print(np.zeros_like(a))
    print(np.ones_like(a))
    print(np.zeros((3,3,3,3)))

def train_numpys2():
    a = np.arange(0, 10, 5)
    b = np.linspace(0, 10, 5)
    print(a, b)
    print(np.logspace(0, 10, 5))
    print(np.logspace(1, 10, 10, base=2))
    print(np.random.rand(3,4))
    print(np.random.randint(1,10,(3,3)))
    print(np.random.randn(5,5))
    a = np.random.normal(15, 1, (2,4))
    print(a)
    np.random.seed(1000)
    print(np.random.rand(2,2))
    print(np.random.randint(2,5,(3,3,3)))
    a = np.linspace(1,9,9, dtype=int)
    a = a.reshape(3,3)
    b = np.arange(1, 10).reshape(3,3)
    print(a)
    print(b)
    print(a+b)
    print(np.add(a,b))
    print(a-b)
    print(np.subtract(a,b))
    print(a/b)
    print(np.divide(a,b))
    print(a*b)
    print(np.multiply(a,b))
    print(np.dot(a,b))
    print(np.exp(b))
    print(np.sqrt(a))
    print(np.sin(a))
    print(np.log(a))
    print(a==b)
    print(a>b)
    print(np.mean(a))
    a = np.ones((4,4), dtype=int)
    a[2,3] = 9
    a[1:3, 1:3] = 5
    print(a)
    arr = np.array([1,2,3,4])
    print(arr[1])
    arr2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    print(arr2[:,1])
    print(arr2[1,4])
    arr = np.arange(1, 11).reshape(2,5)
    print(arr)
    print(arr[1, -1])
    print(arr[-1, -1])
    print(arr.reshape(-1,2))
    print(arr2.reshape(-1,1))
    print(arr[1, 1:4])
    print(arr[:2, 2])

def train_numpys3():
    arr = np.arange(1, 11).reshape(2,5)
    print(arr.dtype)
    arr = np.array([1.1,2.1,3.1,0.0])
    new_arr = arr.astype(int)
    print(new_arr)
    print(new_arr.dtype)
    new_arr = arr.astype(bool)
    print(new_arr)
    arr = np.arange(1,13)
    print(arr.shape)
    print(arr)
    arr = np.arange(24).reshape(4,6)
    print(arr)
    print(arr.shape)
    print(arr.reshape(4,1,6))
    print(arr[:, np.newaxis])
    print(arr.reshape(24,1))
    print(arr.reshape(-1,1).shape)
    print(arr[np.newaxis,:])
    arr = np.arange(6).reshape(2,3)
    print(arr)
    for x in arr:
        print(x)
        for y in x:
            print(y)
    for x in np.nditer(arr):
        print(x)
    arr = np.arange(16).reshape(1,4,4)
    print(arr)
    for idx, x in np.ndenumerate(arr):
        print(idx, x)
    arr1 = np.array([1,2,3])
    arr2 = np.array([4,5,6])
    print(np.concatenate((arr1, arr2)))
    arr1 = np.array([[1,2],[3,4]])
    arr2 = np.array([[5,6],[7,8]])
    a_0 = np.concatenate((arr1, arr2))
    a_1 = np.concatenate((arr1, arr2), axis=1)
    print(a_0.shape)
    print(a_1.shape)
    arr = np.array([1,2,3,4,5,4])
    print(np.where(arr==4))
    print(np.sort(arr))
    mx = np.ma.array([1,2,3,4,5], mask=[True, False, False, False, False])
    print(mx)
    print(mx.data)
    print(mx.mask)
    print(mx>3)
    mx_l = mx[mx>3]
    print(mx_l)

def train_numpys4():
    a = np.array([1,2])
    b = np.array([3,4])
    print(np.dot(a,b))
    a = np.array([[1,2],[3,4]])
    b = np.array([[4,5],[6,7]])
    print(a)
    print(b)
    print(np.dot(a,b))
    c = 10
    print(np.dot(a, c))
    print(a)
    b = np.array([10, 100])
    print(b)
    print(np.dot(a, b))
    print(np.dot(b, a))
    print(b.shape)
    print(a.shape)
    a = np.arange(6).reshape(2,3)
    print(a)
    c = np.array([10, 100, 1000])
    print(c)
    print(np.dot(a,c))
    print(a.T)
    a = np.arange(3*4*5*6).reshape((3,4,5,6))
    b = np.arange(3*4*5*6).reshape((3,4,6,5))
    print(np.dot(a,b).shape)
    print(np.dot(a,b)[2,3,4,2,3,4])
    print(sum(a[2,3,4,:]*b[2,3,:,4]))
    a = np.arange(2*2*3).reshape((2,2,3))
    b = np.arange(2*2*3).reshape((2,3,2))
    print(a)
    print(b)
    print(np.dot(a, b))

def train_numpys5():
    arr = np.array([1,2,3,4,5,4,4])
    x = np.where(arr==4)
    print(x)
    print(type(x))
    arr = np.arange(6).reshape(2,3)
    print(arr)
    print(np.where(arr<3, 100, -100))
    print(np.where(arr<3, 100, arr))
    arr = np.array([3,2,0,1])
    print(np.sort(arr))
    temp = np.random.randint(1, 10, 10)
    print(temp)
    print(np.sort(temp))
    temp.sort()
    print(temp)
    temp = np.random.randint(1, 10, 10)
    reverse_order = np.sort(temp)[::-1]
    print(reverse_order)
    arr = np.array([41,42,43,44,45])
    x = [True, False, True, False, True]
    newarr = arr[x]
    print(newarr)
    mx = np.ma.array([1,2,3,4,5], mask=[True, False, False, False, False])
    print(mx)
    print(mx.mask)
    print(arr[mx.mask])

def train_numpys6():
    print(np.zeros((2,3)))
    print(np.ones((2,3)))
    print(np.empty((2,3)))
    arr = np.array([[1.,2.], [6.,7.]])
    print(arr.shape, arr.ndim)
    print(np.empty_like(arr))
    print(np.zeros_like(arr))
    print(np.ones_like(arr))
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print(a)
    print(b)
    print(a+b)
    print(a-b)
    print(a*100)
    print(a+100)
    print(a*b)
    print(a/b)
    c = np.array([10,30])
    print(a.shape, b.shape, c.shape)
    print(a+c)
    print(a*c)
    a3 = np.array([1,10,100])
    b3 = np.array([1])
    print(a3-b3)
    a3 = np.array([[1,10,100], [1000,1000,1000]])
    b3 = np.array([1,2,3])
    print(a3-b3)
    b4 = np.array([[1], [2]])
    print(b4)
    print(a3-b4)
    print(a3.shape)
    print(b4.shape)
    a = np.arange(3*2).reshape((2,3))
    print(a)
    a2 = a.T
    print(a2)
    print(np.sum(a2))
    print(np.sum(a2, axis=0))
    print(np.sum(a2, axis=1))
    print(np.mean(a2))
    print(np.mean(a2, axis=0))
    print(np.mean(a2, axis=1))
    print(np.max9(a2))
    print(np.max9(a2, axis=0))
    print(np.max9(a2, axis=1))
    a2 = np.array([[10,20,30], [40,50,60]])
    print(a2)
    print(np.argmin(a2))
    print(np.argmax(a2))
    print(np.argmin(a2, axis=0))
    print(np.exp(a3))
    print(np.sin(a3))
    print(np.arange(1,2,0.1))
    a1 = np.arange(6).reshape(3,2)
    print(a1)
    b1 = np.array([[10, 100, 1000]])
    print(np.dot(b1,a1))
    b = np.arange(6).reshape((2,3))
    print(b)
    print(np.dot(a1, b))
    print(np.dot(b, a1))
    a1 = np.array([1, 3, 5])
    b1 = np.array([4,2,1])
    print(np.dot(a1, b1))
    a = np.array([[1,3],[2,4]])
    print(a.shape)
    b=2
    print(np.dot(a,b))
    print(np.dot(b,a))
    x1 = 1
    x2 = 2
    w1 = 10
    w2 = 100
    a = np.array([x1, x2])
    b = np.array([w1, w2])
    print(np.dot(a,b))

def train_numpys7():
    def AND(x1, x2):
        w1, w2, b1 = 0.5, 0.5, 0.7
        temp = x1*w1 + x2*w2
        if temp <= b1:
            return 0
        elif temp > b1:
            return 1

    print(AND(0,0))
    print(AND(0,1))
    print(AND(1,0))
    print(AND(1,1))

    def AND(x1, x2):
        w1, w2= 0.5, 0.5
        b1 = -0.7
        temp = x1*w1 + x2*w2 +b1
        if temp <= 0:
            return 0
        elif temp > 0:
            return 1
    print(AND(0,0))
    print(AND(0,1))
    print(AND(1,0))
    print(AND(1,1))

    def AND(x1, x2):
        w1, w2= 0.5, 0.5
        b1 = -0.7
        z1 = x1*w1 + x2*w2 +b1
        if z1 <= 0:
            return 0
        elif z1 > 0:
            return 1
    print(AND(0,0))
    print(AND(0,1))
    print(AND(1,0))
    print(AND(1,1))

    def AND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b1 = -0.7
        z1 = np.dot(x, w) + b1
        if z1 <= 0:
            return 0
        elif z1 > 0:
            return 1
    print(AND(0,0))
    print(AND(0,1))
    print(AND(1,0))
    print(AND(1,1))

def train_numpys8():
    def OR(x1, x2):
        x = np.array([x1, x2])
        w = np.array([1, 1])
        b1 = -0.5
        z1 = np.dot(x, w) + b1
        if z1 <= 0:
            return 0
        else:
            return 1
    print(OR(0,0))
    print(OR(0,1))
    print(OR(1,0))
    print(OR(1,1))
    x1 = np.linspace(-1, 2, 20)
    print(x1)
    x2 = -x1 + 0.5
    print(x2)
    plt.figure(figsize=(8,8))
    plt.plot(x1,x2)
    plt.axvline(x=0, color='k')
    plt.axhline(y=0, color='k')
    plt.scatter([0],[0], marker='o', color='r')
    plt.scatter([1,0,1], [0,1,1], marker='^', color='r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.fill_between(x1, x2, -1.5, alpha=0.5)
    plt.grid()
    plt.show()
    def AND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([1,1])
        b1 = -1.2
        z1 = np.dot(x, w) + b1
        if z1 <= 0:
            return 0
        else:
            return 1
    print(AND(0,0))
    print(AND(0,1))
    print(AND(1,0))
    print(AND(1,1))
    def NAND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([-1, -1])
        b1 = 1.2
        z1 = np.dot(x, w) + b1
        if z1 <= 0:
            return 0
        else:
            return 1
    print(NAND(0,0))
    print(NAND(0,1))
    print(NAND(1,0))
    print(NAND(1,1))
    def XOR(x1, x2):
        s1 = NAND(x1, x2)
        s2 = OR(x1, x2)
        y = AND(s1, s2)
        return y
    print(XOR(0,0))
    print(XOR(0,1))
    print(XOR(1,0))
    print(XOR(1,1))

def train_numpys9():
    def step_func(x):
        h = x>0
        y = h.astype(int)
        return y
    x = np.array([0.1, -2.0, 1.0])
    h = x>0
    print(h)
    print(h.astype(int))
    print(step_func(x))
    def step_func(x):
        if x > 0:
            return 1
        else:
            return 0
    print(step_func(1))
    def step_func(x):
        '''
        x : numpy array, float
        h : numpy array, bool
        y : numpy array, int 0, 1
        '''
        h = x>0
        y = h.astype(int)
        return y
    x = np.arange(-5, 5, 0.1)
    print(step_func(x))
    plt.plot(x, step_func(x))
    plt.show()
    def relu(x):
        return np.maximum(0, x)
    print(relu(x))
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1.1, 5.1)
    plt.show()
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    t = np.array([-1.0, 1.0, 2.0])
    print(sigmoid(t))
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
    x = np.arange(-15.0, 15.0, 0.1)
    y1 = sigmoid(x)
    y2 = step_func(x)
    plt.plot(x, y1, label='sigmoid')
    plt.plot(x, y2, linestyle='--', label='step_function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

def train_numpys10():
    x = np.array([1, 2])
    W = np.array([[1,3,5], [2,4,6]])
    print(x)
    print(W)
    print(x.shape, W.shape)
    Y = np.dot(x, W)
    print(Y)
    a = np.array([[1,3], [2,4]])
    print(a)
    b = np.array([[[1,1],[0,0]], [[0,0], [1,1]], [[1,0], [0,1]]])
    print(b)
    print(a.shape, b.shape)
    print(np.dot(a,b))
