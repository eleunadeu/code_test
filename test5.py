# 데이터 교육 복습 5

#41. 스크랩핑 함수
from bs4 import BeatifulSoup
import re
import request

def scraping(url, soup, rows):
  '''
  url : 데이터를 가져올 사이트 주소 예) url = "https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds"
  soup : beatifulsoup 객체 생성 코드 예) soup = BeatifulSoup(resp.text, 'lxml')
  rows : css 선택자를 사용해 원하는 요소를 추출 예) rows = soup.select('div > ul > li')
  웹페이지에서 원하는 내용을 추출하기 위해 사용하는 코드 예시 코드로는 해당 사이트에서 li 태그들을 내용을 리스트로 저장할 수 있음
  웹페이지마다 구성이 다르기 때문에 해당 페이지에 맞춰서 코드 수정 필요함
  '''
  resp = request.get(url)
  print(type(soup))
  print(type(rows))
  file_name = 'C:/Users/eleun/.jupyter/test.html'
  page = open(file_name, 'r').read()
  print(page)

def scraping2(url, soup, rows):
  print(soup)
  print(soup.prettify())
  print(type(soup))
  children_list = list(soup.children)
  print(children_list)
  print(type(children_list[2]))
  scl_child = list(soup_children_list[2].children)
  print(scl_child)
  print(soup1.body)
  print(soup1.find_all('p'))
  print(soup1.head)
  print(soup1.head.next_sibling.next_sibling)

def find_test(soup, soup1, rows):
  '''
  find를 활용하는 함수, 웹페이지에 맞춰서 코드 및 re 정규식 수정 필요
  '''
  al_test = soup.find_all('p')
  for i in al_test:
    print(type(i.text))
  print(al_test[0].text)
  for row in rows:
    market = re.findall('\((.*)\|', row.text)
    ticker = re.findall('NYSE Arca\| (.*)\)', row.text)
    if market:
      print(market, ticker)
  find_list = list(soup1.body.children)
  print(len(find_list))
  for i in find_list:
    print(type(i))
  print(type(soup1.body.find_all('div')))
  print(dir(soup1.body.find_all('div')))
  print(dir(soup1.body.find_all('li')))

#42. 크롤링 함수
import pandas as pd
def crowling(url, soup_opt, row_opt):
  resp = requests.get(url)
  soup = BeatifulSoup(soup_opt)
  rows = soup.select(row_opt)
  li_list = soup.body.find_all('li')
  print(li_list)
  print(dir(li_list[35]))
  result_dict= {}
  for item in li_list:
      if 'NYSE' in item.text:
          dummy_list = item.text.split('(')
          name = dummy_list
          market_ticker = dummy_list[-1].split(')')[0].replace(':','').replace('|','')
          result_dict[market_ticker] = [name]
  final_dict = {}
  for i in result_dict:
      dummy_list = i.replace(u'\xa0', u' ').split(' ')
      if dummy_list[0] != 'NYSE' or dummy_list[1] != 'Arca':
          dummy_list[0]='NYSE'
      _list = [ j for j in dummy_list if j !='']
      final_dict[_list[-1]] = [' '.join(_list[:-1]), result_dict[i][0].strip(' ')]
  print(final_dict)
  df = pd.DataFrame(final_dict)
  print(df)

#43. 웹 크롤링 연습
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
def Food():
    url = 'https://www.chicagomag.com/chicago-magazine/january-2023/our-30-favorite-things-to-eat-right-now/'
    hdr = {'User-Agent':'Mozilla/5.0'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    tmp = soup.find_all('div', 'article-body')[0]
    food_list = []
    for item in tmp.find_all('h2'):
        food_list.append(item.text)
    restaurant_list = []
    for item in tmp.find_all('h3'):
        restaurant_list.append(item.text)
    money_list = []
    address_list = []
    for item in tmp.find_all('p'):
        sample_text = item.get_text()
        idx_of_dollar = sample_text.index('$')
        money = sample_text[idx_of_dollar:]
        dummy_address = sample_text[idx_of_dollar+len(money):]
        if dummy_address.split(' ')[0] == 'for':
            dummy_address = dummy_address[dummy_address.index('. ')+2:]
        money_list.append(money)
        address_list.append(dummy_address)
    data = {'food':food_list, 'restaurant':restaurant_list, 'price':money_list, 'address':address_list}
    df = pd.DataFrame(data)
    print(df)

def Food_Ranking():
    url = 'https://www.chicagomag.com/chicago-magazine/january-2023/our-30-favorite-things-to-eat-right-now/'
    hdr = {'User-Agent':'Mozilla/5.0'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    tmp = soup.find_all('div', 'article-body')[0]
    food_list = []
    for item in tmp.find_all('h2'):
        food_list.append(item.text)
    restaurant_list = []
    for item in tmp.find_all('h3'):
        restaurant_list.append(item.text[3:].replace(u'\xa0',u''))
    money_list = []
    address_list = []
    for item in tmp.find_all('p'):
        sample_text = item.get_text()
        idx_of_dollar = sample_text.index('$')
        money = sample_text[idx_of_dollar].split(' ')[0]
        dummy_address = sample_text[idx_of_dollar + len(money)]
        if dummy_address.split(' ')[0] == 'for':
            dummy_address = dummy_address[dummy_address.index('. ')+2:]
            money_list.append(money.strip('.'))
            address_list.append(dummy_address)
    data = {'food':food_list, 'restaurant':restaurant_list, 'price':money_list, 'address':address_list}
    df = pd.DataFrame(data)
    print(df)

def Re_Food():
    url = 'https://www.chicagomag.com/chicago-magazine/january-2023/our-30-favorite-things-to-eat-right-now/'
    hdr = {'User-Agent':'Mozilla/5.0'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    tmp = soup.find_all('div', 'article-body')[0]
    food_list = []
    for item in tmp.find_all('h2'):
        food_list.append(item.text)
    restaurant_list = []
    for item in tmp.find_all('h3'):
        restaurant_list.append(item.text[3:].replace(u'\xa0',u''))
    money_list = []
    address_list = []
    for item in tmp.find_all('p'):
        sample_text = item.get_text()
        idx_of_dollar = sample_text.index('$')
        dummy = sample_text[idx_of_dollar:]
        money = dummy.split(' ')[0].strip('.')
        dummy = dummy[re.search('\. \d+', dummy).start()+2:]
        money_list.append(money)
        address_list.append(dummy)

#44. 정규식 연습
import re
def re_match(opti):
  print(re.match(opti))

def re_search(opti):
  print(re.search(opti))

def re_findall(opti):
  print(re.findall(opti))

def re_compile(opti, opti2, opti3):
  comp = re.compile(opti)
  find = comp.match(opti2)
  search = comp.search(opti3)
  print(find)
  print(search)

def re_sub(comp, opti):
  '''
  comp : re.compile() 한 객체
  '''
  sub_result = comp.sub(opti)
  print(sub_result)

#45. 판다스 활용 연습
def data_train(url):
  df = pd.read_csv(url, encoding='euc-kr')
  print(df)
  print(df.head(3))
  print(df.tail(3))
  print(df.shape)
  print(df.info())
  print(df.select_dtypes(include=object).columns)
  print(df.describe())
  print(df.count())
  print(df.value_counts())
  print(df['일자'].value_counts())
  return df

#46. Folium 활용 맵 제작 및 저장
def seoul_map_edit(file):
    '''
    file : 엑셀로 된 지도관련 자료, 함수에 사용된 자료는 서울 대학관련 자료
    '''
    seoul_map = folium.Map(location=[37.55, 126.98], zoom_start=12)
    seoul_map.save('./seoul.html')
    df = pd.read_excel(file, engine='openpyxl')
    df.columns = ['name', 'lat', 'long']
    print(df.head())
    seoul_map = folium.Map(location=[37.55, 126.98], zoom_start=12,
                            titles='Stamen Terrain')
    for name, lat, lng in zip(df.name, df.lat, df.long):
        print(name, lat, lng)
    for name, lat, lng in zip(df.name, df.lat, df.long):
        folium.Marker([lat, lng], popup=name).add_to(seoul_map)
    seoul_map.save('./univer.html')

import json
def map_edit(file_path, geo_path):
  '''
  file_path : 사용할 액셀 자료 저장 파일 경로 예) file_path = 'C:/Users/eleun/downloads/g_pop.xlsx'
  geo_path : 사용할 위경도 자료 파일 경로 예) geo_path = 'C:/Users/eleun/downloads/g_district.json'
  '''
    df = pd.read_excel(file_path, index_col='구분', engine='openpyxl')
    df.columns = df.columns.map(str)
    print(df.head())
    try:
        geo_data = json.load(open(geo_path, encoding='utf-8'))
    except:
        geo_data = json.load(open(geo_path, encoding='utf-8-sig'))
    
    g_map = folium.Map(location=[37.5502, 126.982], tiles='Stamen Terrain', zoom_start=9)
    year = '2017'
    folium.Choropleth(geo_data=geo_data, data=df[year], columns=[df.index, df[year]],
    fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.3,
    threshold_scale=[10000, 100000, 300000, 500000, 700000],
    key_on='feature.properties.name',).add_to(g_map)
    g_map.save('g_pop.html')

#47. seaborn 및 pandas 활용 연습
import seaborn as sns
def sea_test():
    df = sns.load_dataset('titanic')
    print(df.head())
    print(df.info())
    print(df['deck'].value_counts(dropna=False))
    print(df.isnull().sum())
    print(df.columns)
    print(df.dropna(axis=1, thresh=500).columns)
    df_age = df.dropna(subset=['age'], how='any', axis=0)
    print(len(df_age))
    df = sns.load_dataset('titanic')
    print(df['age'].head(10))
    print('\n')
    mean_age = df['age'].mean(axis=0)
    df['age'].fillna(mean_age, inplace=True)
    print(df['age'].head(10))
    print(df['embark_town'][825:830])
    print('\n')
    most_freq = df['embark_town'].value_counts(dropna=True).idxmax()
    print(most_freq)
    print('\n')
    df['embark_town'].fillna(most_freq, inplace=True)
    print(df['embark_town'][825:830])
    print(df['embark_town'][825:830])
    print('\n')
    df['embark_town'].fillna(method='ffill', inplace=True)
    print(df['embark_town'][825:830])
    idx = ['row1', 'row2', 'row3']
    col = ['col1', 'col2', 'col3']
    data = [[1, 2, 300], [100, 5, 6], [7, 300, np.nan]]
    df = pd.DataFrame(data, idx, col)
    print(df)
    print(df.idxmax())

#48. matplotlib 사용 그래프 제작 연습
import matplotlib.pyplot as plt
def graph_train():
    plt.rcParams["legend.loc"] = "upper center"

    #자료 초기화
    y = pd.Series([1, 4, 2, 4, np.nan, np.nan, 7, 5, 4, np.nan, np.nan, 1, 2])
    x = np.arange(0, len(y))

    #차트 생성
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(25,3))
    ax[0].plot(x, y.fillna(0), color='red', label='updated')    
    ax[0].plot(x, y, label='original')
    ax[0].set_title('zero')

    ax[1].plot(x, y.fillna(y.mean()), color='red')
    ax[1].plot(x, y)
    ax[1].set_title('mean')

    ax[2].plot(x, y.ffill(), color='red')
    ax[2].plot(x, y)
    ax[2].set_title('ffill')

    ax[3].plot(x, y.interpolate(), color='red')
    ax[3].plot(x, y)
    ax[3].set_title('interpolate')

    fig.legend()
    plt.show()

#49. pandas 활용 연습
def data_test2():
    df = pd.DataFrame({'c1':['a', 'a', 'b', 'a', 'b'],
                        'c2':[1, 1, 1, 2, 2,],
                        'c3':[1, 1, 2, 2, 2,]})
    print(df)
    print('\n')
    df_dup = df.duplicated()
    print(df_dup)
    print('\n')
    col_dup = df['c2'].duplicated()
    print(col_dup)
    df2 = df.drop_duplicates()
    print(df2)
    df3 = df.drop_duplicates(subset=['c2', 'c3'])
    print(df3)

import pandas as pd
import numpy as np
def auto_mpg():
    df = pd.read_csv('C:/Users/eleun/downloads/auto-mpg.csv', header=None)
    df.columns = ['mpg','cylinder', 'displacement', 
                'horsepower','weight','acceleration', 
                'model year','origin', 'name']
    print(df.dtypes)
    print('\n')
    print(df['horsepower'].unique())
    print('\n')
    df['horsepower'].replace('?', np.nan, inplace=True)
    df.dropna(subset=['horsepower'],axis=0, inplace=True)
    df['horsepower'] = df['horsepower'].astype('float')
    print(df['horsepower'].dtypes)
    print(df['origin'].unique())
    df['origin'].replace({1:'USA', 2:'EU', 3:'JPN'}, inplace=True)
    print(df['origin'].unique())
    print(df['origin'].dtypes)
    df['origin'] = df['origin'].astype('category')
    print(df['origin'].dtypes)
    df['origin'] = df['origin'].astype('str')
    print(df['origin'].dtypes)
    print(df['model year'].sample(3))
    df['model year'] = df['model year'].astype('category')
    print(df['model year'].sample(3))
    df['horsepower'] = df['horsepower'].astype('float')
    count, bin_dividers = np.histogram(df['horsepower'], bins=3)
    print(bin_dividers)
    bin_name = ['저출력', '보통출력', '고출력'] 
    df['hp_bin'] = pd.cut(x=df['horsepower'], bins= bin_dividers, labels=bin_name,
                          include_lowest=True)
    print(df[['horsepower', 'hp_bin']].head(15))
    horsepower_dummies = pd.get_dummies(df['hp_bin'])
    print(horsepower_dummies.head(15))
    print(df.horsepower.describe())
    print('\n')
    df.horsepower = df.horsepower/abs(df.horsepower.max())
    print(df.horsepower.head())
    print('\n')
    print(df.horsepower.describe())
    print('\n')
    print(df.horsepower.describe())
    print('\n')
    min_x = df.horsepower - df.horsepower.min()
    min_max = df.horsepower.max() - df.horsepower.min()
    df.horsepower = min_x/min_max
    print(df.horsepower.head())
    print('\n')
    print(df.horsepower.describe())
