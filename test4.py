# 데이터 교육 복습 4

#31.folium 사용 함수
import folium
map = folium.Map(location=[37.5602, 126.982]. zoom_start=7, tiles='cartodbpositron')
def map_tour(data):
  '''
    서드 파티 모듈 folium을 사용해서 좌표 값을 입력한 위치를 지도에 표시하고
    지도를 저장하는 함수
    사용 방법: 1. data 값에 key값(장소 이름)과 value값(해당 위치의 좌표(위,경도))
    를 입력한 리스트를 저장한다.
    2. 함수의 folium 설정을 원하는 대로 수정한다(반경, 팝업여부, 색깔, 저장이름 등)
    3. 1의 데이터 값을 넣고 함수를 돌린다.
    4. ls를 이용해 파일이 저장되었는지 확인 후 해당 위치로 가서 저장된 
    maps 파일을 확인한다.
    '''
  for i in data:
    name = i
    lat_ = data[i][0]
    long_ = data[i][1]
    folium.CircleMarker([lat_, long_], radius=4, popup=name, color='red', fill_color='red').add_to(map)
    map.save('tour.html')
    print('ok')

#32.global 연습 함수
result= 0
def add(num):
    '''
    계산기를 함수로 만든 것, 결과 값이 저장된 채로 다음 값을 더하기 위해
    글로벌을 사용해 함수 바깥의 결과 값을 함수로 가져와서 사용.
    '''
  global result
  result += num
  return result

#33. Class 연습 함수
class Calculator:
  '''
  계산기 함수를 클래스를 사용하여 다시 만들어 낸 것.
  사용 방법 인스턴스 오브젝트(예: x) = 클래스(이름)()*(파라미터 가 있을 경우 입력 필요, 없으면 빈 괄호)
  인스턴스 오브젝트 variable(예x.add(2))로 사용
  '''
  def __init__(self):
    self.result = 0
  def add(self, num):
    self.result += num
    return self.result

class S1:
  def __init__(self):
    self.a = 1

class S2:
  def __init__(self, a):
    self.a = a

class S3:
  a = 1

class MyClass:
  a = 1
  def __init__(self, name, age):
    self.name = name
    self.age = age

class MyClass2:
  def __init__(self):
    self.age = 21
    self.name = 'Guido'

class coffee:
  def print_info(self):
    print('상품명과 가격을 출력합니다')

class coffee1:
  '''
    입력되어 있는 값을 출력하는 함수가 포함된 클래스
    '''
  def __init__(self):
    self.product = '카누'
    self.price = '3000'
  def print_info(self):
    print('상품명과 가격을 출력합니다.', self.product, self.price)

class coffee2:
  '''
  입력한 파라미터 값을 출력해주는 함수가 있는 클래스
  '''
  def __init__(self, product, price):
    self.product = product
    self.price = price
  def print_info(self, numbers):
    self.numbers = numbers
    print('상품명과 가격을 출력합니다.', self.product, self.price)
    print('개수 :', self.numbers)

#34. Class 연습 2
class A:
  def f(self):
    print('base')
class B:
  pass
class C(list):
  pass

class MyClass3:
  def set(self, v):
    self.value = v
  def get(self):
    return self.value
  def temp(self):
    self.value2 = self.get()

class MyClass4:
  def __init__(self, a):
    self.value = 100
  def set(self, v):
    self.value = v
  def get(self):
    return self.value

class FourCal:
  '''
  입력된 값을 사칙연산 해주는 함수 4개와 값을 변경하는 함수가 포함된 클래스
  '''
  def __init__(self, first, second):
    self.first = first
    self.second = second
  def add(self):
    return self.first + self.second
  def sub(self):
    return self.first - self.second
  def mul(self):
    return self.first * self.second
  def div(self):
    return self.first / self.second
  def set(self, a, b):
    self.first = a
    self.second = b

class PowerFourCal(FourCal):
  '''
  입력된 값 중 첫번째 값의 제곱을 구해주는 함수가 있는 클래스
  '''
  def power(self):
    return self.first**2

class SafePowerFourCal(FourCal):
  '''
  입력된 값 중 두번째 값이 0일 때 0을 출력하고 아닐 경우 두 값을 나누는 함수가 포함된 클래스
  '''
  def div(self):
    if self.second == 0:
      return 0
    return self.first/self.second

#35.Class 연습 3
class Country:
  """Super Class"""
  name = '국가명'
  population = '인구'
  captial = '수도'
  def show(self):
    print('국가 클래스의 메서드 입니다.')

class Korea(Country):
  """Sub Class"""
  def __init__(self, name):
    self.name = name
  def show_name(self):
    print('국가 이름은 :', self.name)

class Base:
  def f(self):
    self.g()
  def g(self):
    print('Base')

class Derived(Base):
  def g(self):
    print('Derived')

class A1:
  def __init__(self, name):#constuctor
    self.name
    self.name = '춘향이'
  def nf(self):
    print(self.name)

class B1(A1):
  def __init__(self, name):
    self.name = name
    print('이상한 클래스')
    super().__init__(self.name) #super를 사용할 경우 위의 춘향이가 출력 됨

class A2(object):
  def saves(self):
    print('A saved')

class C(A2):
  def saves(self):
    print('C saved')
    super().saves()

class MyClass5:
  def __add__(self, x): #__add__는 '+' 연산을 지원하는 메서드
    print('add {} called'.format(x))

#36. class 연습 4
class Animal:
  def cry(self):
    print('...')
class Dog(Animal):
  def cry(self):
    print('멍멍')
class Duck(Animal):
  def cry(self):
    print('꽥꽥')
class Fish(Animal):
  pass

class tests:
  def __init__(self, int_a):
    self.int_a = int_a

class A3:
  def __f(self):
    print('__f called')

import time

class Time:
  '''
  함수 식이 작동되는 시간을 측정하는 함수를 클래스로 바꾼 것
  '''
  def __init__(self, function, datas):
    self.function = function
    self.datas = datas
  def exec_time(self):
    before = time.time()
    self.function(self.datas)
    return (time.time() - before)

class StopWatch:
  '''
  위 클래스와 다른 방식으로 작동되는 클래스
  '''
  def __init__(self):
    self._creationTime - time.time()
  def elapsedTime(self):
    return time.time() - self._creationTime
  def intervalTime(self, func, datas):
    func(datas)
    return self.elapsedTime()

class Book:
  def set_info(self, title, writer):
    self.title = title
    self.writer = writer
  def print_info(self):
    print(f'책 제목: {self.title}')
    print(f'책 저자 : {self.writer}')

#37. 데이터 프레임 연습
def exam_data():
    exam_data = {'이름':['서준', '우현', '인아'], 
                '수학':[90,80,70], '영어':[98, 89, 95], 
                '음악':[85, 95, 100], '체육':[100, 90, 90]}
    df = pd.DataFrame(exam_data)
    df.set_index('이름', inplace=True)
    print(df)
    print('\n')
    print(df.iloc[0][3])
    print(df.loc['서준']['체육'])
    print(df.loc['서준', '체육'])

def dict_data():
    dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 
                'c2':[7,8,9], 'c3':[10,11,12], 
                'c4':[13,14,15]}
    df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
    print(df)
    print('\n')
    
    new_index = ['r0', 'r1', 'r2', 'r3', 'r4']
    ndf = df.reindex(new_index)
    print(ndf)
    print('\n')
    new_index = ['r0', 'r1', 'r2', 'r3', 'r4']
    ndf = df.reindex(new_index, fill_value=0)
    print(ndf)
    print('\n')
    ndf2 = df.reset_index(drop=True)
    print(ndf2)
    df.sort_index(ascending=False)
    ndf3 = df.sort_values(by='c1', ascending=False)
    print(ndf3)
