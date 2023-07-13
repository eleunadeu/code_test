# 데이터 교육 복습 3

#21
def add_dict(data, key_list, value_list):
  '''
  딕셔너리에 새로운 key와 value를 지정해 추가하는 함수
  data: 딕셔너리, key_list: 추가할 key 값이 있는 리스트,
  value_list: 추가할 value가 있는 리스트
  '''
  for i, j in eumerate(key_list):
    data[j] = value_list[i]
  return data

#22
def cannon_add_dict(data, key, value):
  '''입력 받은 key와 value 값이 리스트일 경우 각 항목을 key와 value로 딕셔너리
    (parameter: data)에 추가하고 아닐 경우 key와 value로 추가하는 함수
    data: 딕셔너리 key, value: 넣고싶은 값(자연수, 튜플, 리스트)
    key가 list일 때 value가 list가 아닐 경우 오류가 남.
    '''
  if isinstance(key, list) and isinstance(value, list):
    add_dict(data, key, value)
  else:
    data[key] = value
  return data

#23 DNA test 함수
def dna_test(dna_data):
  if (len(dna_data)% 3) != 0:
    return False
  if not dna_data.startswith('ATG'):
    return False
  for i in range(len(dna_data)-3):
    if i % 3 == 0:
      if dna_data[i:i+3] == 'TAA':
        return False
      if dna_data[i:i+3] == 'TAG':
        return False
      if dna_data[i:i+3] == 'TGA':
        return False
  if dna_data.endswith('TAA'):
    return True
  if dna_data.endswith('TAG'):
    return True
  if dna_data.endswith('TGA'):
    return True
  return False

#24 Global, 함수 영역 연습
def a():
  print('here')
  print('enclosed')
  def b():
    def range(n):
      return('test', 'python')
      print('local')
      sum([1,3])
      for i in range(10):
        print('여기는 반복문 안 입니다.', i)
  b()
  return 'ok'

#25 lambda 함수
f = lambda x, y : x+y
incre = lambda x, i_def = 1: x+i_def
def f1(x):
  return x*x + 3*x - 10
def f2(x):
  return x*x*x
def g(func):
  return [func(x) for x in range(-10, 10)]

func_list = [lambda x, y: x + y , lambda x, y: x - y , lambda x, y: x * y , lambda x, y : x / y]
'''
람다 함수를 사용해서 사칙연산을 수행하는 함수
x, y : 자연수 입력
람다 함수가 들어간 리스트를 인덱싱 해서 계산하는 것도 가능
'''
def test_func(x, y):
  for i, j in enumerate(func_list):
    print(i+1, j(x, y))
# func_list 버전 2
plus = lambda x, y: x + y
minus = lambda x, y: x - y
multiply = lambda x, y: x * y
divide = lambda x, y: x / y
list_oper = [plus, minus, multiply, divide]

def test_func2(n1, n2, n3, n4):
  for i, j, k in [(n1, n2, plus), (n3, n4, minus)]:
    print(k(i, j))

#26
def menu():
  print("0. add")
  print("1. sub")
  print("2. mul")
  print("3. div")
  print("4. quit")
  return int(input('Select menu: '))

def menu_test():
  '''
    사칙 연산을 선택해서 계산하는 함수
    위의 menu함수와 연동해서 사용 됨
    ''' 
  while 1:
    sel = menu()
    if sel < 0 or sel > len(func_list):
      continue
    if sel == len(func_list):
      break
    x = int(input('First operand:'))
    y = int(input('Second operand:'))
    print('Result = ', func_list[sel](x, y), end="\n\n")

#27
def single_double(data):
  '''
  필터 함수를 활용해서 입력된 값의 홀수와 짝수를 리스트로 변환하는 함수
  data : 자연수 입력
  '''
  result_1 = list(filter(lambda x: x%2, range(data)))
  result_2 = list(filter(lambda x: x%2 == 0, range(data)))
  print('홀수: ', result_1)
  print('짝수: ', result_2)

#28 Time Library 사용 함수
import time
datas = list(range(100000000))
def extend_edit(datas):
  '''
  입력된 리스트를 더미 리스트에 넣고
  extend를 사용해 선택한 항목(인덱스 넘버의 내용)을 변경하는 함수
  '''
  dummy_list = []
  dummy_list.extend(datas)
  dummy_list[1] = 14
  return dummy_list

def copy_edit(datas):
  '''
  copy를 사용해 입력된 리스트를 더미 리스트에 넣고
  선택한 항목의 내용을 변경하는 함수
  '''
  dummy_list = datas.copy()
  dummy_list[1] = 14
  return dummy_list

def func_exe_time(func, datas):
  '''
  함수의 실행시간 측정하는 함수
  func에 들어가는 리스트의 결과 값은 출력을 지정하지 않았기 때문에
  출력되지 않고 해당 함수를 실행하는데 걸린 시간만 출력됨.
  '''
  before = time.time()
  func(datas)
  return (time.time() - before)

#29 Pandas 사용 함수
import pandas as pd
import requests

def box_office(url):
  '''
  url : 인터넷 주소
  url에 있는 정보를 데이터 리스트에 저장한 후 pandas를 활용하여
  데이터표로 만드는 함수
  '''
  data = requests.get(url).json()
  datas = pd.DataFrame(data['boxOfficeResult']['dailyBoxOfficeList'])
  print(datas)

#30 위도, 경도 정리 함수
def map_test(data):
  '''
  data: 지명, 위도, 경도 정보가 있는 리스트
  리스트 내의 지명, 위도, 경도를 보여주는 함수
  '''
  for i in data:
    name = i
    lat_ = data[i][0]
    long_ = data[i][1]
    print(name, lat_, long_)
