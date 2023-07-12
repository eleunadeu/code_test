# 데이터 교육 과정 함수화 코드 2

#10. twosumtwofor 함수
def TwoSumTwofor(numb, target):
  '''
  리스트 안의 임의의 두 숫자의 합이 타겟 값과 일치하는 경우 반환하는 함수
  numb : 숫자로 된 리스트
  target : 타겟 값
  '''
  for i in range(len(numb)):
    for j in range(i+1, len(numb)):
      if numb[i]+numb[j] == target:
        return(i, j)

#11. TwoSumMinus 함수
def TwoSumMinus(numb, target):
  '''
  리스트 안의 두 숫자를 선택 해 뺀 값이 타겟 값과 일치할 때
  두 수의 인덱스를 반환하는 함수
  numb : 숫자로 된 리스트
  target : 타겟 값
  '''
  for inum, item in enumerate(numb):
    diff = target - item
    if diff in numb[inum+1:]:
      return(numb.index(item), numb.index(diff))

#12. TwoSumDict 함수
def TwoSumDict(numb, target):
  '''
  리스트 안의 두 숫자의 합이 목표값과 일치하는 경우 반환하는 함수
  딕셔너리를 사용하므로 더 빠르게 찾을 수 있음
  numb : 숫자 리스트, target : 타겟 값
  '''
  numb_dict = {}
  for inum, item in enumerate(numb):
    numb_dict[item] = inum
    diff = target - item
    if diff in numb_dict:
      return (inum, numb_dict[diff])

#13. 컴퓨터 기본 알고리즘 구현 함수
def brute_force_algorithm(total_head, total_leg):
  '''
  동물의 총 머릿수와 총 다리 개수를 입력하면 어떤 동물이 몇 마리인지
  계산하는 함수
  total_head : 동물의 총 숫자, total_leg : 두 동물의 다리 합계 
  i : 다리 2개인 동물 수, j: 다리 4개인 동물 수
  '''
  for i in range(total_head +1):
    j = total_head - i
    condition = 2*i + 4*j
    if condition == total_leg:
      return (i, j)
  return(None, None)

#14. 13번 함수 응용 함수 1
def banyard():
  total_head = int(input('Enter number of haeds: '))
  total_legs = int(input('Enter number of legs: '))
  (chickens, pigs) = brute_force_algorithm(total_head, total_leg)
  if chickens:
    print('number of chickens:', chickens)
    print('number of pigs:', pigs)
  else:
    print('There is no solution')

#15. 13번 함수 응용 함수 2
def bfa_upgrade(total_heads, total_legs):
  '''
  i: 다리 2개인 동물 수, j: 다리 4개인 동물 수, k: 다리 8개인 동물 수
  '''
  result = []
  for k in range(total_heads+1):
    for i in range(total_heads-k+1):
      j = total_heads - k - i
      sum_legs = 2*i + 4*j + 8*k
      if sum_legs == total_legs:
        result.append((i,j,k))
  return(result)

#16. 15번 함수 응용 함수
def banyard_spider():
  total_head = int(input('Enter number of heads: '))
  total_leg = int(input('Enter number of legs: '))
  result = bfa_upgrade(total_head, total_leg)
  if result:
    print(result)
    for i in result:
      print('============================')
      print('Number of chickens: ', i[0])
      print('Number of pigs: ', i[1])
      print('Number of spiders: ', i[2])
  else:
    print('There is no solution')

import glob
#17. glob 사용 함수 1
def FileCollection(source_paths, file_types):
    '''지정된 경로 내에 있는 지정한 형식의 모든 파일을 가져오는 함수
source_paths: 파일 경로를 지정하는 파라미터
예: 판다라는 폴더를 지정하는 방법
source_paths = '\\ProgramData\\Anaconda3\\lib\\site-packages\\pandas\\
file_types: 파일의 형식을 지정하는 파라미터
예: 'py', 'csv', 'txt' 등
file_list: 결과 값을 저장하는 리스트
'''
    dummy_arg = source_paths+'**/*.'+file_types
    print(dummy_arg)
    file_list = glob.glob(dummy_arg, recursive=True)
    return file_list

#18. glob 사용 함수 2
def GetFileSentence(file_lsit, search_sentence):
    '''위의 FileCollection 함수와 연동해서 사용하는 함수
    위의 함수로 불러들인 파일에서 원하는 값을 찾아 list형식으로 value에 
    저장하고 그 값이 포함된 파일명을 key값으로 저장하는 딕셔너리를 만든다.
    file_list: 위의 함수의 결과로 생긴 list 이름을 입력
    search_sentence: 찾고자 하는 값(예: 'index'라는 단어)을 입력
    '''
    file_dict = {}
    for filename in file_lsit:
        sentence_list = []
        with open(filename) as f:
            lines= f.readlines()
        for item in lines:
            if search_sentence in item:
                sentence_list.append(item)
        if sentence_list:
            file_dict[filename] = sentence_list
    return(file_dict)

#19. A와 B값 변경 함수
def swap(a, b):
  return b, a

#20. 딕셔너리 활용 함수
def create_dict(arg):
    '''각 수의 제곱을 딕셔너리로 만드는 함수
    arg: 자연수 입력
    예제: 10을 입력할 경우 1부터 10까지 제곱한 값을 1:1, 2:4...식으로 저장
    '''
    create_dict= {}
    for i in range(1, arg+1):
        create_dict[i] = i**2
    return create_dict
