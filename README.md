# code_test

# 함수화 모듈 복습 1주차(데이터 교육 과정 복습)

#Module page
#1 튜플 인덱싱, 카운트 연습
def tuple_method_exer():
    a = (1, 2, 3, 4, 2, 3, 3,)
    *b, c = a
    print('index 한 값: ', a.index(2))
    print('count 한 값: ', a.count(3))
    print('b 값: ', b)
    print('c 값: ', c)

#2 구구단 함수1
def gugudan(n):
    '''
    구구단 구현 함수, n: 자연수'''
    for i in range(1, 10):
        print(f'{n} * {i} = {n*i}')

#3 구구단 함수2
def in_out_gugudan(n, last=None):
    '''
    n: 자연수
    last: 자연수, 기본값은 None으로 함수 호출 시 없어도 되는 값

    예제: in_out_gugudan(2, 4)
    2단부터 4단까지 구구단 출력

    예제 : in_out_gugudan(7)
        7단만 출력
    '''
    if last:
        for i in range(n, last+1):
            gugudan(i)
            print()

    else:
        gugudan(n)

#4 약수 출력 함수
def divisor(n):
    '''
    n: 자연수
    예제: divisor(6)
    6의 약수를 출력
    '''
    data = []
    for i in range(1, n+1):
        if n % 1 == 0:
            data.append(i)
    print(data)

#5 공약수 출력 함수
def common_divisor(n , n2):
    data = []
    for i in range(1, n + 1):
        if (n % i == 0) & (n2 % i == 0):
            data.append(i)
    print(data)

#6 딕셔너리 연습 함수
def personal_dict(per_dict):
    '''
    per_dict 작성 시 key 형식은 아래와 동일하게 입력
    name:이름 , phone:전화번호, job:일, address:주소
    '''
    print(per_dict['name'])
    print(per_dict['phone'])
    print(per_dict['job'])
    print(per_dict['address'])

#7 딕셔너리 연습 함수2
def sample_dict(sample_dict):
     sample_dict[50] = {'peach':('korea', 3000, 4000)}
     for i in sample_dict.keys():
        print(sample_dict[i])
        print(sample_dict[10][0])
        print(sample_dict[20][0][3:6])
        print(sample_dict[30][2::])
        print(sample_dict[50]['peach'][0])
        print(sample_dict[20][1])
        print(sample_dict[20][2])
        print(sample_dict[60])
        print(sample_dict.items())
        print(sample_dict.values())

#8 리스트 합계 출력 함수
def sums_list(n):
  '''
  n = 자연수
  '''
    item = []
    item.extend(list(range(1,n)))
    sums = 0
    for i in items:
        sums += i
    return sums

#9 리스트 곱셈 출력 함수
def multiply_list(n):
    '''
    n = 리스트
    '''
    tot = 1
    for i in n:
        tot *= i
    return tot
