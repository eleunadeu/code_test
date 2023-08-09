#데이터 교육 복습 22

#81. 딥러닝 연습 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dml_train_2():
    from mnist import load_mnist
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)
    print(x_train[0].shape)
    print(t_train[0])
    img = x_train[0].reshape(28, 28)
    print(img.shape)
    plt.imshow(img)
    plt.imshow(x_train[10].reshape(28,28))
    print(t_train[10])

    import pickle

    def get_data():
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        return x_test, t_test
    
    def init_network():
        with open('C:/Users/eleun/Downloads/sample_weight.pkl','rb') as f:
            network = pickle.load(f)
        return network
    
    def predict(network, X):
        W1, W2, W3 = network['W1'],network['W2'],network['W3']
        B1, B2, B3 = network['b1'],network['b2'],network['b3']

        A1 = np.dot(X, W1) + B1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W2) + B2
        Z2 = sigmoid(A2)
        A3 = np.dot(Z2, W3) + B3
        Y = softmax(A3) #확률
        return Y
    
    x, t = get_data()
    print(x.shape, t.shape)
    network = init_network()
    print(network['W1'])
    print(network.keys())
    print(t[:3])

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스 산출
        if p == t[i]:
            accuracy_cnt += 1
    print(accuracy_cnt/len(x))

    print(y.shape)
    print(np.argmax(y))

    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    B1, B2, B3 = network['b1'],network['b2'],network['b3']
    print(x.shape, W1.shape, W2.shape, W3.shape)
    print(i, y.shape)

    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size] # x[0:100], x[100:200], ...
        #print(i, i+batch_size)
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:t+batch_size])
    print('Accuracy : ', accuracy_cnt/len(x))

    print(x_batch.shape, x.shape, p.shape, accuracy_cnt)

    def mean_squared_error(y, t):
        return 0.5 * np.sum((y-t)**2)
    
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    print(mean_squared_error(y, t))
    t = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    print(mean_squared_error(y, t))
    print(-np.log(0,1), -np.log(0.6))
    x = np.arange(0.001, 1.0, 0.001)
    y = np.log(x)
    plt.plot(x, y)
    plt.show()
    x = np.arange(0.001, 1.0, 0.001)
    y = -np.log(x)
    plt.plot(x, y)
    plt.show()

    def cross_entropy_error(y, t):
        delta = 1e-7
        return -np.sum(t*np.log(y+delta))
    
    print(cross_entropy_error(y, t))
    train_s = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_s, batch_size)
    print(batch_mask)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(x_batch.shape, t_batch.shape)
    print(t.size, x_batch.size)

    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t]))/ batch_size
    
    y = np.array([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0], [10.1, 10.05, 10.1, 10.0, 10.05, 10.1, 10.0, 10.6, 10.0, 10.0]])
    t = np.array([1, 2])

    batch_size = y.shape[0]
    print(y.shape, y.shape[0], batch_size)
    print(y[np.arange(batch_size), t])
    print(y [[0, 1], [1, 2]])

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx] # 10
        x[idx] = tmp_val + h # 10 + 0.0001
        fxh1 = f(x)

        x[idx] = tmp_val - h # 10 - 0.0001
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x**2)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr*grad # x = x - lr*grad
    return x, np.array(x_history)

def softmax(a):
    if a.ndim > 1:
        y = []
        for i in a:
            c = np.max(i)
            exp_a = np.exp(i-c) # 오버 플로 대책
            sum_exp_a = np.sum(exp_a)
            y.append(exp_a / sum_exp_a)
    else:
        c = np.max(a)
        exp_a = np.exp(a-c) #오버 플로 대책
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) #정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
    
def dml_train_3():
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    x = np.arange(20).reshape(4, 5)
    print(x.shape, x.size)
    print(x.reshape(-1, 1))
    print(x = x.reshape(2, 2, 5))
    it = np.nditer(x, flags=['multi_index'], op_flags = ['readwrite'])
    while not it.finished:
        idx = it.multi_index
        print(idx, x[idx])
        it.iternext()

    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
    itit_x = np.array([-3.0, 4.0])
    lr = 0.1
    step_num= 20
    x, x_history = gradient_descent(function_2, init_x, lr, step_num)
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.show()

    print(x_history)

    #학습률이 클 때 : lr = 10
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x=init_x, lr=10, step_num=100))
    #학습률이 작을 때 : lr = 1e-10
    print(gradient_descent(function_2, init_x=init_x, lr = 1e-10, step_num=100))

    def softmax_old(a):
        c = np.max(a)
        exp_a = np.exp(a-c) #오버플로 대책
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    x = np.array([[1, 2, 3], [4, 9, 7]])
    print(softmax_old(x))
    x = np.array([1, 2, 3])
    print(softmax(x))
    x = np.array([[1, 2, 3], [4, 9, 7]])
    print(softmax(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()

    return grad

def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def softmax(a):
    if a.ndim == 1:
        y = soft_max(a)
        return y
    y = []
    for i in a:
        y.append(soft_max(i).tolist())
    return np.array(y)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std= 0.01):
        self.params ={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        #forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        #backyard
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
