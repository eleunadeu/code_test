# 데이터 교육 복습 24

#83. 딥러닝 연습 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
        for key, val in params.items():
            self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-7)


class Adam:

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
class MultiLayerNet:
    """완전연결 다층 신경망
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기
    """

    def __init__(self, input_size, hidden_size_list, output_size, 
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        #가중치 초기화
        self.__init__weight(weight_init_std)

        #계층 생성
        activation_layer = {'sigmoid':Sigmoid, 'relu':Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], 
                                                    self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str[idx]])
        
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """가중치 초기화
        
        Parameters
        ----------
        weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
            'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
            'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size] # [784, 100, 100, 100, 10]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1]) #ReLu를 사용할 때 권장 초기값
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1]) #sigmoid를 사용할 때의 권장 초기값
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list)

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        """손실 함수를 구한다.
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블 
        
        Returns
        -------
        손실 함수의 값
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' +str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1) #t가 1차원이 아니므로, one-hot 인코딩으로 되어 있는 경우

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        """기울기를 구한다(수치 미분).
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        Returns
        -------
        각 층의 기울기를 담은 딕셔너리(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        Returns
        -------
        각 층의 기울기를 담은 딕셔너리(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads


def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def dml_train_6():
    from mpl_toolkits.mplot3d import Axes3D
    X = np.arange(-10, 10, 0.5)
    Y = np.arange(-10, 10, 0.5)
    XX, YY = np.meshgrid(X, Y)
    ZZ = (1 / 20) * XX**2 + YY**2

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='hot')
    plt.show()

    plt.contour(XX, YY, ZZ, 100, colors='k')
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.show()

    def f(x, y):
        return x**2 / 20.0 + y**2

    def df(x, y):
        return x / 10.0, 2.0*y

    init_pos = (-7.0, 2.0)
    params = {}
    params['x'], params['y'] = init_pos[0], init_pos[1]
    grads = {}
    grads['x'], grads['y'] = 0, 0


    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=0.95)
    optimizers["Momentum"] = Momentum(lr=0.1)
    optimizers["AdaGrad"] = AdaGrad(lr=1.5)
    optimizers["Adam"] = Adam(lr=0.3)

    idx = 1

    for key in optimizers:
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]
        
        for i in range(30):
            x_history.append(params['x'])
            y_history.append(params['y'])
            
            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)
        

        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)
        
        X, Y = np.meshgrid(x, y) 
        Z = f(X, Y)
        
        # 외곽선 단순화
        mask = Z > 7
        Z[mask] = 0
        
        # 그래프 그리기
        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color="red")
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        #colorbar()
        #spring()
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")
        
    plt.show()

    from mnist import load_mnist

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    optimizers = {}
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                    output_size=10)
        train_loss[key] = []

    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            print('=================='+'iteration : '+str(i)+'===========')
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ' :: '+str(loss))

    markers = {'SGD':'o', 'Momentum':'x', 'AdaGrad':'s', 'Adam':'D'}
    x = np.arange(max_iterations)

    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    def sigmoid(x):
        return 1 / (1+np.exp(-x))
    def ReLu(x):
        return np.maximum(0, x)
    def tanh(x):
        return np.tanh(x)

    input_data = np.random.randn(1000, 100)

    W = np.random.randn(100, 100)

    hidden_layer_size = 5
    activations = {}
    node_num = 100

    def get_activation(hidden_layer_size, x, w, a_func=sigmoid):
        for i in range(hidden_layer_size):
            if i != 0:
                x = activations[i -1]
            
            a = np.dot(x, w)

            z = a_func(a)

            activations[i] = z
        return activations

    w = np.random.randn(node_num, node_num) * 1 #정규분포 초기값 사용

    z = sigmoid
    x = input_data
    activations = get_activation(hidden_layer_size, x, w, z)

    def get_histogram(activations):
        for i, a in activations.items():
            plt.subplot(1, len(activations), i+1)
            plt.title(str(i+1) + "-layer")
            if i != 0: plt.yticks([], [])
            # plt.xlim(0.1, 1)
            # plt.ylim(0, 7000)
            plt.hist(a.flatten(), 30, range=(0,1))
        plt.show()

    get_histogram(activations)

    w = np.random.randn(node_num, node_num) * 0.01
    activations = get_activation(hidden_layer_size, x, w, z)
    get_histogram(activations)

    w = np.random.randn(node_num, node_num) * np.sqrt(1.0/ node_num)
    activations = get_activation(hidden_layer_size, x, w, z)
    get_histogram(activations)

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    weight_init_types = {'std=0.01':0.01, 'Xavier':'sigmoid', 'He':'relu'}
    optimizer = SGD(lr=0.01)

    networks = {}
    train_loss = {}
    for key in weight_init_types.keys():
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                    output_size=10)
        train_loss[key] = []

    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in weight_init_types.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            weight_init_types[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            print('=================='+'iteration : '+str(i)+'===========')
            for key in weight_init_types.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ' :: '+str(loss))

    markers = {'std=0.01':'o', 'Xavier':'s', 'He':'D'}
    x = np.arange(max_iterations)

    for key in weight_init_types.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.ylim(0, 2.5)
    plt.legend()
    plt.show()


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
