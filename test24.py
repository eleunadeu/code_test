# 데이터 교육 복습 24

#83. 딥러닝 연습 4
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
