# 데이터 교육 복습 15

#75.ml 연습 5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_ml_6():
    X = np.array([[3,4],[1,4],[2,3],[6,-1],[7,-1],[5,-3]] )
    y = np.array([-1,-1, -1, 1, 1 , 1 ])

    from sklearn.svm import SVC
    
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    print('w : ', clf.coef_)
    print(' b = ', clf.intercept_)
    print(clf.support_)
    print(clf.support_vectors_)
    w0 = clf.coef_[0][0]
    w1 = clf.coef_[0][1]
    a = -w0/w1
    b = -clf.intercept_[0]/w1
    print(w0, w1, a, b)

    half_margin = 1/w1
    x_data = np.arange(11)
    print(half_margin, x_data)

    hyperline= a*x_data + b
    down_line = hyperline + half_margin
    up_line = hyperline - half_margin
    print(hyperline, down_line, up_line)

    plt.figure(figsize=(6,6))
    plt.plot(X[:, 0][y==-1], X[:, 1][y==-1], 'rx')
    plt.plot(X[:, 0][y==1], X[:,1][y==1], 'bo')
    plt.plot(x_data, hyperline)
    plt.plot(x_data, up_line, '--', color='orange')
    plt.plot(x_data, down_line, '--', color='orange')
    plt.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.plot([2,6], [3,-1], color='b')
    plt.ylim(-5,5)
    plt.xlim(0,10)
    plt.yticks(np.arange(plt.ylim()[0], plt.ylim()[1], 1))
    plt.xticks(np.arange(plt.xlim()[0], plt.xlim()[1], 1))
    plt.text(2.3,3, 'B(2,3)')
    plt.text(6,-1.5, 'A(6,-1)')
    plt.show()

    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import SVC

    cancer = load_breast_cancer()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    svc = SVC()
    svc.fit(X_train, y_train)
    print(svc.score(X_train, y_train))
    print(svc.score(X_test, y_test))

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svc=SVC()
    svc.fit(X_train_scaled, y_train)
    print(svc.score(X_train_scaled, y_train))
    print(svc.score(X_test_scaled, y_test))

    Xs = np.array([[1,50],[5, 20], [3, 80],[5,60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf = SVC(kernel='linear', C=100)
    svm_clf.fit(Xs, ys)

    svm_clf = SVC(kernel='linear', C=100)
    svm_clf.fit(Xs, ys)
    plt.figure(figsize=(12,3.2))
    plt.subplot(111)
    plt.plot(Xs[:, 0][ys==1], Xs[:,1][ys==1], 'bo')
    plt.plot(Xs[:, 0][ys==0], Xs[:,1][ys==0], 'ms')
    plot_svc_decision_boundary(svm_clf, 0, 6)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel('$x_1$', fontsize=20, rotation=0)
    plt.title("None Scaler", fontsize=16)
    plt.axis([0, 6, 0, 90])
    plt.show()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf = SVC(kernel='linear', C=100)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(111)
    plt.plot(X_scaled[:, 0][ys==1], X_scaled[:,1][ys==1], 'bo')
    plt.plot(X_scaled[:, 0][ys==0], X_scaled[:,1][ys==0], 'ms')
    plot_svc_decision_boundary(svm_clf, -2, 2)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel('$x_1$', fontsize=20, rotation=0)
    plt.title("Using Scaler", fontsize=16)
    plt.show()

def train_ml_7():
    import mglearn
    from sklearn.svm import SVC
    
    X, y = mglearn.tools.make_handcrafted_dataset()
    print(X)
    print(y)
    svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    #데이터 포인트
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    #서포트 벡터
    sv = svm.support_vectors_
    #dual_coef_ 부호에 의해 서포트 벡터의 클래스 레이블 결정
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel("feature 0")
    plt.ylabel("feature 1")
    plt.show()

    from sklearn.svm import LinearSVC

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    i = 0
    for model, ax in zip([LinearSVC(C=0.01), LinearSVC(C=10), LinearSVC(C=1000)], axes):
        i = i+1
        clf = model.fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=0.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title(clf.__class__.__name__ + '_'+str(i))

    from sklearn.svm import SVC

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1, 2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
    axes[0, 0].legend(["class 0", "class 1", "class 0 support vector", "class 1 support vector"],
                  ncol=4, loc=(.9, 1.2))
    
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0) # 한개만 참일때 True, 나머지는 False
    y_xor = np.where(y_xor, 1, 0) # True 는 1로, 나머지는 0으로 고치기
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
                c='b', marker='o', label='Class 1', s=50)
    plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
                c='r', marker='s', label='Class 0', s=50)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("XOR Problem")
    plt.show()

    svc = SVC(kernel='linear').fit(X_xor, y_xor)
    import matplotlib as mpl

    def plot_xor(X, y, model, title, xmin=-3, xmax=3, ymin=-3, ymax=3):
        XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000),
                            np.arange(ymin, ymax, (ymax-ymin)/1000))
        ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T), XX.shape)
        plt.contourf(XX, YY, ZZ, cmap=mpl.cm.Paired_r, alpha=0.5)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b',
                    marker='o', label='클래스 1', s=50)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r',
                    marker='s', label='클래스 0', s=50)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    plot_xor(X_xor, y_xor, svc, 'XOR:SVC')

    X = np.arange(6).reshape(3, 2)
    print(X)

    from sklearn.preprocessing import FunctionTransformer

    def basis(X):
        return np.vstack([X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2]).T
    
    print(FunctionTransformer(basis).fit_transform(X))
    X_xor2 = FunctionTransformer(basis).fit_transform(X_xor)
    plt.scatter(X_xor2[y_xor==1, 0], X_xor2[y_xor==1, 1], c='b', marker='o', s=50)
    plt.scatter(X_xor2[y_xor==0, 0], X_xor2[y_xor==0, 1], c='r', marker='s', s=50)
    plt.ylim(-6, 6)
    plt.title("Data Distribution")
    plt.xlabel(r"$\phi_1$")
    plt.ylabel(r"$\phi_2$")
    plt.show()

    from sklearn.pipeline import Pipeline

    basimodel = Pipeline([('basis', FunctionTransformer(basis)), ('svc', SVC(kernel='linear'))]).fit(X_xor, y_xor)
    plot_xor(X_xor, y_xor, basimodel, 'kernel')
    plt.show()

    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    scaler = StandardScaler()
    X = iris.data[:, [2, 3]]
    X_data = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_data, iris.target, test_size=0.2, random_state=0)
    model1 = SVC(kernel='linear').fit(X_test, y_test)
    model2 = SVC(kernel='poly', C=1, gamma=10).fit(X_test, y_test)
    model3 = SVC(kernel='rbf', C=1, gamma=1).fit(X_test, y_test)

    def plot_iris(X, y, model, title, xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5):
        XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000),
                            np.arange(ymin, ymax, (ymax-ymin)/1000))
        ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T), XX.shape)
        plt.contourf(XX, YY, ZZ, cmap=mpl.cm.Paired_r, alpha=0.5)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='^', label='0', s=100)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='g', marker='o', label='1', s=100)
        plt.scatter(X[y == 2, 0], X[y == 2, 1], c='b', marker='s', label='2', s=100)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("patal length")
        plt.ylabel("patal width")
        plt.title(title)
        

    plt.figure(figsize=(8, 12))
    plt.subplot(311)
    plot_iris(X_test, y_test, model1, 'Linear SVC')
    plt.figure(figsize=(8, 12))
    plt.subplot(312)
    plot_iris(X_test, y_test, model2, 'poly SVC')
    plt.figure(figsize=(8, 12))
    plt.subplot(313)
    plot_iris(X_test, y_test, model3, 'rbf SVC')
    plt.show()

    from sklearn.datasets import make_blobs

    X, y = make_blobs(random_state=156)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend()
    plt.show()

    l_svm = LinearSVC().fit(X, y)
    print(l_svm.coef_)
    print(l_svm.intercept_)

    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    line = np.linspace(-15,15)
    for coef, intercept, color in zip(l_svm.coef_, l_svm.intercept_, mglearn.cm3.colors):
        plt.plot(line, -(line*coef[0] + intercept) / coef[1], c=color)
    plt.legend(['class 0', 'class 1', 'class 2', 'C.B. 0', 'C.B. 1', 'C.B. 2'], loc=(1.01, 0.3))
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
    model = LinearSVC(C=10)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    pred_y = model.predict(X_test)
    print(np.where(y_test != pred_y))
    plt.scatter(X_test[:, 1], X_test[:, 2], c=y_test)
    plt.show()

    X=iris.data[:, [2,3]]
    y=iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearSVC(C=10)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    plt.figure(figsize=[10,8])
    mglearn.plots.plot_2d_classification(model, X_train, eps=0.5, cm='ocean')
    mglearn.discrete_scatter(X_test[:,0], X_test[:,1], y_test)
    plt.show()

def train_ml_8():
    df = pd.read_csv('C:/Users/eleun/Downloads/handphone_train.csv')
    print(df.head())
    print(df.info())
    df.price_range.value_counts()

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_data = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, stratify=y, random_state=156)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    params = {'C':[0.1, 0.2, 0.5, 1, 2, 5, 10, 20], 'gamma':[0.1, 0.25, 0.3, 0.5, 0.7, 1]}
    svc = SVC(kernel='rbf')
    grid_cv = GridSearchCV(svc, param_grid=params, cv=3, refit=True)
    grid_cv.fit(X_train, y_train)
    print(grid_cv.best_params_)
    print(grid_cv.best_score_)

    best = grid_cv.best_estimator_.fit(X_train, y_train)
    print(best.score(X_test, y_test))
    pred = best.predict(X_test)
    print(accuracy_score(y_test, pred))
    get_clf_eval(y_test, pred)
