# 데이터 교육 복습 16

#75. ml 연습 6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_ml_9():
    from sklearn.svm import SVC
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    svm = SVC()
    svm.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    print(svm.score(X_test_scaled, y_test))

    from sklearn.model_selection import GridSearchCV

    params = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(SVC(), param_grid=params, cv=5)
    grid.fit(X_train_scaled, y_train)
    print(grid.best_score_)
    print(grid.score(X_test_scaled, y_test))
    print(grid.best_params_)

    from sklearn.pipeline import Pipeline

    pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))

    params = {'svm__C':[0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(pipe, param_grid=params, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_score_)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn import set_config

    pipe = make_pipeline(StandardScaler(), PCA(), LogisticRegression())
    set_config(display='diagram')
    pipe

    pipe = Pipeline([('scaler', StandardScaler()), ('svm', PCA())])
    pipe = make_pipeline(StandardScaler(), PCA(), LogisticRegression())
    print(pipe)
    print(pipe.steps)
    print(pipe.steps[1][1])
    print(pipe.steps[0][0])
    print(pipe.named_steps)
    print(pipe.named_steps['pca'].components_.shape)
    print(pipe.named_steps['pca'].components_)

    bostonDF = pd.read_csv('C:/Users/eleun/Downloads/boston_house.csv')
    print(bostonDF.head())
    print(bostonDF.info())
    bostonDF.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(bostonDF.head())
    print(bostonDF.info())
    y_target = bostonDF['PRICE']
    X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.3, random_state=156)
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge

    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
    param_grid = {'polynomialfeatures__degree':[1,2,3], 'ridge__alpha':[0.01, 0.1, 1]}
    grid = GridSearchCV(pipe, param_grid= param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.score(X_test, y_test))

    from sklearn.ensemble import RandomForestClassifier

    pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
    param_grid = [
        {'classifier':[SVC()], 'preprocessing':[StandardScaler()],
         'classifier__gamma':[0.001, 0.01, 0.1, 1, 10, 100],
         'classifier__C':[0.1, 1, 10, 100]},
         {'classifier':[RandomForestClassifier(n_estimators=50)], 'preprocessing':[None], 'classifier__max_features':[1, 2, 3]}]

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error

    # 2차원 배열을 만들어 'data'라는 변수에 할당
    data = {'x' : [13, 19, 16, 14, 15, 18],
            'y' : [40, 83, 62, 57, 58, 63]}

    # data라는 변수의 값을 data frame 형태로 변환
    data = pd.DataFrame(data)
    print(data)

    X = pd.DataFrame(data['x'])
    y = data.y
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    print(lr.intercept_)
    print(lr.coef_)
    data.plot(kind='scatter', x='x', y='y')
    plt.show()
    prediction = lr.predict(X)
    residuals = data['y'] - prediction
    print(residuals)
    for i in range(len(residuals)):
        print(data['y'][i], '-', np.round(prediction[i],2), '=', np.round(residuals[i], 2))
    RSS = (residuals**2).sum()
    print(RSS)
    MSE = RSS/6
    print(MSE)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    TSS = ((data['y']-data['y'].mean())**2).sum()
    print(TSS)
    R_squared = 1 - (RSS/TSS)
    print(R_squared)
    print(lr.score(pd.DataFrame(data['x']), data['y']))

def train_ml_10():
    np.random.seed(0)
    X = 2 * np.random.rnad(100, 1)
    print(X)
    y = 6 + 4 * X + np.random.randn(100, 1)
    print(y.shape)
    print(X.shape)
    plt.scatter(X, y)
    plt.show()

    def get_cost(y, y_preds):
        N = len(y)
        cost = np.sum(np.square(y-y_preds))/N
        return cost
    
    def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
        N = len(y)
        w1_update = np.zeros_like(w1)
        w0_update = np.zeros_like(w0)
        y_pred = np.dot(X, w1.T) + w0
        diff = y-y_pred
        w0_factors = np.ones((N, 1))
        w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
        w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))
        return w1_update, w0_update
    
    def gradient_descent_steps(X, y, iters=100000):
        w0 = np.zeros((1,1))
        w1 = np.zeros((1,1))

        for ind in range(iters):
            w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
            w1 = w1 - w1_update
            w0 = w0 - w0_update
        return w1, w0
    
    w1, w0 = gradient_descent_steps(X, y, iters=10000)
    y_pred = w1[0,0]*X + w0
    print('w1 :', w1[0,0], 'w0 :', w0[0,0])
    print('total Cost: ' , get_cost(y, y_pred))
    plt.scatter(X, y)
    plt.plot(X, y_pred, color='black')
    plt.show()
    
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X, y)
    print(lr.coef_, lr.intercept_)

    def gradient_descent_steps(X, y, iters=100000):
        w0 = np.zeros((1,1))
        w1 = np.zeros((1,1))

        for ind in range(iters):
            np.random.seed(ind)
            stochastic_random_index = np.random.permutation(X.shape[0])
            sample_X = X[stochastic_random_index[0:10]]
            sample_y = y[stochastic_random_index[0:10]]
            w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
            w1 = w1 - w1_update
            w0 = w0 - w0_update
        return w1, w0
    
    print(np.random.permutation(X.shape[0]))
    w1, w0 = gradient_descent_steps(X, y, iters=100000)
    y_pred = w1[0,0]*X + w0
    print('w1 : ', w1[0,0], 'w0 : ',w0[0,0])
    print('total Cost: ', get_cost(y, y_pred))
    plt.scatter(X, y)
    plt.plot(X, y_pred, 'k')
    plt.show()

    bostonDF = pd.read_csv('C:/Users/eleun/Downloads/boston_house.csv')
    y_target = bostonDF['PRICE']
    X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)
    print(X_data)
    print(y_target)
    print(X_data.columns)
    fig, axes = plt.subplots(figsize=(16, 8), ncols=7, nrows=2)
    for i, feature in enumerate(X_data.columns):
        row = int(i/7)
        col = i % 7
        sns.regplot(x=feature, y=y_target, data=X_data, ax=axes[row][col])
        plt.show()

    print(row)
    print(col)
    print(X_data.shape, y_target.shape)
    bostonDF.plot(kind='scatter', x='RM', y='PRICE', figsize=(6,6), color='blue', xlim=(4,8), ylim=(10,45))
    plt.show()
    y_target = bostonDF['PRICE']
    X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.3, random_state=156)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    lr = LinearRegression()
    lr.fit(X=pd.DataFrame(bostonDF['RM']), y=bostonDF['PRICE'])
    print(lr.coef_, lr.intercept_)
    prediction = lr.predict(pd.DataFrame(bostonDF['RM']))
    print(prediction)
    bostonDF.plot(kind='scatter', x='RM', y='PRICE', figsize=(6,6), color='blue', xlim=(4,8), ylim=(10,45))
    plt.plot(bostonDF['RM'], prediction, color='red')
    plt.show()
    print(lr.score(X=pd.DataFrame(bostonDF['RM']), y=bostonDF['PRICE']))
        
    from sklearn.metrics import mean_squared_error, r2_score

    print(mean_squared_error(bostonDF['PRICE'], prediction))
    print(mean_squared_error(bostonDF['PRICE'], prediction)**0.5)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.3, random_state=156)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(X_train.head())
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_preds = lr.predict(X_test)
    print(r2_score(y_test, y_preds))
    print(mean_squared_error(y_test, y_preds))
    print(np.sqrt(mean_squared_error(y_test, y_preds)))

    from scipy import stats

    fig, ax = plt.subplots()
    stats.probplot(y_test-y_preds, plot=ax)
    plt.show()
    fig, ax = plt.subplots()
    stats.probplot(bostonDF['PRICE'], plot=ax)
    plt.show()
    
    from sklearn.preprocessing import PolynomialFeatures

    X = np.arange(4).reshape(2,2)
    print(X)
    poly = PolynomialFeatures(degree=2)
    poly.fit()
    poly_ftr = poly.transform(X)
    print(poly_ftr)
    print(poly.get_feature_names_out())

def train_ml_11():
    bostonDF = pd.read_csv('C:/Users/eleun/Downloads/boston_house.csv')
    print(bostonDF.head())
    bostonDF.drop('Unnamed: 0', axis=1, inplace=True)
    y_target = bostonDF['PRICE']
    X_data = bostonDF.drop('PRICE', axis=1, inplace=False)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.3, random_state=156)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_preds = lr.predict(X_test)

    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_test, y_preds)
    print(mse)
    print(np.sqrt(mse))
    lr = LinearRegression()
    neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
    print(neg_mse_scores)
    print(-1*neg_mse_scores)
    rmse_scores = np.mean(np.sqrt(-1*neg_mse_scores))
    print(rmse_scores)
    lr.fit(X_train, y_train)
    print(lr.intercept_)
    print(np.round(lr.coef_, 1))
    coeff = pd.Series(data=np.round(lr.coef, 1), index=X_data.columns)
    print(coeff.sort_values(ascending=False))
    
    X = np.arange(0,4).reshape(2,2)
    def polynomial_func(x):
        y = 1 + 2*x[:,0] + 3*x[:,0]**2 + 4*x[:,1]**3
        return y
    
    y = polynomial_func(X)
    print(y)
    print(X)

    from sklearn.preprocessing import PolynomialFeatures

    poly_ftr = PolynomialFeatures(degree=3).fit_transform(X)
    print(poly_ftr)
    lr = LinearRegression()
    lr.fit(poly_ftr, y)
    print(np.round(lr.coef_, 2))
    poly = PolynomialFeatures(3)
    poly.fit(X)
    print(poly.get_feature_names_out())

    from sklearn.pipeline import Pipeline

    print(X)
    print(y)

    pipe = Pipeline([('poly', PolynomialFeatures(3)), ('linear', LinearRegression())])
    pipe.fit(X, y)
    print(pipe.named_steps['linear'].coef_)
    def true_func(X):
        return np.cos(1.5 * np.pi * X)
    np.random.seed(0)
    n_samples = 30
    X = np.sort(np.random.rand(n_samples))
    y = true_func(X) + np.random.randn(n_samples)*0.1
    plt.scatter(X, y)
    plt.show()

    degrees = [1, 4, 15]
    for i in degrees:
        print(i)
    print(X)

    plt.figure(figsize=(14, 5))
    for i, j in enumerate(degrees):
        polynomial_features = PolynomialFeatures(degree=j , include_bias=False)
        linear_regression = LinearRegression()
        pipe = Pipeline([('polynomial_features', polynomial_features), ('linear_regression', linear_regression)])

        scores = cross_val_score(pipe, X.reshape(-1, 1), y, scoring='neg_mean_squared_error', cv=10)
        pipe.fit(X.reshape(-1, 1), y)
        coefs = pipe.named_steps['linear_regression'].coef_

        print('\n Degree {0}의 회귀계수 : {1}'.format(j, np.round(coefs, 2)))
        print('\n Degree {0}의 MSE : {1}'.format(j, -1*np.mean(scores)))

        ax=plt.subplot(1,3,i+1)
        plt.setp(ax, xticks=(), yticks=())
        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipe.predict(X_test[:, np.newaxis]), label='Model')

        plt.plot(X_test, true_func(X_test), '--', label='True Function')
        plt.scatter(X, y, edgecolor='b', s=20, label='Samples')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.title('Degree {} \n MSE = {:.2e}(+/-{:.2e})'.format(j, -scores.mean(), scores.std()))
        plt.show()

    from sklearn.linear_model import Ridge

    y_target = bostonDF['PRICE']
    X_data = bostonDF.drop('PRICE', axis=1, inplace=False)
    ridge = Ridge(alpha=10)
    neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
    print(neg_mse_scores)
    rmse = np.sqrt(-1*neg_mse_scores)
    print(rmse)
    print(np.mean(rmse))
    alphas = [0, 0.1, 1, 10, 100]

    for i in alphas:
        ridge = Ridge(alpha=i)
        neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
        avg_rmse = np.mean(np.sqrt(-1*neg_mse_scores))
        print(i, ' : ', avg_rmse)

    for i in alphas:
        ridge = Ridge(alpha=i)
        ridge.fit(X_data, y_target)
        coef = pd.Series(data=ridge.coef_, index=X_data.columns)
        print('----', i, '----------')
        print(coef)

    def alpha_w_plot(alpha, coef, pos, fig, axes):
        plot_title = 'alpha : '+str(alpha)
        coeff = coef.sort_values(ascending=False)
        axes[pos].set_title(plot_title)
        axes[pos].set_xlim(-3, 6)
        sns.barplot(x=coeff.values, y=coeff.index, ax=axes[pos])

    fig, axes = plt.subplots(figsize=(18,6), nrows=1, ncols=5)
    for pos, alpha in enumerate(alphas):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_data, y_target)
        coef = pd.Series(data=ridge.coef_, index=X_data.columns)
        alpha_w_plot(alpha, coef, pos, fig, axes)
        plt.show()
