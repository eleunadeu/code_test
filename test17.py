# 데이터 교육 복습 17

#76.ml 연습 7
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def train_ml_12():
    bostonDF = pd.read_csv('C:/Users/eleun/Downloads/boston_house.csv')
    y_target = bostonDF['PRICE']
    X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)

    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    def get_linear_reg_eval(model_name, params=None, X_data=None, y_target=None):
        coeff_df = pd.DataFrame()
        for param in params:
            if model_name == 'Ridge':
                model = Ridge(alpha=param)
            if model_name == 'Lasso':
                model = Lasso(alpha=param)
            if model_name == 'ElasticNet':
                model = ElasticNet(alpha=param, l1_ratio=0.7)
            model.fit(X_data, y_target)
            coeff = pd.Series(data=model.coef_, index=X_data.columns)
            colname = 'alpha : '+str(param)
            coeff_df[colname] = coeff
            neg_mse_scores = cross_val_score(model, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
            avg_rmse = np.mean(np.sqrt(-1*neg_mse_scores))
            print(param, ' : ', avg_rmse)
        return coeff_df
    
    alphas = [0.07, 0.1, 0.5, 1.0, 3.0]
    result = get_linear_reg_eval('Lasso', params=alphas, X_data=X_data, y_target=y_target)
    print(result.sort_values(by=result.columns[0], ascending=False))
    result = get_linear_reg_eval('ElasticNet', alphas, X_data, y_target)
    print(result.sort_values(by=result.columns[0], ascending=False))
    
    print(np.log(2.3))
    print(np.log(0))
    print(np.log1p(0))
    print(np.log(1+0))
    print(np.log(2.3), np.log(1+2.3), np.log1p(2.3))
    print(np.expm1(np.log1p(2.3)))

    alphas = [0.1, 1, 10, 100]
    scale_methods = [(None, None), ('Standard', None), ('Standard', 2), ('MInMax', None), ('MinMax', 2), ('Log', None)]
    
    def get_linear_reg_eval(model_name, params=None, X_data=None, y_target=None):
        param_avg_rmse_dict = {}
        for param in params:
            if model_name == 'Ridge':
                model = Ridge(alpha=param)
            if model_name == 'Lasso':
                model = Lasso(alpha=param)
            if model_name == 'ElasticNet':
                model = ElasticNet(alpha=param, l1_ratio=0.7)
            neg_mse_scores = cross_val_score(model, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
            avg_rmse = np.mean(np.sqrt(-1*neg_mse_scores))
            param_avg_rmse_dict[str(param)] = avg_rmse
        return param_avg_rmse_dict
    
    def get_scaled_data(method=None, p_degree=None, input_data=None):
        if method == 'Standard':
            scaled_data = StandardScaler().fit_transform(input_data)
        elif method == 'MinMax':
            scaled_data = MinMaxScaler().fit_transform(input_data)
        elif method == 'Log':
            scaled_data = np.log1p(input_data)
        else:
            scaled_data = input_data
        if p_degree != None:
            scaled_data = PolynomialFeatures(degree=p_degree, include_bias=False).fit_transform(scaled_data)
        return scaled_data
    
    result_dict = {}
    for sm in scale_methods:
        X_data_scaled = get_scaled_data(method=sm[0], p_degree=sm[1], input_data=X_data)
        result = get_linear_reg_eval('Ridge', params=alphas, X_data=X_data_scaled, y_target=y_target)
        print(result, sm)
        result_dict[str(sm)] = result

    print(result)

    scale_methods = [StandardScaler(), MinMaxScaler(), (StandardScaler(), \
                 PolynomialFeatures(degree=2, include_bias=False)),
                 (MinMaxScaler(), PolynomialFeatures(degree=2, include_bias=False))]
    alphas = [0.1, 1, 10, 100]

    def linear_reg_eval_pipe(alphas=None, method=None, X_data=None, y_target=None):
        result=[]
        alpha_dict={}
        for a1 in alphas:
            sm_mn={}
            for model in [Ridge(alpha=a1), Lasso(alpha=a1), ElasticNet(alpha=a1, l1_ratio=0.7)]:
                mn = type(model).__name__
                for scale_method in method:
                    if type(scale_method) == tuple:
                        pipe = make_pipeline(scale_method[0], scale_method[1], model)
                        sn = type(scale_method[0]).__name__[:2] + type(scale_method[1]).__name__[:2]
                    else:
                        pipe = make_pipeline(scale_method, model)
                        sn = type(scale_method).__name__[:4]
                    neg_mse_scores = cross_val_score(pipe, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
                    avg_rmse = np.mean(np.sqrt(-1*neg_mse_scores))
                    result.append((str(a1), mn, sn, avg_rmse))
                    sm_mn[mn[:5]+' : '+sn] = avg_rmse
                alpha_dict['alpha_'+str(a1)] = sm_mn
        return alpha_dict
    
    lrep_result = linear_reg_eval_pipe(alphas, scale_methods, X_data, y_target)
    df = pd.DataFrame(lrep_result)
    print(df)

def train_ml_13():
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    x = np.arange(-15, 15, 0.01)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()

    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    cancer = load_breast_cancer()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)
    lr_clf = LogisticRegression(max_iter=5000)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)

    from sklearn.metrics import accuracy_score, roc_auc_score

    print(accuracy_score(y_test, lr_pred))
    print(roc_auc_score(y_test, lr_pred))

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    lr_clf.fit(X_train_scaled, y_train)
    lr_pred = lr_clf.predict(X_test_scaled)
    print(accuracy_score(y_test, lr_pred))
    print(roc_auc_score(y_test, lr_pred))

    params = {'penalty':['l2'], 'C':[0.01, 0.1, 1, 3, 5, 10]}

    from sklearn.model_selection import GridSearchCV

    grid_lr_clf = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=3)
    grid_lr_clf.fit(X_train_scaled, y_train)
    print(grid_lr_clf.best_params_)
    print(grid_lr_clf.best_score_)
    
    card = pd.read_csv('C:/Users/eleun/Downloads/creditcard.csv')
    print(card.head())
    print(card.info())
    print(card.Class.value_counts())
    card = card.drop('Time', axis=1, inplace=True)
    X_features = card.iloc[:, :-1]
    y_target = card.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    print(y_train.value_counts()/y_train.shape[0]*100)
    print(y_test.value_counts()/y_test.shape[0]*100)
    lr_clf = LogisticRegression(max_iter=5000)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_probas = lr_clf.predict_proba(X_test)[:,1]
    get_clf_eval(y_test, lr_pred, lr_probas)
    
    plt.figure(figsize=(8, 4))
    sns.distplot(card.Amount)
    plt.show()

    amount = pd.DataFrame(card.Amount)
    amount = scaler.fit_transform(amount)
    card_df = card.copy()
    card_df.drop('Amount', axis=1, inplace=True)
    card_df.insert(0, 'Amount_scaled')
    X_features = card_df.iloc[:, :-1]
    y_target = card_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    lr_clf = LogisticRegression(max_iter=5000)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_probas = lr_clf.predict_proba(X_test)[:,1]
    get_clf_eval(y_test, lr_pred, lr_probas)

    amount_n = np.log1p(card.Amount)
    card_df.insert(0, 'Amount_log', amount_n)
    new_df = card_df.drop('Amount_scaled', axis=1)
    X_features = new_df.iloc[:, :-1]
    y_target = new_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    lr_clf = LogisticRegression(max_iter=5000)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_probas = lr_clf.predict_proba(X_test)[:,1]
    get_clf_eval(y_test, lr_pred, lr_probas)

def train_ml_14():
    boston_df = pd.read_csv('C:/Users/eleun/Downloads/boston_house.csv')
    print(boston_df.head())
    boston_df.drop('Unnamed: 0', axis=1, inplace=True)
    y_target = boston_df.PRICE
    X_data = boston_df.drop('PRICE', axis=1, inplace=False)

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    
    rf = RandomForestRegressor(random_state=0, n_estimators=1000)
    neg_mse_scores = cross_val_score(rf, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
    print(neg_mse_scores)
    rmse_scores = np.sqrt(-1*neg_mse_scores)
    print(np.mean(rmse_scores))
    print(rmse_scores)

    def reg_avg_rmse_eval(models, X_data, y_target):
        for model in models:
            neg_mse_scores = cross_val_score(model, X_data, y_target, scoring='neg_mean_squared_error', cv=5)
            rmse_scores = np.sqrt(-1*neg_mse_scores)
            avg_rmse_scores = np.mean(rmse_scores)
            print(f'Model : {type(model).__name__} RMSE 평균 : {avg_rmse_scores}')
    
    dt_reg = DecisionTreeRegressor(max_depth=4, random_state=0)
    rf_reg = RandomForestRegressor(n_estimators=1000, random_state=0)
    gb_reg = GradientBoostingRegressor(n_estimators=1000, random_state=0)
    xgb_reg = XGBRegressor(n_estimators=1000, random_state=0)
    lgbm_reg = LGBMRegressor(n_estimators=1000, random_state=0)
    models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgbm_reg]

    reg_avg_rmse_eval(models, X_data, y_target)

    rf_reg.fit(X_data, y_target)
    feature_series = pd.Series(data=rf_reg.feature_importances_, index=X_data.columns)
    feature_series = feature_series.sort_values(ascending=False)
    sns.barplot(x=feature_series, y=feature_series.index)
    plt.show()

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    dtr1 = DecisionTreeRegressor(max_depth=2)
    dtr2 = DecisionTreeRegressor(max_depth=7)

    bostonDF_sample = boston_df[['RM']]
    bostonDF_sample['PRICE'] = y_target.values
    bostonDF_sample = bostonDF_sample.sample(n=100, random_state=0)
    print(bostonDF_sample.info())

    plt.figure()
    plt.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c='darkorange')
    plt.show()

    X_features = bostonDF_sample['RM'].values.reshape(-1, 1)
    y_target = bostonDF_sample['PRICE'].values.reshape(-1, 1)
    lr.fit(X_features, y_target)
    dtr1.fit(X_features, y_target)
    dtr2.fit(X_features, y_target)
    X_test = np.arange(4.0, 8.5, 0.04).reshape(-1, 1)
    pred_lr = lr.predict(X_test)
    pred_dt1 = dtr1.predict(X_test)
    pred_dt2 = dtr2.predict(X_test)

    fig , (ax1, ax2, ax3) = plt.subplots(figsize=(14,4), ncols=3)

    ax1.set_title('Linear Regression')
    ax1.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
    ax1.plot(X_test, pred_lr,label="linear", linewidth=2 )

    ax2.set_title('Decision Tree Regression: \n max_depth=2')
    ax2.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
    ax2.plot(X_test, pred_dt1, label="max_depth:2", linewidth=2 )

    ax3.set_title('Decision Tree Regression: \n max_depth=7')
    ax3.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
    ax3.plot(X_test, pred_dt2, label="max_depth:7", linewidth=2)
    plt.show()

def train_ml_15(): 
    bike_df = pd.read_csv('C:/Users/eleun/Downloads/bike_train.csv')
    print(bike_df.head())
    print(bike_df.info())
    bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime)
    print(bike_df.info())
    bike_df['year'] = bike_df.datetime.apply(lambda x: x.year)
    bike_df['month'] = bike_df.datetime.apply(lambda x: x.month)
    bike_df['day'] = bike_df.datetime.apply(lambda x: x.day)
    bike_df['hour'] = bike_df.datetime.apply(lambda x: x.hour)
    print(bike_df)
    
    drop_columns = ['datetime', 'casual', 'registered']
    bike_df.drop(drop_columns, axis=1, inplace=True)

    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    y_target = bike_df['count']
    X_features = bike_df.drop(['count'], axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0)
    lr_reg = LinearRegression()
    lr_reg.fit(X_train, y_train)
    pred = lr_reg.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def evaluate_regr(y, pred):
        rmse_val = np.sqrt(mean_squared_error(y, pred))
        mae_val = mean_absolute_error(y, pred)
        print(f'RMSE : {rmse_val:.3F}, MAE : {mae_val:.3F}')

    evaluate_regr(y_test, pred)

    def get_top_error_data(y_test, pred, n_tops=5):
        result_df = pd.DataFrame(y_test.values, columns=['real_count'])
        result_df['predicted_count'] = np.round(pred)
        result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])
        print(result_df.sort_values('diff', ascending=False).iloc[0:n_tops,:])

    get_top_error_data(y_test, pred)

    y_target.hist()
    plt.show()

    y_log_transform = np.log1p(y_target)
    y_log_transform.hist()
    plt.show()

    y_target_log = np.log1p(y_target)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size=0.3, random_state=0)
    lr_reg = LinearRegression()
    lr_reg.fit(X_train, y_train)
    pred = lr_reg.predict(X_test)
    y_test_exp = np.expm1(y_test)
    pred_exp = np.expm1(pred)
    evaluate_regr(y_test_exp, pred_exp)
    get_top_error_data(y_test_exp, pred_exp, n_tops=5)
    coef = pd.Series(lr_reg.coef_, index=X_features.columns)
    coef_sort = coef.sort_values(ascending=False)
    sns.barplot(x=coef_sort.values, y=coef_sort.index)
    plt.show()
