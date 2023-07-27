# 데이터 교육 복습 13

#71 ml 연습 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def traintestsplit(X_data, y_data):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=156)
    print(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

def rfc_setting(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_test)
    score = accuracy_score(y_test, pred)
    print(score)
    return rfc, score

def gscv_setting(ml, params, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    grid_cv = GridSearchCV(ml, param_grid=params, cv=3, refit=True)
    grid_cv.fit(X_train, y_train)
    b_params = grid_cv.best_params_
    b_score = grid_cv.best_score_
    estimator = grid_cv.best_estimator_
    print(b_params, b_score, estimator)
    return b_params, b_score, estimator

def ml_bargraph(estimator, name):
    import seaborn as sns
    ftr_importance_values = estimator.feature_importances_
    ftr_importance = pd.Series(ftr_importance_values, index=name)
    ftr_10 = ftr_importance.sort_values(ascending=False)[:10]
    plt.figure(figsize=(8,6))
    sns.barplot(x=ftr_10, y=ftr_10.index)
    plt.show()

def get_clf_eval(y_test, pred=None, pred_probas=None):
        from sklearn.metrics import confusion_matrix, accuracy_score
        from sklearn.metrics import precision_score, recall_score
        from sklearn.metrics import f1_score, roc_auc_score

        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        if pred_probas is not None:
            auc_score = roc_auc_score(y_test, pred_probas)
            print(confusion)
            print(f'정확도 :  {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, f1_score: {f1:.4f}, AUC : {auc_score:.4f}')
            return
        print(confusion)
        print(f'정확도 :  {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, f1_score: {f1:.4f}')

def train_ml_2():
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = traintestsplit(cancer.data, cancer.target)
    rfc = rfc_setting(X_train, X_test, y_train, y_test)
    params = {'n_estimators':[100, 200, 300], 'max_depth':[6,8,10,12], 'min_samples_split':[8, 16, 20]}
    b_params, b_score, estimator = gscv_setting(rfc, params, X_train, y_train)
    rf_best_estimator=estimator
    pred = rf_best_estimator.predict(X_test)
    print(accuracy_score(y_test, pred))
    ml_bargraph(rf_best_estimator, cancer.feature_names)

    otto = pd.read_csv('C:/Users/eleun/Downloads/otto_train.csv')
    print(otto.head())

    label = []
    for i in otto.target:
        label.append(i.split('_')[1])
    X_data = otto.drop(['id','target'], axis=1)
    y_data = label
    X_train, X_test, y_train, y_test = traintestsplit(X_data, y_data)
    rfc = rfc_setting(X_train, X_test, y_train, y_test)
    params = {'n_estimators':[100, 200, 300], 'max_depth':[6, 8, 10, 12], 'min_samples_split':[2,4,6,8]}
    b_params, b_score, estimator = gscv_setting(rfc, params, X_train, y_train)
    rf_best_estimator=estimator
    pred = rf_best_estimator.predict(X_test)
    print(accuracy_score(y_test, pred))

    rfc_hands = RandomForestClassifier(n_estimators=300, min_samples_split=4)
    rfc_hands.fit(X_train, y_train)
    print(accuracy_score(y_test, rfc_hands.predict(X_test)))

    ml_bargraph(rf_best_estimator, X_data.columns)

def gbclf_setting(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    gb_clf = GradientBoostingClassifier(random_state=0)
    gb_clf.fit(X_train, y_train)
    gb_pred = gb_clf.predict(X_test)
    print(accuracy_score(y_test, gb_pred))

def train_ml_3():
    feature_df = pd.read_csv('C:/User/eleun/Downloads/Dataset/features.txt', sep='\s+',header=None, names=['column_index', 'column_name'])
    feature_df['dup_cnt'] = pd.DataFrame(feature_df.groupby('column_name').cumcount())
    print(feature_df.head())
    feature_df['feature_name'] = feature_df.column_name + feature_df.dup_cnt.astype('str')
    print(feature_df.head())
    print(feature_df.teil())
    def get_feature_name(df):
        df['dup_cnt'] = pd.DataFrame(df.groupby('column_name').cumcount())
        df['feature_name'] = df.column_name + df.dup_cnt.astype('str')
        return df.feature_name.tolist()
    def get_train_test_data():
        df = pd.read_csv('C:/User/eleun/Downloads/Dataset/features.txt', sep='\s+',header=None, names=['column_index', 'column_name'])
        f_names = get_feature_name(df)
        X_train = pd.read_csv('C:/User/eleun/Downloads/Dataset/train/X_train.txt', sep='\s+', names=f_names)
        X_test = pd.read_csv('C:/User/eleun/Downloads/Dataset/test/X_test.txt', sep='\s+', names=f_names)
        y_train = pd.read_csv('C:/User/eleun/Downloads/Dataset/train/y_train.txt', sep='\s+', names=['action'])
        y_test = pd.read_csv('C:/User/eleun/Downloads/Dataset/test/y_test.txt', sep='\s+', names=['action'])
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = get_train_test_data()

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    rfc = rfc_setting(X_train, X_test, y_train, y_test)

    print(rfc.get_params())

    from sklearn.model_selection import GridSearchCV

    params = {'max_depth':[6,8,10,12,14], 
          'min_samples_leaf':[2,4,6,8],
          'min_samples_split':[2,4,6,8]}

    from sklearn.ensemble import RandomForestClassifier

    def rfc_setting_njobs(X_train, y_train, params):
        rfc = RandomForestClassifier(random_state=0, n_jobs=-1)
        grid_cv = GridSearchCV(rfc, param_grid=params, cv=3, n_jobs=-1)
        grid_cv.fit(X_train, y_train)
        print(grid_cv.best_params_)
        print(grid_cv.best_score_)
    
    rfc_setting_njobs(X_train, y_train, params)
    
    from sklearn.metrics import accuracy_score

    rfc1 = RandomForestClassifier(n_estimators=200, min_samples_leaf=8 , min_samples_split=2 , random_state=0)
    rfc1.fit(X_train, y_train)
    pred = rfc1.predict(X_test)
    print(accuracy_score(y_test, pred))

    ml_bargraph(rfc1, X_train.columns)

    gbclf_setting(X_train, y_train, X_test, y_test)

    from sklearn.metrics import log_loss

    y_true = [0, 0, 1, 1]
    y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]

    print(log_loss(y_true, y_pred))
    print(log_loss([0, 1], [[0.9, 0.1], [0.2, 0.8]]))
    print(-(np.log(0.9)+np.log(0.8))/2)

    import xgboost as xgb
    from xgboost import plot_importance
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer()

    X_train, X_test, y_train, y_test = traintestsplit(dataset.data, dataset.target)

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    params = {'max_depth':3, 'eta':0.1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'early_stoppings':100}
    wlist=[(dtrain, 'train'), (dtest, 'eval')]

    xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=400, evals=wlist)

    pred_probs = xgb_model.predict(dtest)
    preds = [ 1 if x>0.5 else 0 for x in pred_probs]

    print(preds[:10])
    print(pred_probs[:4])

    get_clf_eval(y_test, preds, pred_probs)

    fig, ax=plt.subplots(figsize=(10,12))
    plot_importance(xgb_model, ax=ax)
    plt.show()

    from xgboost import XGBClassifier
    xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
    xgb_wrapper.fit(X_train, y_train, early_stopping_rounds=800, eval_set=[(X_test, y_test)], eval_metric='logloss')
    w_preds = xgb_wrapper.predict(X_test)

    get_clf_eval(y_test, w_preds)
