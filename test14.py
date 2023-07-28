# 데이터 교육 복습 14

#72. ml연습 4
def xgbclf_setting(X_train, X_test, y_train, y_test):
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    xgb_clf = XGBClassifier(silent=True, n_estimators=500, random_state=156)
    xgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=[(X_train, y_train), (X_test, y_test)])
    print(roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1]))

def lgbm_setting(X_train, X_test, y_train, y_test):
    import lightgbm
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    lgbm_clf = LGBMClassifier(n_estimators=500)
    lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='logloss',eval_set=[(X_train, y_train), (X_test, y_test)])
    print(roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:, 1]))

def train_ml_4():
    from sklearn.datasets import load_breast_cancer
    import xgboost as xgb
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X_features = data.data
    y_labels = data.target
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=156)

    import lightgbm
    from lightgbm import LGBMClassifier
    from lightgbm import plot_importance

    lgbm = LGBMClassifier(n_estimators=400)
    evals = [(X_test, y_test)]
    lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals)
    preds = lgbm.predict(X_test)
    get_clf_eval(y_test, preds)
    fig, ax = plt.subplots(figsize=(10,12))
    plot_importance(lgbm, ax=ax)
    plt.show()

    cust_df = pd.read_csv('C:/Users/eleun/Downloads/santander_train.csv')
    print(cust_df.head())
    print(cust_df.isna().sum().sum())
    pd.options.display.max_info_columns=1000
    print(cust_df.info())
    print(cust_df['TARGET'].value_counts()/cust_df.TARGET.count())
    print(cust_df.var3.value_counts())
    cust_df['var3'].replace(-999999, 2, inplace=True)
    print(cust_df.var3.value_counts())
    cust_df.drop('ID', axis=1, inplace=True)

    X_features=cust_df.iloc[:, :-1]
    y_labels=cust_df.iloc[:, -1]
    print(X_features.shape, y_labels.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=0, stratify=y_labels)
    print(y_train.value_counts()/y_train.count())
    print(y_test.value_counts()/y_test.count())

    xgbclf_setting(X_train, X_test, y_train, y_test)
    lgbm_setting(X_train, X_test, y_train, y_test)

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score

    params = {'max_depth':[5, 7], 'colsample_bytree':[0.5, 0.75]}
    xgb_clf = XGBClassifier(n_estimators=100)
    grid_cv = GridSearchCV(xgb_clf, param_grid=params)
    grid_cv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    print(grid_cv.best_params_)
    print(roc_auc_score(y_test, grid_cv.best_estimator_.predict_proba(X_test)[:,1]))

    XGB_clf = XGBClassifier(n_estimators=200, max_depth=7, colsample_tree=0.75, learning_rate=0.02, random_state=156)
    XGB_clf.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    print(roc_auc_score(y_test, XGB_clf.predict_proba(X_test)[:,1]))

    LGBM_clf = LGBMClassifier(n_estimator=200)
    params = {'num_leaves':[32, 64], 'max_depth':[128, 160], 'min_child_samples':[60, 100], 'subsample':[0.8, 1]}
    grid_cv = GridSearchCV(LGBM_clf, param_grid=params)
    grid_cv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    grid_cv.best_params_
    print(roc_auc_score(y_test, grid_cv.best_estimator_.predict_proba(X_test)[:,1]))

    lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=32, subsample=0.8, min_cild_samples=100, max_depth=128)
    lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc', eval_set=[(X_test, y_test)], verbose=False)
    print(roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:, 1]))

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    rf_clf = RandomForestClassifier(n_estimators=200, random_state=156)
    rf_clf.fit(X_train, y_train)
    print(roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1]))

    dt_clf = DecisionTreeClassifier(random_state=156)
    dt_clf.fit(X_train, y_train)
    print(roc_auc_score(y_test, dt_clf.predict_proba(X_test)[:, 1]))

def train_ml_5():
    import mglearn
    from sklearn.neighbors import KNeighborsClassifier
    
    dataset =[[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

    row0 = [7, 2]

    np.array(dataset)[:,0:2]
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(np.array(dataset)[:,0:2], np.array(dataset)[:, 2])
    y_knn_pred = knn.predict([[7,2]])

    rows = np.array([[7,2]])

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for n_neighbors, ax in zip([1, 3, 5], axes):
        # fit 메소드는 self 오브젝트를 리턴합니다
        # 그래서 객체 생성과 fit 메소드를 한 줄에 쓸 수 있습니다
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(np.array(dataset)[:,0:2], np.array(dataset)[:, 2])
        mglearn.plots.plot_2d_separator(clf, np.array(dataset)[:,0:2], fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(np.array(dataset)[:,0:2][:, 0], np.array(dataset)[:,0:2][:, 1],  np.array(dataset)[:,2], ax=ax)
        ax.scatter(rows[:, 0], rows[:, 1], c='black', marker='*', s=80, label='test')
        ax.set_title("{}_neighbors".format(n_neighbors))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend(loc=3)
    plt.show()    

    x= []
    y= []
    for i in range(len(dataset)):
        x.append(dataset[i][0])
        y.append(dataset[i][1])

    plt.scatter(x=x, y=y)
    plt.scatter(row0[0], row0[1], s=100, alpha=0.8)
    plt.grid()
    plt.show()

    from math import sqrt
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i]-row[i])**2
        return sqrt(distance)
    # row = [x, y, label]
    # row0 = [x, y]
    for row in dataset:
        distance = euclidean_distance(row0, row)
        print(distance)

    dist_list = []
    for row in dataset:
        distance = euclidean_distance(row0, row)
        dist_list.append((distance, row))
    
    print(dist_list)
    dist_list.sort()
    print(dist_list)

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
    training_accuracy = []
    test_accuracy = []
    k = range(1, 101)

    for n in k:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, y_train)
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))

    plt.plot(k, training_accuracy, label='train accuracy')
    plt.plot(k, test_accuracy, label='test accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('n_neighbors')
    plt.legend()
    plt.grid()
    plt.show()

    from sklearn.model_selection import GridSearchCV

    clf = KNeighborsClassifier()
    params = {'n_neighbors':range(1, 101)}
    grid_cv = GridSearchCV(clf, param_grid=params, scoring='accuracy', cv=5)
    grid_cv.fit(X_train, y_train)
    print(grid_cv.best_params_)
    print(grid_cv.best_score_)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = KNeighborsClassifier()
    grid_cv = GridSearchCV(clf, param_grid=params, scoring='accuracy', cv=5)
    grid_cv.fit(X_train, y_train)
    print(grid_cv.best_params_)
    print(grid_cv.best_score_)

    citrus = pd.read_csv('C:Users/eleun/Downloads/citrus.csv')
    print(citrus)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(citrus.name)
    citrus['target'] = le.transform(citrus.name)
    print(citrus)
    print(citrus.value_counts())
    
    X_features = citrus.iloc[:, 1:-1]
    y_labels = citrus.target

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, stratify=y_labels, random_state=156)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    plt.scatter(citrus[citrus.name=='orange'].diameter,citrus[citrus.name=='orange'].weight)
    plt.scatter(citrus[citrus.name=='grapefruit'].diameter,citrus[citrus.name=='grapefruit'].weight)
    plt.show()

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # 결정 경계에서 w0*x0 + w1*x1 + b = 0 이므로
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
