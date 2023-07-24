# 데이터 교육 복습 10

#68. merchine learning 연습
def train_sklearn_ml():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import accuracy_score
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)
    dt_clf = DecisionTreeClassifier()
    parmeters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
    grid_dt_clf = GridSearchCV(dt_clf, param_grid=parmeters, cv=3, refit=True)
    grid_dt_clf.fit(X_train, y_train)
    s_df = pd.DataFrame(grid_dt_clf.cv_results_)
    print(s_df)
    s_df[['params', 'mean_test_score', 'rank_test_score']]
    grid_dt_clf = GridSearchCV(dt_clf, param_grid=parmeters, cv=3, refit=True, scoring='accuracy')
    grid_dt_clf.fit(X_train, y_train)
    s_df = pd.DataFrame(grid_dt_clf.cv_results_)
    print(s_df[['params', 'mean_test_score', 'rank_test_score']])
    print(grid_dt_clf.best_params_)
    print(grid_dt_clf.best_score_)
    estimator = grid_dt_clf.best_estimator_
    pred = estimator.predict(X_test)
    print(accuracy_score(y_test, pred))
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    print(iris_df.mean())
    print(iris_df.var())
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(iris_df)
    iris_scaled = scaler.transform(iris_df)
    print(iris_scaled[:2])
    print(type(iris_scaled))
    iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
    print(iris_df_scaled.mean())
    print(iris_df_scaled.var())
    from sklearn.preprocessing import MinMaxScaler
    print(iris_df.head())
    print(iris_df.min())
    print(iris_df.max())
    scaler = MinMaxScaler()
    scaler.fit(iris_df)
    iris_scaled = scaler.transform(iris_df)
    print(iris_scaled[:2])
    iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
    print(iris_df_scaled.min())
    print(iris_df_scaled.max())

def train_sklearn_ml_2():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    titanic = pd.read_csv('C:/Users/eleun/Downloads/titanic_train.csv')
    y_titanic = titanic['Survived']
    x_titanic = titanic.drop('Survived', axis=1)
    print(x_titanic)
    print(x_titanic.info())
    def fillna(df):
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        df['Cabin'].fillna('N', inplace=True)
        df['Embarked'].fillna('N', inplace=True)
        return df

    def drop_features(df):
        df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
        return df

    def format_features(df):
        df['Cabin'] = df['Cabin'].str[:1]
        features = ['Cabin', 'Sex', 'Embarked']
        for feature in features:
            le = LabelEncoder()
            le.fit(df[feature])
            df[feature] = le.transform(df[feature])
        return df

    def adj_scale(df):
        scaler = MinMaxScaler()
        scaler.fit(df)
        scaled_array = scaler.transform(df)
        df_scaled = pd.DataFrame(data=scaled_array, columns=df.columns)
        return df_scaled

    def transform_features(df):
        df = fillna(df)
        df = drop_features(df)
        df = format_features(df)
        df = adj_scale(df)
        return df
    
    X_titanic = transform_features(x_titanic)
    print(X_titanic)
    X_train, X_test, y_train, y_test = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=11)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(X_train.info())
    
    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()
    lr_clf = LogisticRegression()
    
    def fit_predict(estimator, X_train, y_train, X_test, y_test):
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        return (accuracy_score(y_test, pred))
    
    print(fit_predict(dt_clf, X_train, y_train, X_test, y_test))
    print(fit_predict(rf_clf, X_train, y_train, X_test, y_test))
    print(fit_predict(lr_clf, X_train, y_train, X_test, y_test))
    n_iter = 0
    kfold = KFold(n_splits=5)
    score = []
    for train_index, test_index in kfold.split(X_titanic):
        X_train = X_titanic.iloc[train_index]
        X_test = X_titanic.iloc[test_index]
        y_train = y_titanic.iloc[train_index]
        y_test = y_titanic.iloc[test_index]
        score_item = fit_predict(dt_clf, X_train, y_train, X_test, y_test)
        score.append(score_item)
        n_iter += 1
        print(n_iter, ' : ', score_item)
    print(np.mean(score))
    scores = cross_val_score(dt_clf, X_titanic, y_titanic, cv=5)
    print(scores)
    print(np.mean(scores))
    parameters ={'max_depth':[1,2,3,5,10], 'min_samples_split':[2,3,5]}
    grid_dtree = GridSearchCV(dt_clf, param_grid=parameters, cv=5)
    grid_dtree.fit(X_train, y_train)
    print(grid_dtree.best_score_)
    best_esti = grid_dtree.best_estimator_
    pred = best_esti.predict(X_test)
    print(accuracy_score(y_test, pred))
    x_miss = np.array([[1,2,3,None], [5, np.nan, 7, 8], [None, 10, 11, 12], [13, np.nan, 15, 16]])
    print(x_miss)
    im = SimpleImputer(strategy='mean')
    print(im.fit_transform(x_miss)) #column 평균으로 대체
    im = SimpleImputer(strategy='median')
    print(im.fit_transform(x_miss))
    im = SimpleImputer(strategy='most_frequent')
    print(im.fit_transform([[7,2,np.nan], [4, np.nan, 6], [10, 5, 9], [3,5,9]]))
