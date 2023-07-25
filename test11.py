# 데이터 교육 복습 11

#69 marchine learning 연습 2
def train_sklearn_ml_3():
    from sklearn.base import BaseEstimator
    class MyDummyClassifier(BaseEstimator):
        def fit(self, X, y=None):
            pass
        def predict(self, X):
            pred = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                if X['Sex'].iloc[i] == 1:
                    pred[i] = 0
                else:
                    pred[i] = 1
            return pred
    titanic = pd.read_csv('C:/Users/eleun/Downloads/titanic_train.csv')
    print(titanic.info())
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler

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
        #df = adj_scale(df)
        return df
    
    y_t = titanic['Survived']
    X_t = titanic.drop('Survived', axis=1)
    X_tc = transform_features(X_t)
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X_tc, y_t)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    m_clf = MyDummyClassifier()
    m_clf.fit(X_train, y_train)
    pred = m_clf.predict(X_test)
    print(accuracy_score(y_test, pred))
    
    from sklearn.preprocessing import Binarizer

    x = [[1, -1, 2], [2, 0 , 0], [0, 1.1, 1.5]]
    print(x, type(x))
    binar = Binarizer(threshold=1.1)
    print(binar.fit_transform(x))

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    print(confusion_matrix(y_test, pred))

    def get_clf_eval(y_test, pred):
        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        print(confusion)
        print(f'정확도 :  {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}')

    print(get_clf_eval(y_test, pred))

    from sklearn.linear_model import LogisticRegression

    lr_clf = LogisticRegression(max_iter=5000)
    lr_clf.fit(X_train, y_train)
    pred = lr_clf.predict(X_test)
    get_clf_eval(y_test, pred)

    pred_proba = lr_clf.predict_proba(X_test)
    print(pred_proba[:3])
    print(pred[:3])
    print(pred_proba.shape, pred.shape)
    result = np.concatenate([pred_proba[:3], pred[:3, np.newaxis]], axis=1)
    print(result)
    binar = Binarizer(threshold=0.9)
    y_binar = binar.fit_transform(pred_proba)
    print(y_binar[:, 1])
    print(y_binar.shape, y_test.shape)
    print(get_clf_eval(y_test, pred)) # threshold = LogisticRegression에서 설정한 값
    print(get_clf_eval(y_test, y_binar[:, 1]))

    def get_eval_binar(thresholds, y_tests, pred_probas):
        binar = Binarizer(threshold=thresholds)
        y_binar = binar.fit_transform(pred_probas)
        print(f'### : {thresholds}')
        get_clf_eval(y_tests, y_binar[:, 1])

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    for i in thresholds:
        get_eval_binar(i, y_test, pred_proba)
    
    print(pred_proba.shape)

    from sklearn.metrics import precision_recall_curve

    pred_proba = lr_clf.predict_proba(X_test)[:, 1]
    print(pred_proba.shape)

    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba)
    sample_index = np.arange(0, thresholds.shape[0], 15)
    print(sample_index)
    print(thresholds[sample_index])
    print(precisions[sample_index])
    print(recalls[sample_index])

    def pre_reca_plot(precisions, recalls, thresholds):
        fig,axes = plt.subplots()
        axes.plot(thresholds, precisions, 'k--', label='precisions')
        axes.plot(thresholds, recalls, 'g--', label='recalls')
        axes.legend()
        axes.set_xlabel('Threshold')
        axes.set_ylabel('Precision & Recall')
        axes.grid()
        plt.show()

    pre_reca_plot(precisions[:thresholds.shape[0]], recalls[:thresholds.shape[0]], thresholds[:thresholds.shape[0]])

    def precision_recall_curve_plot(y_test, pred_proba_c1):
        # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
        precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
        
        # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행, 정밀도는 점선으로 표시
        plt.figure(figsize=(8, 6))
        threshold_boundary = thresholds.shape[0]
        plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
        plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')
        
        # threshold 값 X축의 scale을 0.1 단위로 변경
        start, end = plt.xlim()
        plt.xticks(np.round(np.arange(start, end, 0.1), 2))
        
        # X, Y축 label과 legend, grid 설정
        plt.xlabel('Threshold value')
        plt.ylabel('Precision and Recall value')
        plt.legend()
        plt.grid()
        plt.show()

    precision_recall_curve_plot(y_test, pred_proba)

    from sklearn.metrics import f1_score

    f1 = f1_score(y_test, pred)
    print(f1)

    def get_clf_eval(y_test, pred):
        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        print(confusion)
        print(f'정확도 :  {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, f1_score: {f1:.4f}')

    def get_eval_binar(thresholds, y_tests, pred_probas):
        binar = Binarizer(threshold=thresholds)
        y_binar = binar.fit_transform(pred_probas)
        print(f'### : {thresholds}')
        get_clf_eval(y_tests, y_binar[:, 1])
    
    pred_proba = lr_clf.predict_proba(X_test)
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    for i in thresholds:
        get_eval_binar(i, y_test, pred_proba)

    from sklearn.metrics import roc_curve

    pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    print(fprs)
    print(tprs)
    print(thresholds)

    def roc_curve_plot(y_test, pred_proba_c1):
        fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
        plt.figure(figsize=(6,6))
        plt.plot(fprs, tprs, label='ROC')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('FPR', fontsize=25)
        plt.ylabel('TPR', fontsize=25)
        plt.title('ROC Curve', fontsize=25)
        plt.tick_params(length=8, labelsize=18)
        plt.tight_layout()
        plt.grid()
        plt.legend(loc='lower right')

    roc_curve_plot(y_test, pred_proba_c1)

    from sklearn.metrics import roc_auc_score

    r_auc_s = roc_auc_score(y_test, pred_proba_c1)

    r_auc_s

    def get_clf_eval(y_test, pred=None, pred_probas=None):
        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc_score = roc_auc_score(y_test, pred_probas)
        print(confusion)
        print(f'정확도 :  {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, f1_score: {f1:.4f}, AUC : {auc_score:.4f}')

    get_clf_eval(y_test, pred, pred_proba_c1)

    def get_eval_binar(thresholds, y_tests, pred_probas):
        binar = Binarizer(threshold=thresholds)
        y_binar = binar.fit_transform(pred_probas)
        print(f'### : {thresholds}')
        get_clf_eval(y_tests, y_binar[:, 1], pred_probas[:, 1])

    pred_proba = lr_clf.predict_proba(X_test)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    for i in thresholds:
        get_eval_binar(i, y_test, pred_proba)

def train_sklearn_ml_4():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    def get_clf_eval(y_test, pred=None, pred_probas=None):
        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc_score = roc_auc_score(y_test, pred_probas)
        print(confusion)
        print(f'정확도 :  {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, f1_score: {f1:.4f}, AUC : {auc_score:.4f}')


    diabetes = pd.read_csv('C:/Users/eleun/Downloads/diabetes.csv')
    print(diabetes.info())
    print(diabetes.head())

    X = diabetes.iloc[:, :-1]
    y = diabetes.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    pred_proba = dt_clf.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_proba)

    lr_clf = LogisticRegression(max_iter=5000)
    lr_clf.fit(X_train, y_train)
    pred = lr_clf.predict(X_test)
    pred_proba = lr_clf.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_proba)

    print(diabetes.describe())

    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    plt.hist(diabetes['Glucose'], bins=10)
    plt.show()

    zero_count = diabetes[diabetes['Glucose']==0]['Glucose'].count()
    print(zero_count)

    for feature in zero_features:
        zero_count = diabetes[diabetes[feature]==0][feature].count()
        print(feature, zero_count, zero_count/768*100)
    
    mean_zero = diabetes[zero_features].mean()
    print(mean_zero)

    diabetes[zero_features] = diabetes[zero_features].replace(0, mean_zero)

    for feature in zero_features:
        zero_count = diabetes[diabetes[feature]==0][feature].count()
        print(feature, zero_count, zero_count/768*100)

    print(diabetes.describe())

    print(y.value_counts())

    X = diabetes.iloc[:, :-1]
    y = diabetes.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    pred_proba = dt_clf.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_proba)

    scaler = MinMaxScaler()
    scaler.fit(X)
    scaled_array = scaler.transform(X)
    m_scaler = pd.DataFrame(data=scaled_array, columns=X.columns)

    scaler = StandardScaler()
    scaler.fit(X)
    scaled_array = scaler.transform(X)
    s_scaler = pd.DataFrame(data=scaled_array, columns=X.columns)

    dt_clf = DecisionTreeClassifier()
    lr_clf = LogisticRegression(max_iter=5000)

    X_train, X_test, y_train, y_test = train_test_split(m_scaler, y, test_size=0.2, random_state=123, stratify=y)
    pred = dt_clf.predict(X_test)
    pred_proba = dt_clf.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_proba)
    lr_clf.fit(X_train, y_train)
    pred = lr_clf.predict(X_test)
    pred_proba = lr_clf.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_proba)

    X_train, X_test, y_train, y_test = train_test_split(s_scaler, y, test_size=0.2, random_state=123, stratify=y)
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    pred_proba = dt_clf.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_proba)
    lr_clf.fit(X_train, y_train)
    pred = lr_clf.predict(X_test)
    pred_proba = lr_clf.predict_proba(X_test)[:, 1]
    get_clf_eval(y_test, pred, pred_proba)
