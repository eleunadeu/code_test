# 데이터 교육 복습 9

#67. sklearn 활용 연습
from sklearn import linear_model
import numpy as np

def model_build(model, data):
  '''
  model: linear_model.LinearRegression()
  data: 모델에 fit할 데이터(x값과 y값이 있어야 함)
  '''
  col_list =[]
  for i in data.columns:
    col_list.append(i)
  X= data[col_list[0]].values[:,np.newaxis]
  y= data[col_list[1]]
  model.fit(X, y)
  print('b value =', model.intercept_)
  print('a value =', model.coef_)
  pred = model.predict(X)
  print(pred)
  return model

def train_sklearn():
    data = {'x':[13,19,16,14,15,18], 'y':[40,83,62,57,58,63]}
    data = pd.DataFrame(data)
    print(data)
    data.plot(kind='scatter', x='x', y='y', figsize=(5,5), color='blue')
    plt.show()
    lr = linear_model.LinearRegression()
    lr_ = model_build(lr, data)
    plt.figure(figsize=(10, 8))
    plt.scatter(data.x, data.y, s=125)
    plt.plot(data.x, prediction, color='black')
    plt.vlines(14, 50.7, 57, colors='green', linestyle='--')
    plt.vlines(18, 63, 72.09, colors='green', linestyle='--')
    plt.text(13.7, 53, r'$e_1$', size=20, color='red')
    plt.text(17.7, 68, r'$e_2$', size=20, color='red')
    plt.text(18, 78, r'$y=a_1 x + a_0$',size=15, color='blue')
    plt.ylabel('y', size=20)
    plt.xlabel('x', size=20)
    plt.text(13.7, 58, r'$(x_1, y_1)$', size=15, color='blue')
    plt.text(17.7, 61.5, r'$(x_2, y_2)$', size=15, color='blue')
    plt.show()
    residuals = data['y'] - pred
    for i in range(len(residuals)):
        print(data['y'][i], '-', np.round(pred[i],2), '=', np.round(residuals[i],2))
    x = data.x.values
    print(x)
    y = data.y.values
    print(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_square = np.sum(x*x)
    sum_xy = np.sum(x*y)
    n = data.x.count()
    a_1 = (n*sum_xy - sum_x*sum_y)/(n*sum_x_square - sum_x*sum_x)
    print(a_1)
    a_0 = np.mean(y) - np.mean(x)*a_1
    print(a_0)
    print(np.polyfit(x, y, 1))

#68. sklearn data set, model 및 기능 활용 연습
def train_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(type(iris))
    keys = iris.keys()
    print(keys)
    print(iris.target)
    print(iris.target.shape)
    print(iris.target_names)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    iris_data = iris.data
    iris_label = iris.target
    print('iris target값:', iris_label)
    print('iris target명:', iris.target_names)
    iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
    iris_df['label'] = iris.target
    print(iris_df.head(3))
    iris_df.label.value_counts()
    X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)
    print(X_train.shape)
    print(y_train.shape)
    dt_clf = DecisionTreeClassifier(random_state=11)
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    print(y_test, pred)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, pred))

def train_iris2():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    iris= load_iris()
    X_train = iris.data
    y_train = iris.target
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_train)
    print(accuracy_score(y_train, pred))
    print(iris.keys())
    print(iris['DESCR'])
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    print(df.shape)
    print(df.head())
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    print(df.head())
    df['target'] = iris.target
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.loc[df.duplicated(),:])
    print(df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9),:])
    df = df.drop_duplicates()
    print(df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9),:])
    sns.heatmap(df.corr(), square=True, annot=True, cbar=True)
    plt.show()
    print(df['target'].value_counts())
    plt.hist(x='sepal_length', data=df)
    plt.show()
    sns.displot(x='sepal_width', kind='hist', data=df)
    plt.show()
    sns.displot(x='petal_width', kind='kde', data=df)
    plt.show()
    sns.displot(x='sepal_length', hue='target', kind='kde', data=df)
    plt.show()
    for col in ['sepal_width', 'petal_length', 'petal_width']:
        sns.displot(x=col, hue='target', kind='kde', data=df)
    plt.show()
    sns.pairplot(df, hue='target', size=2.5, diag_kind='kde')
    plt.show()
    print(X_train[0])
    X_test = X_train[0]
    print(dt_clf.predict(X_test.reshape(1,4)))

def train_titanic():
    titanic_df = pd.read_csv('C:/Users/eleun/Downloads/titanic_train.csv')
    print(titanic_df.head())
    print(titanic_df.info())
    titanic_df.Age.fillna(titanic_df.Age.mean(), inplace=True)
    titanic_df.Cabin.fillna('N', inplace=True)
    titanic_df.Embarked.fillna('N', inplace=True)
    print(titanic_df.isnull().sum())
    print(titanic_df.Cabin.value_counts())
    print(titanic_df.Embarked.value_counts())
    print(titanic_df.Sex.value_counts())
    titanic_df['Cabin'] = titanic_df.Cabin.str[:1]
    print(titanic_df.head())
    sns.barplot(x='Sex', y='Survived', data=titanic_df)
    plt.show()
    print(titanic_df.groupby(['Sex', 'Survived']).Name.count())
    sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
    plt.show()
    print(titanic_df.Age)
    def get_category(age):
        cat = ''
        if age <= -1: cat = 'Unknown'
        elif age <= 5: cat = 'Baby'
        elif age <= 12: cat = 'Child'
        elif age <= 18: cat = 'Teenager'
        elif age <= 25: cat = 'Student'
        elif age <= 35: cat = 'Young Adult'
        elif age <= 60: cat = 'Adult'
        else: cat = 'Elderly'
        return cat
    age_cat_df = titanic_df.Age.apply(lambda x: get_category(x))
    print(age_cat_df)
    titanic_df['Age_cat'] = titanic_df.Age.apply(lambda x: get_category(x))
    print(titanic_df.head())
    plt.figure(figsize=(10,6))
    sns.barplot('Age_cat', 'Survived', hue='Sex', data=titanic_df)
    plt.show()
    titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Age_cat'], axis=1, inplace=True)
    print(titanic_df.head())
    from sklearn import preprocessing
    a_train = np.array(['pc', 'mobile', 'pc'])
    encoder = preprocessing.LabelEncoder()
    encoder.fit(a_train)
    print(encoder.transform(a_train))
    def encode_features(dataDF):
        features = ['Cabin', 'Sex', 'Embarked']
        for feature in features:
            le = preprocessing.LabelEncoder()
            le = le.fit(dataDF[feature])
            dataDF[feature] = le.transform(dataDF[feature])
        return dataDF
    titanic_df = encode_features(titanic_df)
    print(titanic_df)
    y_titanic_df = titanic_df['Survived']
    X_titanic_df = titanic_df.drop('Survived', axis=1)
    print(y_titanic_df.head())
    print(X_titanic_df.head())
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=156)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    dt_pred = dt_clf.predict(X_test)
    print(accuracy_score(y_test, dt_pred))
    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    print(accuracy_score(y_test, rf_pred))
    from sklearn.linear_model import LogisticRegression
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    print(accuracy_score(y_test, lr_pred))

def train_kfold():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = iris.data
    label = iris.target
    dt_clf = DecisionTreeClassifier(random_state=156)
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(features):
        print('train : ', train_index)
        print('test : ', test_index)
        print()
    print(train_index)
    print(features[train_index].shape)
    print(label[train_index].shape)
    for train_index, test_index in kfold.split(features):
        dt_clf.fit(features[train_index], label[train_index])
        dt_pred = dt_clf.predict(features[test_index])
        print('정확도 : ', accuracy_score(label[test_index], dt_pred))

    iris_df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    print(iris_df)
    iris_df['label'] = iris.target
    print(iris['label'].value_counts())
    kfold = KFold(n_splits=3)
    n_iter= 0
    for train_index, test_index in kfold.split(iris_df):
        n_iter += 1
        label_train = iris_df['label'].iloc[train_index]
        label_test = iris_df['label'].iloc[test_index]
        print('## : ',n_iter)
        print(label_train.value_counts())
        print(label_test.value_counts())
        print('===========')

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3)
    n_iter = 0
    for train_index, test_index in kfold.split(iris_df):
        n_iter += 1
        label_train = iris_df['label'].iloc[train_index]
        label_test = iris_df['label'].iloc[test_index]
        print('## : ',n_iter)
        print(label_train.value_counts())
        print(label_test.value_counts())
        print('===========')

    dt_clf = DecisionTreeClassifier(random_state=156)

    skfold = StratifiedKFold(n_splits=3)

    n_iter = 0
    cv_accuracy = []

    for train_index, test_index in skfold.split(features, label):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = label[train_index], label[test_index]

        dt_clf.fit(X_train, y_train)

        pred = dt_clf.predict(X_test)

        accuracy = np.round(accuracy_score(y_test, pred), 4)
        cv_accuracy.append(accuracy)
    
    print(cv_accuracy)
    print(np.mean(cv_accuracy))

def train_cvs():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    dt_clf = DecisionTreeClassifier(random_state=156)

    data = iris_data.data
    label = iris_data.target

    scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)

    print(np.round(scores, 4))
    print(np.mean(scores))

    iris_data = load_iris()
    dt_clf = DecisionTreeClassifier()

    data = iris_data.data
    label = iris_data.target

    scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)

    print(np.round(scores, 4))
    print(np.mean(scores))
