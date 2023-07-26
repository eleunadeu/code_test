#데이터 교육 복습 12

#70 ml 연습 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_graphviz():
    import graphviz
    import warnings
    
    warnings.filterwarnings('ignore')
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    iris_data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2)
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    
    from sklearn.tree import export_graphviz

    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)

    print(dt_clf.feature_importances_)

    sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)

    from sklearn.datasets import make_classification

    X_features, y_labels = make_classification(n_features = 2, n_classes=3, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=0)
    print(X_features.shape)
    print(y_labels)

    plt.scatter(X_features[:, 0], X_features[:, 1], c=y_labels, s=25)
    plt.show()

    dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)
    
    def visualize_boundary(model, X, y):
        fig,ax = plt.subplots()
        
        # 학습 데이타 scatter plot으로 나타내기
        ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
                clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')
        xlim_start , xlim_end = ax.get_xlim()
        ylim_start , ylim_end = ax.get_ylim()
        
        # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
        model.fit(X, y)
        # meshgrid 형태인 모든 좌표값으로 예측 수행. 
        xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # contourf() 를 이용하여 class boundary 를 visualization 수행. 
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3,
                            levels=np.arange(n_classes + 1) - 0.5,
                            cmap='rainbow', clim=(y.min(), y.max()),
                            zorder=1)
    
    visualize_boundary(dt_clf, X_features, y_labels)

    dt_clf = DecisionTreeClassifier(min_samples_leaf=6).fit(X_features, y_labels)
    visualize_boundary(dt_clf, X_features, y_labels)

    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    print(cancer.feature_names)
    print(cancer.target_names)
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    print(cancer_df)

    cancer_df['target'] = cancer.target
    print(cancer_df.head())
    print(cancer_df.target.value_counts())
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    dt_clf = DecisionTreeClassifier(random_state=0)
    dt_clf.fit(X_train, y_train)
    print(dt_clf.score(X_train, y_train))
    print(dt_clf.score(X_test, y_test))

    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X_train, y_train)
    print(tree.score(X_train, y_train))
    print(tree.score(X_test, y_test))
    print(cancer.data.shape[1])

    def plot_feature_importances(model):
        n_features = cancer.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel('feature_importance')
        plt.ylabel('feautre')
        plt.ylim(-1, n_features)
        plt.show()

    plot_feature_importances(tree)

    # export_graphviz()의 결과로 out_file로 지정된 tree.dot 파일을 생성함
    export_graphviz(tree, out_file="tree.dot", class_names=cancer.target_names, \
                feature_names = cancer.feature_names, impurity=True, filled=True)
    
    #위에서 생선된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook 상에서 시삭화
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)

def train_ml():
    from sklearn.tree import DecisionTreeClassifier

    dt_clf = DecisionTreeClassifier()

    df = pd.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']], columns=['alphabet'])
    print(df)
    print(df.groupby('alphabet').cumcount())

    df = pd.read_csv('C:/Users/eleun/Downloads/data/features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])
    dummy_df = df.groupby('column_name').count()
    print(dummy_df)
    print(dummy_df[dummy_df['column_index']>1].count())
    
    print(dt_clf.get_params())

    f_df = pd.DataFrame(df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    print(f_df.head())

    df2 = f_df.reset_index()
    df1 = df.reset_index()
    print(df1)
    print(df2)
    new_df = pd.merge(df1, df2, how='outer', on='index')
    print(new_df.head())
    new_df['new_feature_name'] = new_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) if x[1]>0 else x[0], axis=1)
    print(new_df[new_df['dup_cnt']>0])
    
    feature_name = new_df.iloc[:, 4]
    ftn = feature_name.values.tolist()
    X_train = pd.read_csv('C:/Users/eleun/Downloads/data/X_train.txt', sep='\s+', names=ftn)
    print(X_train.head())
    X_test = pd.read_csv('C:/Users/eleun/Downloads/data/X_test.txt', sep='\s+', names=ftn)
    print(X_test.head())
    print(X_test.shape, X_train.shape)
    y_train = pd.read_csv('C:/Users/eleun/Downloads/data/y_train.txt', sep='\s+', names=['action'])
    y_test = pd.read_csv('C:/Users/eleun/Downloads/data/y_test.txt', sep='\s+', names=['action'])
    print(y_train.shape, y_test.shape)
    print(y_train.value_counts())
    print(X_train.isna().sum().sum())
    print(X_test.isna().sum().sum())

    dt_clf = DecisionTreeClassifier(random_state=156)
    dt_clf.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score

    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(accuracy)
    print(dt_clf.get_params())
    
    from sklearn.model_selection import GridSearchCV

    params = {'max_depth':[6, 8, 10, 12, 16, 20, 24]}
    cvs = 5
    gdt_clf = GridSearchCV(dt_clf, param_grid=params, cv=cvs, scoring='accuracy')
    gdt_clf.fit(X_train, y_train)
    print(gdt_clf.best_params_, gdt_clf.best_score_)

    cv_results_df = pd.DataFrame(gdt_clf.cv_results_)
    print(cv_results_df[['param_max_depth', 'mean_test_score']])

    max_list = [6, 8, 10, 12, 16, 20, 24]
    for i in max_list:
        dt_clf = DecisionTreeClassifier(max_depth=i, random_state=156)
        dt_clf.fit(X_train, y_train)
        sc = dt_clf.score(X_test, y_test)
        print(sc)

    params = {'max_depth':[6, 8, 10, 12, 16, 20, 24], 'min_samples_split':[16,24]}
    gdt_clf = GridSearchCV(dt_clf, param_grid=params, cv=cvs, scoring='accuracy')
    gdt_clf.fit(X_train, y_train)
    gdt_clf.best_score_
    estimator = gdt_clf.best_estimator_
    pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(accuracy)

    ftr_importances = estimator.feature_importances_
    ftr_impo = pd.Series(ftr_importances, index=X_train.columns)
    ftr_top20 = ftr_impo.sort_values(ascending=False)[:20]
    print(ftr_top20)
    print(type(ftr_top20))

    plt.figure(figsize=(8,6))
    sns.barplot(x=ftr_top20, y=ftr_top20.index)
    plt.show()

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    cancer = load_breast_cancer()
    data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    print(data_df.head(5))

    lr_clf = LogisticRegression(max_iter=5000)
    knn_clf = KNeighborsClassifier(n_neighbors=8)
    vo_clf = VotingClassifier(estimators=[('LR', lr_clf), ('KNN', knn_clf)], voting='soft')

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    vo_clf.fit(X_train, y_train)
    pred = vo_clf.predict(X_test)
    print(accuracy_score(y_test, pred))

    classifier = [lr_clf, knn_clf]
    for classifier in classifier:
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
        print(classifier.__class__.__name__, accuracy_score(y_test, pred))
