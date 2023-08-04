#데이터 교육 복습 19

#78. ml 연습 9
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def train_ml_19():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification, load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    x, y = make_classification(n_samples=500, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=156)
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_scaled, y_train)
    train_lda = clf.transform(train_scaled)
    test_lda = clf.transform(test_scaled)
    print(train_lda.shape, test_lda.shape)

    slope = clf.coef_[0, 1]/clf.coef_[0, 0]
    print(slope)
    t = np.arange(np.min(x[:, 0]), np.max(x[:, 0]), 0.1)

    label0 = y==0
    label1 = y==1
    # plt.subplot(1,3,1)
    plt.plot(t, slope*t, color='red')
    plt.scatter(x[label0,0], x[label0,1])
    plt.scatter(x[label1,0], x[label1,1])
    plt.xlim(np.min(x[:,0])-0.1, np.max(x[:,0])+0.1)
    plt.ylim(np.min(x[:,1])-0.1, np.max(x[:,1])+0.1)
    plt.xlabel('feature #1')
    plt.ylabel('feature #2')
    plt.legend(labels=['w', 'class 0', 'class 1'])
    plt.title('before transform')

    plt.figure()
    plt.hist(train_lda[y_train==0], 100)
    plt.hist(train_lda[y_train==1], 100)
    plt.title('train data')

    plt.figure()
    plt.hist(test_lda[y_test==0], 100)
    plt.hist(test_lda[y_test==1], 100)
    plt.title('test data')
    plt.show()

    from numpy.linalg import svd

    np.random.seed(121)
    a = np.random.randn(4,4)
    print(np.round(a, 3))
    print(a.shape)
    U, Sigma, Vt = svd(a)
    print(U.shape, Sigma.shape, Vt.shape)
    print('U matrix : \n', np.round(U, 3))
    print('Sigma matrix : \n', np.round(Sigma, 3))
    print('Vt matrix : \n', np.round(Vt, 3))
    Sigma_mat = np.diag(Sigma)
    print(Sigma_mat)
    a_ = np.dot(np.dot(U, Sigma_mat), Vt)
    print(np.round(a_, 3))
    print(np.round(a, 3))

    a[2] = a[0] + a[1]
    a[3] = a[0]
    print(np.round(a, 3))
    U, Sigma, Vt = svd(a)
    print(U.shape, Sigma.shape, Vt.shape)
    print('U matrix : \n', np.round(U, 3))
    print('Sigma matrix : \n', np.round(Sigma, 3))
    print('Vt matrix : \n', np.round(Vt, 3))
    Sigma_mat = np.diag(Sigma)
    a_ = np.dot(np.dot(U, Sigma_mat), Vt)
    print(np.round(a_, 3))
    print(np.round(a, 3))

    U_ = U[:, :2]
    Sigma_ = np.diag(Sigma[:2])
    Vt_ = Vt[:2]
    print(U_.shape, Sigma_.shape, Vt_.shape)
    a_ = np.dot(np.dot(U_, Sigma_), Vt_)

    from scipy.sparse.linalg import svds

    np.random.seed(121)
    matrix = np.random.random((6, 6))
    print(matrix)
    U, Sigma, Vt = svd(matrix, full_matrices=False)
    print(Sigma)
    U_tr, Sigma_tr, Vt_tr = svds(matrix, k=4)
    print(Sigma_tr)
    matrix_tr = np.dot(np.dot(U_tr, np.diag(Sigma_tr)), Vt_tr)
    print(matrix_tr)
    print(matrix)

    from sklearn.decomposition import TruncatedSVD

    iris = load_iris()
    tsvd = TruncatedSVD(n_components=2)
    tsvd.fit(iris.data)
    iris_tsvd = tsvd.transform(iris.data)
    plt.scatter(x=iris_tsvd[:, 0], y=iris_tsvd[:, 1], c=iris.target)
    plt.show()

    irisDF = pd.DataFram(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(irisDF)
    print(kmeans.labels_)
    irisDF['target'] = iris.target
    irisDF['cluster'] = kmeans.labels_
    print(irisDF)

    iris_result = irisDF.groupby(['target', 'cluster']).count()
    print(iris_result)

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
    print(X.shape, y.shape)
    unique, counts = np.unique(y, return_counts=True)
    print(unique)
    print(counts)

    clusterDF = pd.DataFrame(data=X, columns=['ft1', 'ft2'])
    clusterDF['target'] = y
    print(clusterDF)

    markers = ['o', 's', '^']

    for un in unique:
        target_cluster = clusterDF[clusterDF.target==un]
        plt.scatter(x=target_cluster.ft1, y=target_cluster.ft2, edgecolors='k', marker=markers[un])
    plt.show()

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    clusterDF['kmeans_label'] = cluster_labels
    print(clusterDF)
    centers = kmeans.cluster_centers_
    print(centers)
    print(cluster_labels)
    unique_labels= cluster_labels

    for un in unique_labels:
        target_cluster = clusterDF[clusterDF.kmeans_label==un]
        center_x_y = centers[un]
        plt.scatter(x=target_cluster.ft1, y=target_cluster.ft2, edgecolors='k', marker=markers[un])
        plt.scatter(x=center_x_y[0], y=center_x_y[1], s=400, color='white', alpha=0.9, edgecolors='k', marker=markers[un])
        plt.scatter(x=center_x_y[0], y=center_x_y[1], s=150, color='k', edgecolors='k', marker='$%d$'%un)
    plt.show()


def visualize_silhouette(cluster_lists, X_features): 
    
        from sklearn.datasets import make_blobs
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples, silhouette_score

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import math

        n_cols = len(cluster_lists)

        fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)

        for ind, n_cluster in enumerate(cluster_lists):
            clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
            cluster_labels = clusterer.fit_predict(X_features)
            
            sil_avg = silhouette_score(X_features, cluster_labels)
            sil_values = silhouette_samples(X_features, cluster_labels)
            y_lower = 10 # 그려지는 사각형의 맨 밑에 시작 y값이 10부터 시작하게 한다. 
            axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                            'Silhouette Score :' + str(round(sil_avg,3)) )
            axs[ind].set_xlabel("The silhouette coefficient values")
            axs[ind].set_ylabel("Cluster label")
            axs[ind].set_xlim([-0.1, 1])
            axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10]) # 150 + (([2,3,4,5]+1) *10)
            axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
            axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            for i in range(n_cluster): #i=2, n_clusterr=3 이면
                ith_cluster_sil_values = sil_values[cluster_labels==i] # 군집 2 인 각 데이터들의 실루엣 값
                ith_cluster_sil_values.sort() #실루엣값 작은거부터 큰거로 정렬
                
                size_cluster_i = ith_cluster_sil_values.shape[0] # 군집2 인 데이터의 수
                y_upper = y_lower + size_cluster_i # 군집 1의 꼭대기 y값 + 10 + 군집 2의 데이터의 개수
                
                #np.arange(y_lower, y_upper)로 만들어진 숫자 개수는 군집 2의 데이터 개수와 똑같게 된다. 
                color = cm.nipy_spectral(float(i) / n_cluster)
                axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                    facecolor=color, edgecolor=color, alpha=0.7)
                axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10 # 군집 1의 맨 상위값 + 10
                
            axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
            plt.show()


def train_ml_20():
    from sklearn.datasets import load_iris

    iris = load_iris()
    irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)

    from sklearn.cluster import KMeans

    def set_kmeans(model, X_data):
        model.fit(X_data)
        labels = model.labels_
        return labels
    
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
    labels = set_kmeans(kmeans, irisDF)
    print(labels)

    from sklearn.metrics import silhouette_samples, silhouette_score

    X_features = irisDF.iloc[:, :4]
    print(X_features.head())
    X_features['cluster'] = labels
    print(X_features.head())

    score_samples = silhouette_samples(iris.data, X_features.cluster)
    print(score_samples.shape)
    X_features['silhouette_samples'] = score_samples
    print(X_features.head())

    print(silhouette_score(iris.data, X_features.cluster))

    visualize_silhouette([2,3,4,5], iris.data)

    from sklearn.cluster import DBSCAN

    def set_dbscan(X_data, eps=None, samples=None):
        if eps != None:
            dbscan = DBSCAN(eps=eps, min_samples=samples, metric='euclidean')
            labels = dbscan.fit_predict(X_data)
        else:
            dbscan = DBSCAN()
            labels = dbscan.fit_predict(X_data)
        return labels
    
    dbscan_labels = set_dbscan(iris.data, 0.6, 8)
    irisDF['dbscan_cluster'] = dbscan_labels
    print(irisDF.head())
    irisDF['target'] = iris.target
    print(irisDF.groupby(['target'])['dbscan_cluster'].value_counts())
    dbscan_labels = set_dbscan(iris.data, 0.8, 8)
    irisDF['dbscan_cluster_0.8'] = dbscan_labels
    print(irisDF.head())
    print(irisDF.groupby(['target'])['dbscan_cluster_0.8'].value_counts())
    dbscan_labels = set_dbscan(iris.data, 0.6, 16)
    irisDF['dbscan_cluster_sam_16'] = dbscan_labels
    print(irisDF.groupby(['target'])['dbscan_cluster_sam_16'].value_counts())

    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, shuffle=True, noise=0.05, factor=0.5, random_state=0)
    clusterDF = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
    clusterDF['target'] = y
    print(clusterDF.head())
    dbscan_labels = set_dbscan(X, 0.2, 10)
    clusterDF['dbscan_cluster'] = dbscan_labels
    print(clusterDF.head())

    def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
        if iscenter :
            centers = clusterobj.cluster_centers_
            
        unique_labels = np.unique(dataframe[label_name].values)
        markers=['o', 's', '^', 'x', '*']
        isNoise=False

        for label in unique_labels:
            label_cluster = dataframe[dataframe[label_name]==label]
            if label == -1:
                cluster_legend = 'Noise'
                isNoise=True
            else :
                cluster_legend = 'Cluster '+str(label)
            
            plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70,\
                        edgecolor='k', marker=markers[label], label=cluster_legend)
            
            if iscenter:
                center_x_y = centers[label]
                plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                            alpha=0.9, edgecolor='k', marker=markers[label])
                plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',\
                            edgecolor='k', marker='$%d$' % label)
        if isNoise:
            legend_loc='upper center'
        else: legend_loc='upper right'
        
        plt.legend(loc=legend_loc)
        plt.show()

    dbscan = DBSCAN(eps=0.2, min_samples=10, metric='euclidean')
    visualize_cluster_plot(dbscan, clusterDF, 'dbscan_cluster', iscenter=False)
    kmeans = KMeans(n_cluster=2, max_iter=1000, random_state=0)
    kmeans_labels = kmeans.fit_predict(X)
    clusterDF['kmeans_cluster'] = kmeans_labels
    visualize_cluster_plot(kmeans, clusterDF, 'kmeans_cluster', iscenter=False)

    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    def moon_scatter(X, y_pred):
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, edgecolors='k')
        plt.show()

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    moon_scatter(X, y_pred)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_pred = set_dbscan(X_scaled)
    moon_scatter(X, y_pred)

    y_pred = set_dbscan(X, 0.2, 5)
    moon_scatter(X, y_pred)


def train_ml_21():

    retail_df = pd.read_excel('C:/Users/eleun/Downloads/Online_Retail.xlsx')
    print(retail_df.head())
    print(retail_df.info())
    print(retail_df.describe())
    retail_df = retail_df[retail_df.Quantity>0]
    retail_df = retail_df[retail_df.UnitPrice>0]
    retail_df = retail_df[retail_df.CustomerID.notnull()]
    print(retail_df.Country.value_counts())
    retail_df = retail_df[retail_df.Country=='United Kingdom']
    print(retail_df.info())
    retail_df['sale_amount'] = retail_df.Quantity*retail_df.UnitPrice
    print(retail_df.info())
    retail_df['CustomerID'] = retail_df.CustomerID.astype(int)
    print(retail_df.CustomerID)
    print(retail_df.groupby(['InvoiceNo', 'StockCode'])['InvoiceNo'].count())
    aggregtions = {'InvoiceDate':'max',
                   'InvoiceNo':'count',
                   'sale_amount':'sum'}
    cust_df = retail_df.groupby('CustomerID').agg(aggregtions)
    print(cust_df)
    cust_df = cust_df.rename(columns={'InvoiceDate':'Recency',
                                      'InvoiceNo':'Frequency',
                                      'sale_amount':'Monetary'})
    cust_df = cust_df.reset_index()
    print(cust_df.head(3))
    print(cust_df.info())

    import datetime as dt

    cust_df['Recency'] = dt.datetime(2011,12,10) - cust_df['Recency']
    print(cust_df)
    cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days+1)
    print(cust_df.head())
    print(cust_df.sort_values('Recency'))

    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
    ax1.set_title('Recency Histogram')
    ax1.hist(cust_df['Recency'])

    ax2.set_title('Frequency Histogram')
    ax2.hist(cust_df['Frequency'])

    ax3.set_title('Monetary Histogram')
    ax3.hist(cust_df['Monetary'])
    plt.show()

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    X_features = cust_df[['Recency', 'Frequency', 'Monetary']].values
    X_features_scaled = StandardScaler().fit_transform(X_features)
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(X_features_scaled)
    cust_df['cluster_labels'] = labels

    print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(X_features_scaled,labels)))

    visualize_silhouette([2,3,4,5], X_features_scaled)

    cust_df['Recency_log'] = np.log1p(cust_df['Recency'])
    cust_df['Frequency_log'] = np.log1p(cust_df['Frequency'])
    cust_df['Monetary_log'] = np.log1p(cust_df['Monetary'])

    X_features = cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']].values
    X_features_scaled = StandardScaler().fit_transform(X_features)

    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(X_features_scaled)
    cust_df['cluster_label'] = labels

    print('실루엣 스코어는 : {0:.3f}'.format(silhouette_score(X_features_scaled,labels)))

    visualize_silhouette([2,3,4,5], X_features_scaled)
