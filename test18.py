# 데이터 교육 복습 18

#77. ml 연습 8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def train_ml_16():
    bike_df = pd.read_csv('C:/Users/eleun/Downloads/bike_train.csv')
    bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime)
    bike_df['year'] = bike_df.datetime.apply(lambda x: x.year)
    bike_df['month'] = bike_df.datetime.apply(lambda x: x.month)
    bike_df['day'] = bike_df.datetime.apply(lambda x: x.day)
    bike_df['hour'] = bike_df.datetime.apply(lambda x: x.hour)
    drop_columns = ['datetime', 'casual', 'registered']
    bike_df.drop(drop_columns, axis=1, inplace=True)

    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    def evaluate_regr(y, pred):
        rmse_val = np.sqrt(mean_squared_error(y, pred))
        mae_val = mean_absolute_error(y, pred)
        print(f'RMSE : {rmse_val:.3F}, MAE : {mae_val:.3F}')

    y_target = bike_df['count']
    X_features = bike_df.drop(['count'], axis=1, inplace=False)
    y_target_log = np.log1p(y_target)

    print(X_features.head())
    X_features_ohe = pd.get_dummies(X_features, columns=['year', 'month', 'day', 'hour', 'holiday', 'workingday', 'season', 'weather'])
    print(X_features_ohe)

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import cross_val_score
    
    X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target, test_size=0.3, random_state=0)

    def model_prediction_eval(models, X_train, X_test, y_train, y_test):
        for model in models:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            print(model.__class__.__name__)
            evaluate_regr(y_test, pred)
            print('___________________________')
    
    lr_reg = LinearRegression()
    ri_reg = Ridge()
    la_reg = Lasso()
    dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
    rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
    gb_reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
    xgb_reg = XGBRegressor(random_state=0, n_estimators=1000)
    lgbm_reg = LGBMRegressor(random_state=0, n_estimators=1000)
    models = [lr_reg, ri_reg, la_reg, dt_reg, rf_reg, gb_reg, xgb_reg, lgbm_reg]

    model_prediction_eval(models, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.3, random_state=0)
    def model_predict_eval(models, X_train, X_test, y_train, y_test):
        for model in models:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            y_test = np.expm1(y_test)
            pred = np.expm1(X_test)
            print(model.__class__.__name__)
            evaluate_regr(y_test, pred)
            print('___________________________')

    model_predict_eval(models, X_train, X_test, y_train, y_test)

    df = pd.read_csv('C:/Users/eleun/Downloads/California_Houses.csv')
    print(df.head())
    print(df.info())
    print(df.isna().sum().sum())
    target = df.Median_House_Value
    data = df.iloc[:, 1:]
    print(data)
    print(target)

    def Regression_process(x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        lr = LinearRegression().fit(X_train, y_target)
        print('R_square (train set, test set)', np.round(lr.score(X_train, y_train),4), np.round(lr.score(X_test, y_test), 4))
    print(data.head())
    
    x = data.drop(['Latitude', 'Longitude'], axis=1)
    y = target
    print(x.columns)
    Regression_process(x, y)

    def reg_graph_plot(y):
        from scipy.stats import norm
        from scipy import stats
        import matplotlib.style as style

        style.use('ggplot')
        sns.barplot(y, fit=norm)
        plt.show()

        fig, ax = plt.subplots()
        stats.probplot(y, plot=ax)
        plt.show()

        sns.boxplot(y)

    reg_graph_plot(y)

    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    outlier = []
    for i in y:
        if i < (q1 - 1.5*iqr) or i>(q3 + 1.5*iqr):
            outlier.append(i)

    print(np.round(len(outlier)/len(df)*100, 2))
    o_low = q1-1.5*iqr
    o_high = q3+1.5*iqr
    x=x[(y > o_low) & (y < o_high)]
    y=y[(y > o_low) & (y < o_high)]

    print(x.shape, y.shape)

    reg_graph_plot(y)
    Regression_process(x, y)
    print(x.columns)

    fig, axes = plt.subplots(figsize=(16, 8), ncols=6, nrows=2)
    for i, feature in enumerate(x.columns):
        row = int(i/6)
        col = i % 6
        sns.regplot(x=feature, y=y, data=x, ax=axes[row][col])

    plt.figure(figsize=(20, 12))
    for i, column in enumerate(x.columns, 1):
        plt.subplot(3, 4, i)
        sns.distplot(x=x[column],color='indianred')
        plt.legend()
        plt.xlabel(column)

    y = np.log1p(y)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x= scaler.fit_transform(x)
    Regression_process(x, y)
    
    import matplotlib.style as style
    style.use('ggplot')
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from scipy.stats import norm
    from scipy import stats

    sns.set_style("whitegrid")

    def visualize_target(y):
        print('Feature: {}, Skewness: {}, Kurtosis: {}'.format(y.name,round(y.skew(),5),round(y.kurt(),5)))
        
        fig = plt.figure(constrained_layout=True, figsize=(12,6))
        grid = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)

        ax1 = fig.add_subplot(grid[0:2, :4])
        ax1.set_title('Histogram')
        sns.distplot(y, norm_hist=True,fit=norm, ax = ax1,color='indianred')

        ax2 = fig.add_subplot(grid[2:, :4])
        ax2.set_title('QQ_plot')
        stats.probplot(y, plot = ax2)

        ax3 = fig.add_subplot(grid[:, 4])
        ax3.set_title('Box Plot')
        sns.boxplot(y=y, orient='v', ax = ax3,color='indianred')
        plt.show()

    visualize_target(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    lr=LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)
    plt.subplots(figsize=(8,8))
    plt.scatter(y_pred, y_test, s=4, color='indianred')
    plt.plot(y_test, y_test, color='cornflowerblue', linewidth=3)
    plt.xlim(0, 500000)
    plt.ylim(0, 500000)
    plt.show()

    X = pd.DataFrame(x, columns=['Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms',
                    'Population', 'Households', 'Distance_to_coast', 'Distance_to_LA',
                    'Distance_to_SanDiego', 'Distance_to_SanJose',
                    'Distance_to_SanFrancisco'])
    
    from sklearn.metrics import r2_score
    
    X_data = X.Median_Income.to_numpy().reshape(-1, 1)
    lr.fit(X_data, y)
    pred = lr.predict(X_data)
    print(lr.coef_, lr.intercept_)
    print(r2_score(y, pred))

    style.use('ggplot')
    sns.set_style('whitegrid')
    plt.scatter(X_data, y, c='red')
    plt.plot(X_data, pred, c='blue')
    plt.show()


def train_ml_17():
    house_df_org = pd.read_csv('C:/Users/eleun/Downloads/house_price.csv')
    house_df = house_df_org.copy()
    print(house_df.head(3))
    print(house_df.dtypes.value_counts())
    isnull_series = house_df.isnull().sum()
    print(isnull_series[isnull_series>0].sort_values(ascending=False))
    print(house_df.shape)
    sns.barplot(house_df.SalePrice)
    plt.show()
    house_df['SalePrice'] = np.log1p(house_df.SalePrice)
    sns.distplot(house_df.SalePrice)
    plt.show()
    print(house_df.head())
    house_df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
    house_df.fillna(house_df.mean(), inplace=True)
    print(house_df.isnull().sum().sum())
    ncc = house_df.isnull().sum()[house_df.isnull().sum()>0]
    print(type(ncc))
    print(ncc.index)
    print(house_df.dtypes[ncc.index])

    df_sample = pd.DataFrame({'one':[1,2,3,4], 'two':['a', np.nan, 'a', 'b']})
    print(df_sample)
    print(pd.get_dummies(df_sample))

    house_df_ohe = pd.get_dummies(house_df)
    print(house_df.shape, house_df_ohe.shape)
    print(house_df_ohe.head())
    print(house_df_ohe.isnull().sum().sum())

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    def get_rmse(model, X_test, y_test):
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        print(f'{model.__class__.__name__} 로그변환 된 RMSE : {np.round(rmse, 3)}')
        return rmse
    
    def get_rmses(models, X_test, y_test):
        rmses = []
        for model in models:
            rmse = get_rmse(model, X_test, y_test)
            rmses.append(rmse)
        return rmses
    
    y_target = house_df_ohe['SalePrice']
    X_features = house_df_ohe.drop('SalePrice', axis=1,inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=0)

    lr_reg = LinearRegression().fit(X_train, y_train)
    ri_reg = Ridge().fit(X_train, y_train)
    la_reg = Lasso().fit(X_train, y_train)
    models = [lr_reg, ri_reg, la_reg]

    get_rmses(models, X_test, y_test)

    def get_top_bottom_coef(model):
        coef = pd.Series(model.coef_, index=X_features.columns)
        
        coef_high = coef.sort_values(ascending=False).head(10)
        coef_low = coef.sort_values(ascending=False).tail(10)
        return coef_high, coef_low
    def visualize_coefficient(models):
        fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=3)
        fig.tight_layout() 
        for i_num, model in enumerate(models):
            coef_high, coef_low = get_top_bottom_coef(model)
            coef_concat = pd.concat( [coef_high , coef_low] )
            axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size=25)
            axs[i_num].tick_params(axis="y",direction="in", pad=-120)
            for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
                label.set_fontsize(22)
            sns.barplot(x=coef_concat.values, y=coef_concat.index , ax=axs[i_num])
            plt.show()

    visualize_coefficient(models)


def train_ml_18():
    from sklearn.datasets import load_iris

    iris = load_iris()
    print(iris.feature_names)
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    irisDF = pd.DataFrame(iris.data, columns=columns)
    irisDF['target'] = iris.target
    print(irisDF.head())

    markers=['^', 's', 'o']

    for i, marker in enumerate(markers):
        x_axis_data = irisDF[irisDF['target']==i]['sepal_length']
        y_axis_data = irisDF[irisDF['target']==i]['sepal_width']
        plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

    plt.legend()
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()

    from sklearn.preprocessing import StandardScaler

    iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:,:-1])

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    print(iris_scaled.shape)
    pca.fit(iris_scaled)
    iris_pca = pca.transform(iris_scaled)
    print(iris_pca.shape)
    pca_columns = ['pca_component_1', 'pca_component_2']
    irisDF_pca = pd.DataFrame(iris_pca, columns=pca_columns)
    print(irisDF_pca.head())
    print(irisDF.head())
    irisDF_pca['target'] = iris.target
    markers=['^', 's', 'o']

    for i, marker in enumerate(markers):
        x_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_1']
        y_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_2']
        plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

    plt.legend()
    plt.xlabel('pca_component_1')
    plt.ylabel('pca_component_2')
    plt.show()

    print(pca.explained_variance_ratio_)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    rfc = RandomForestClassifier(random_state=156)
    scores = cross_val_score(rfc, iris.data, iris.target, socring='accuracy', cv=3)
    print(scores)
    print(np.mean(scores))
    scores_pca = cross_val_score(rfc, irisDF_pca.iloc[:,:-1], irisDF_pca.target, scoring='accuracy', cv=3)
    print(scores_pca)
    print(np.mean(scores_pca))

    df = pd.read_excel('C:/Users/eleun/Downloads/credit_card.xls', header=1, sheet_name='Data')
    print(df.shape)
    df.drop('ID', axis=1, inplace=True)
    df.rename(columns={'PAY_0':'PAY_1', 'default payment next month':'default'}, inplace=True)
    print(df.head())
    X_features = df.iloc[:, :-1]
    y_target = df['default']
    print(X_features.info())
    
    corr = X_features.corr()
    plt.figure(figsize=(15,15))
    sns.heatmap(corr, annot=True, fmt='.1g')
    plt.show()

    cols_pay = ['PAY_'+str(i) for i in range(1,7)]
    print(cols_pay)
    scaler = StandardScaler()
    X_features_pay = scaler.fit_transform(X_features[cols_pay])
    pca = PCA(n_components=2)
    pca.fit(X_features_pay)
    print(pca.explained_variance_ratio_)

    cols_bill = ['BILL_AMT'+str(i) for i in range(1,7)]
    print(cols_bill)
    X_features_bill = scaler.fit_transform(X_features[cols_bill])
    pca.fit(X_features_bill)
    print(pca.explained_variance_ratio_)

    rfc = RandomForestClassifier(n_estimators=1000, random_state=156)
    scores = cross_val_score(rfc, X_features, y_target, scoring='accuracy', cv=3)
    pca = PCA(n_components=6)
    X_features_scaled = scaler.fit_transform(X_features)
    X_features_pca = pca.fit_transform(X_features_scaled)
    scores_pca = cross_val_score(rfc, X_features_pca, y_target, scoring='accuracy', cv=3)
    print(scores, np.mean(scores))
    print(scores_pca, np.mean(scores_pca))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    iris_scaled = StandardScaler().fit_transform(iris.data)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(iris_scaled, iris.target)
    iris_lda = lda.transform(iris_scaled)
    print(iris_lda.shape)

    lda_columns = ['lda_component_1', 'lda_component_2']
    irisDF_lda = pd.DataFrame(iris_lda, columns=lda_columns)
    irisDF_lda['target']=iris.target

    markers=['^', 's', 'o']

    for i, marker in enumerate(markers):
        x_axis_data = irisDF_lda[irisDF_lda['target']==i]['lda_component_1']
        y_axis_data = irisDF_lda[irisDF_lda['target']==i]['lda_component_2']
        plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

    plt.legend()
    plt.xlabel('lda_component_1')
    plt.ylabel('lda_component_2')
    plt.show()

    rcf = RandomForestClassifier(random_state=156)
    scores = cross_val_score(rcf, irisDF_lda[lda_columns], iris.target, scoring='accuracy', cv=3)
    print(scores, np.mean(scores))
