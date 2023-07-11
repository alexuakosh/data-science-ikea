from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


def create_model(df):
    """4.1 We had better to take as many as possible columns for the model considering that
    dependency of the price with each column is not as strong and predictive"""
    x = df[
        ['category', 'old_price', 'sellable_online', 'other_colors', 'designer_type', 'depth', 'height', 'width']]
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=7)),
        # ('scaler', StandardScaler())  # There is no need in scaling using multiple linear regression?
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown="ignore"))
    ])
    column_preprocess = ColumnTransformer(transformers=[
        ('numeric', numeric_transformer, ['old_price', 'depth', 'height', 'width']),
        ('categorical', categorical_transformer, ['category', 'sellable_online', 'other_colors', 'designer_type'])
    ])

    model1 = Pipeline(steps=[
        ('preprocess', column_preprocess),
        ('reg', LinearRegression())
    ])  # Model with linear regression
    model2 = Pipeline(steps=[
        ('preprocess', column_preprocess),
        ('reg', RandomForestRegressor())
    ])  # model with random forest regression

    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)

    print('MODEL 1 PERFORMANCE:')
    predicted_price1 = model1.predict(x_test)
    print(f'Model 1 Score: {model1.score(x_test, y_test)}')  # Model score is approx. from 0.6 to 0.7
    print('Mean Squared model 1 Error: ', mean_squared_error(y_test, predicted_price1))
    scores1 = cross_val_score(model1, x, y, cv=5)
    print(f'Cross validation model 1 Scores: {scores1}')

    print('MODEL 2 PERFORMANCE:')
    predicted_price2 = model2.predict(x_test)
    print(f'Model 2 Score: {model2.score(x_test, y_test)}')  # Model score is approx. 0.8 It's better model.
    print('Mean Squared model 2 Error: ', mean_squared_error(y_test, predicted_price2))
    scores2 = cross_val_score(model2, x, y, cv=5)
    print(f'Cross validation model 2 Scores: {scores2}')

    grid_search1 = GridSearchCV(estimator=model1,
                               param_grid={'preprocess__numeric__imputer__n_neighbors': [3, 7, 20, 40]})
    grid_search2 = GridSearchCV(estimator=model2,
                               param_grid={'preprocess__numeric__imputer__n_neighbors': [3, 7, 20, 40]})
    grid_search1.fit(x_train, y_train)
    grid_search2.fit(x_train, y_train)

    print('------------------------------------------------------------------------------')
    print(f'Grid search results of MODEL 1 by n_neighbors for numeric imputer: best score - {grid_search1.best_score_},'
          f' best_param - {grid_search1.best_params_}')
    print(f'Grid search results of MODEL 2 by n_neighbors for numeric imputer: best score - {grid_search2.best_score_},'
          f' best_param - {grid_search2.best_params_}')
    """It seems that it's better to use higher than 7 number of n_neighbors in KNN while missing value imputation."""

