from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


class Regressors:
    def __init__(self, df):
        self.regressors = [LinearRegression(), tree.DecisionTreeRegressor(), RandomForestRegressor()]
        self.df = df
        self.x = self.df[
            ['category', 'old_price', 'sellable_online', 'other_colors', 'designer_type', 'depth', 'height', 'width']]
        self.y = self.df['price']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20)
        self.preprocess = Regressors.preprocess()

    @staticmethod
    def preprocess():
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=7)),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown="ignore"))
        ])
        column_preprocess = ColumnTransformer(transformers=[
            ('numeric', numeric_transformer, ['old_price', 'depth', 'height', 'width']),
            ('categorical', categorical_transformer, ['category', 'sellable_online', 'other_colors', 'designer_type'])
        ])
        return column_preprocess

    def train(self):
        for regressor in self.regressors:
            model = Pipeline(steps=[
                ('preprocess', self.preprocess),
                ('reg', regressor)
            ])
            model.fit(self.x_train, self.y_train)
            grid_search = GridSearchCV(estimator=model,
                                        param_grid={'preprocess__numeric__imputer__n_neighbors': [3, 7, 20]})
            grid_search.fit(self.x_train, self.y_train)
            self.print_results(model, regressor, grid_search)

    def print_results(self, model, regressor, grid_search):
        print(f'{regressor.__class__.__name__} MODEL PERFORMANCE:')
        predicted_price = model.predict(self.x_test)
        print(f'{regressor.__class__.__name__} Model Score: {model.score(self.x_test, self.y_test)}')
        print(f'{regressor.__class__.__name__} Model Mean Squared Error: ', mean_squared_error(self.y_test, predicted_price))
        scores = cross_val_score(model, self.x, self.y, cv=5)
        print(f'{regressor.__class__.__name__} Model Cross validation Scores: {scores}')
        print(
            f'Grid search results of {regressor.__class__.__name__} MODEL by n_neighbors for numeric imputer: best score - {grid_search.best_score_},'
            f' best_param - {grid_search.best_params_}')
        print()



