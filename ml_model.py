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
import tensorflow as tf


class Regressors:
    def __init__(self, df):
        self.regressors = [LinearRegression(), tree.DecisionTreeRegressor(), RandomForestRegressor()]
        self.df = df
        self.x = self.df[
            ['category', 'old_price', 'sellable_online', 'other_colors', 'designer_type', 'depth', 'height', 'width']]
        self.cat_var = ['category', 'sellable_online', 'other_colors', 'designer_type']
        self.num_var = ['old_price', 'depth', 'height', 'width']
        self.y = self.df['price']
        self.encoder = OneHotEncoder()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20)
        self.preprocess = Regressors.preprocess()
        self.models = {}

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
            self.models[regressor.__class__.__name__] = model.score(self.x_test, self.y_test)
        # self.train_ann()
        return {max(self.models): self.models[max(self.models)]}

    def print_results(self, model, regressor, grid_search):
        print(f'{regressor.__class__.__name__} MODEL PERFORMANCE:')
        predicted_price = model.predict(self.x_test)
        print(f'{regressor.__class__.__name__} Model R2 Score: {model.score(self.x_test, self.y_test)}')
        print(f'{regressor.__class__.__name__} Model Mean Squared Error: ', mean_squared_error(self.y_test, predicted_price))
        scores = cross_val_score(model, self.x, self.y, cv=5)
        print(f'{regressor.__class__.__name__} Model Cross validation R2 Scores: {scores}')
        print(
            f'Grid search results of {regressor.__class__.__name__} MODEL by n_neighbors for numeric imputer: best score - {grid_search.best_score_},'
            f' best_param - {grid_search.best_params_}')
        print()

    def train_ann(self):
        X = self.x
        for var in self.cat_var:
            X[var] = self.encoder.fit_transform(X[[var]]).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.2, random_state=0)
        print(X_train.shape)
        print(y_train.shape)
        sc = StandardScaler()
        for var in self.num_var:
            sc.fit(X_train[[var]])
            X_train[var] = sc.transform(X_train[[var]])
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1))
        ann.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.r_squared])
        ann.fit(X_train, y_train, batch_size=32, epochs=100)

    def r_squared(self, y_true, y_pred):
        tss = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        rss = tf.reduce_sum(tf.square(y_true - y_pred))
        r_squared = 1 - (rss / tss)
        return r_squared






