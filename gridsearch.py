import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from collections import namedtuple


pd.options.mode.chained_assignment = None  # default='warn'

Grid = namedtuple("Grid", ['model', 'param_grid'])

grids = [
    Grid(LogisticRegression,
         {'model__solver': ('liblinear',)}),
    Grid(BaggingClassifier,
        {'model__n_estimators': (10, 100, 200, 400, 600, 800),
         'model__max_samples': (0.2, 0.4, 0.6, 0.8, 1.0),
         'model__max_features': (0.2, 0.4, 0.6, 0.8, 1.0)}),
    Grid(RandomForestClassifier,
        {'model__max_depth': (70, 80, 90, 100, None),
         'model__max_features': ('auto', 'log2', None),
         'model__n_estimators': (10, 100, 200, 400, 600, 800)}),
    Grid(GradientBoostingClassifier,
         {'model__max_depth': (3, 4, 5, 6),
          'model__max_features': ('auto', 'log2', None),
          'model__n_estimators': (10, 100, 200, 400, 600, 800)}),
    Grid(SVC,
    {"model__C": (4, 8, 10, 12),
     "model__degree": (3, 4, 5, 6)})
]


# Custom Transformer that extracts columns passed as argument to its constructor
class CategoricalSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self):
        self._feature_names = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region',
                                'TrafficType',
                                'SpecialDay']

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names]


# Custom Transformer that extracts columns passed as argument to its constructor
class NumericalSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self):
        self._feature_names = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                              'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names]


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes in a list of values as its argument
    def __init__(self):
        self.minority_traffic_types = [7, 9, 12, 14, 15, 16, 17, 18, 19]
        self.minority_browser_types = [3, 7, 9, 11, 12, 13]

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):

        # Convert minority operating systems to one "Other" category
        X.loc[:, 'OperatingSystems'] = X['OperatingSystems'].apply(lambda x: x if x < 4 else -1)
        # Convert minority traffic types to one "Other" category
        X.loc[:, 'TrafficType'] = X['TrafficType'].apply(lambda x: x if x not in self.minority_traffic_types else -1)
        # Convert minority browser types to one "Other" category
        X.loc[:, 'Browser'] = X['Browser'].apply(lambda x: x if x not in self.minority_browser_types else -1)

        return X


if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\data\train.csv", index_col=0)
    X, y = df.drop(['Revenue'], axis=1), df['Revenue']
    y = y.apply(lambda x: 1 if x is True else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    for grid in grids:

        # Defining the steps in the categorical pipeline
        categorical_pipeline = Pipeline(steps=[('cat_selector', CategoricalSelector()),
                                               ('cat_transformer', CategoricalTransformer()),
                                               (
                                               'one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='error'))])

        # Defining the steps in the numerical pipeline
        numerical_pipeline = Pipeline(steps=[('num_selector', NumericalSelector()),
                                             ('imputer', SimpleImputer(strategy='median')),
                                             ('std_scaler', StandardScaler())])

        # Combining numerical and categorical piepline into one full big pipeline horizontally
        # using FeatureUnion
        prep_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                       ('numerical_pipeline', numerical_pipeline)])

        full_pipeline = Pipeline(steps=[('prep', prep_pipeline),
                                        ('model', grid.model())])

        print("\nStarting Grid: {}".format(grid))

        search = GridSearchCV(full_pipeline, grid.param_grid, scoring='accuracy', cv=5, verbose=1)

        search.fit(X_train, y_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)



    # The full pipeline as a step in another pipeline with an estimator as the final step

    # Can call fit on it just like any other pipeline
    # full_pipeline.fit(X_train, y_train)

    # Can predict with it like any other pipeline
    # y_pred = full_pipeline.predict(X_test)
    #
    # score = accuracy_score(y_test, y_pred)
    #
    # print(score)
