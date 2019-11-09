import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.options.mode.chained_assignment = None  # default='warn'

# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

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

        # returns numpy array
        return X.values


if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\data\train.csv", index_col=0)
    X, y = df.drop(['Revenue'], axis=1), df['Revenue']
    y = y.apply(lambda x: 1 if x is True else 0)

    # Categrical features to pass down the categorical pipeline
    categorical_features = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
                            'SpecialDay']

    # Numerical features to pass down the numerical pipeline
    numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                          'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']

    # Defining the steps in the categorical pipeline
    categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features)),
                                           ('cat_transformer', CategoricalTransformer()),
                                           ('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='error'))])

    # Defining the steps in the numerical pipeline
    numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                         ('imputer', SimpleImputer(strategy='median')),
                                         ('std_scaler', StandardScaler())])

    # Combining numerical and categorical piepline into one full big pipeline horizontally
    # using FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                   ('numerical_pipeline', numerical_pipeline)])

    # The full pipeline as a step in another pipeline with an estimator as the final step
    full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),
                                      ('model', LogisticRegression(solver='liblinear'))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Can call fit on it just like any other pipeline
    full_pipeline_m.fit(X_train, y_train)

    # Can predict with it like any other pipeline
    y_pred = full_pipeline_m.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    print(score)
