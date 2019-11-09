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

from gridsearch import CategoricalSelector, NumericalSelector, CategoricalTransformer


if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\data\train.csv", index_col=0)
    X, y = df.drop(['Revenue'], axis=1), df['Revenue']
    y = y.apply(lambda x: 1 if x is True else 0)

    # Defining the steps in the categorical pipeline
    categorical_pipeline = Pipeline(steps=[('cat_selector', CategoricalSelector()),
                                           ('cat_transformer', CategoricalTransformer()),
                                           ('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='error'))])

    # Defining the steps in the numerical pipeline
    numerical_pipeline = Pipeline(steps=[('num_selector', NumericalSelector()),
                                         ('imputer', SimpleImputer(strategy='median')),
                                         ('std_scaler', StandardScaler())])

    # Combining numerical and categorical piepline into one full big pipeline horizontally
    # using FeatureUnion
    prep_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                   ('numerical_pipeline', numerical_pipeline)])

    full_pipeline = Pipeline(steps=[('prep', prep_pipeline),
                                    ('model', BaggingClassifier(max_features=1.0, max_samples=0.2, n_estimators=200))])

    full_pipeline.fit(X, y)

    X_test = pd.read_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\data\test.csv", index_col=0)
    X_test.loc[:, 'Revenue'] = ['TRUE' if x==1 else "FALSE" for x in full_pipeline.predict(X_test)]
    X_sub = X_test[['Revenue']].copy()
    X_sub.to_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\experiments\e2.csv")