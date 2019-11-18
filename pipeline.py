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
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import recall_score, precision_score

pd.options.mode.chained_assignment = None  # default='warn'

from gridsearch import CategoricalSelector, NumericalSelector, CategoricalTransformer, f2_score


if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\data\train.csv", index_col=0)
    X, y = df.drop(['Revenue'], axis=1), df['Revenue']
    y = y.apply(lambda x: 1 if x is True else 0)

    prec = []
    rec = []
    f2 = []
    for cv in range(1, 30):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Defining the steps in the categorical pipeline
        categorical_pipeline = Pipeline(steps=[('cat_selector', CategoricalSelector()),
                                               ('cat_transformer', CategoricalTransformer()),
                                               ('one_hot_encoder', OneHotEncoder(sparse=False, drop='first',
                                                                                 handle_unknown='error'))])

        # Defining the steps in the numerical pipeline
        numerical_pipeline = Pipeline(steps=[('num_selector', NumericalSelector()),
                                             ('imputer', SimpleImputer(strategy='median')),
                                             # ('lrfe', LRFEPipeline()),
                                             ('std_scaler', StandardScaler())])

        # Combining numerical and categorical piepline into one full big pipeline horizontally
        # using FeatureUnion
        prep_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                       ('numerical_pipeline', numerical_pipeline)])

        params = {'max_depth': 3, 'max_features': 'log2', 'n_estimators': 100}

        full_pipeline = Pipeline(steps=[('prep', prep_pipeline),
                                        # ('rfe', RFE(estimator=DecisionTreeClassifier(), n_features_to_select=1)),
                                        ('sfm', SelectFromModel(estimator=DecisionTreeClassifier(), threshold='0.75*mean')),
                                        ('model', GradientBoostingClassifier(**params))])

        full_pipeline.fit(X_train, y_train)

        y_pred = full_pipeline.predict(X_test)
        prec.append(precision_score(y_test, y_pred))
        rec.append(recall_score(y_test, y_pred))
        f2.append(f2_score(y_test, y_pred))
        # print("Precision: {}".format(prec[-1]))
        # print("Recall: {}".format(rec[-1]))

    res = pd.DataFrame(np.array([prec, rec, f2]).T)
    print(res.loc[res[2] == res[2].median(), :])
    print(res)

    X_sub = pd.read_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\data\test.csv",
                        index_col=0)
    X_sub.loc[:, 'Revenue'] = ['TRUE' if x == 1 else "FALSE" for x in full_pipeline.predict(X_sub)]
    X_sub = X_sub[['Revenue']].copy()
    X_sub.to_csv(r"C:\Users\olive\Documents\GitHub\Kaggle-ColombusRetailClassification\experiments\e6_f2.csv")