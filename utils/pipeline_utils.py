# Classes for the pipeline
import sklearn.linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

SEED = 256

def agg_index(x):
    return str(x[0]) + "_" + x[1]

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select single columns from data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.key]
    
    
class PandasFeatureUnion(_BaseComposition, TransformerMixin):
    """
    Concatenate featues from the Pipeline
    """

    def __init__(self, transformer_list, custom_idx_col=None):
        self.transformer_list = transformer_list
        self.custom_idx_col = custom_idx_col

    def fit(self, X, y=None):
        for label, transformer in self.transformer_list:
            transformer.fit(X)
        return self

    def transform(self, X, y=None):
        if self.custom_idx_col == None:
            X = X.reset_index(drop=True)
            Xout = pd.DataFrame(index=X.index)
        else:
            Xout = pd.DataFrame(index=X[self.custom_idx_col].unique().tolist())

        for label, transformer in self.transformer_list:
            Xout = Xout.join(transformer.transform(X))
        self.feature_names_ = Xout.columns
        return Xout
    
        
class EncodeByFrequency(BaseEstimator,TransformerMixin):
    '''
    Encodes a column by frequency
    
    Parameters
    ---------------------------
    :param cols : list of cols that this class will apply to
    
    '''
    def __init__(self, cols):
        self.cols = cols
    
    def fit(self,X,y=None):
        self.encoders = {}
        for col in self.cols:
            models = {}
            sort_index = X[col].value_counts().index.tolist()
            X[col] = X[col].astype("category")
            X[col].cat.set_categories(sort_index, inplace=True)
            X = X.sort_values([col])
            for i, val in enumerate(X[col].unique()):
                models[val] = i
            self.encoders[col] = models 
            
        return self
    
    def transform(self,X,y=None):
        for col in self.cols:
            existing_cats = list(self.encoders[col].keys())
            data_cats = X[col].tolist()
            list_diff = list(set(data_cats)-set(existing_cats))
            for item in list_diff:
                max_item = max(self.encoders[col].values())
                self.encoders[col][item] = max_item + 1
    
            X[col] = X[col].replace(self.encoders[col]).astype(int)
        return X


class PandasImputer(BaseEstimator, TransformerMixin):
    """Pandas wrapper for `sklearn.impute.SimpleImputer \
    <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_

    :param cols: list of columns to apply imputer to
    :param missing_values: values classed as null
    :param strategy: str mean, median or most_frequent
    :param kwargs: dict kwargs passed into SimpleImputer
    """
    def __init__(self, cols, missing_values=np.nan, strategy="mean", **kwargs):
        self.cols = cols
        self.missing_values = missing_values
        self.strategy = strategy
        self.kwargs = kwargs

    def fit(self, X, y=None):
        si = SimpleImputer(missing_values=self.missing_values,
                           strategy=self.strategy,
                           **self.kwargs)
        si.fit(X[self.cols])
        self.to_replace_dict = {col: self.missing_values
                                for col in self.cols}
        self.value_dict = {col: si.statistics_[ii]
                           for ii, col in enumerate(self.cols)}
        return self

    def transform(self, X, y=None):
        X = X.replace(self.to_replace_dict, self.value_dict)
        return X


class Aggregate_sum(BaseEstimator, TransformerMixin):
    """ creates dummies from a certain variable and\
        aggreges using certain methodology
    
    :param cols: list of columns this class is applied to
    :param agg_col: str aggregating column
    :param agg_type: str aggregating methodology
    """
    
    def __init__(self, cols, agg_col='Claim_Num', agg_type='sum'):
        self.cols = cols
        self.agg_col = agg_col
        self.agg_type = agg_type
        
    def fit(self, X, y=None):
        self.cols_dummies = {}
        for ii, col in enumerate(self.cols):
            keys = pd.Series(X[col].unique().tolist())
            r_keys = col.lower() + "_" + keys
            self.cols_dummies[col] = {'keys': keys, 
                                      'r_keys': r_keys,
                                      'len' : r_keys.shape[0]}
        return self
    
    def transform(self, X, y=None):
        for ii, col in enumerate(self.cols):
            X_col = pd.concat([X[col], self.cols_dummies[col]['keys']])
            dummies = pd.get_dummies(X_col, 
                prefix=col.lower())[self.cols_dummies[col]['r_keys']][:-self.cols_dummies[col]['len']]
            X_temp = X.merge(dummies, left_index=True, right_index=True)
            X_temp = X_temp[dummies.columns.tolist() + [self.agg_col]]
            X_temp = X_temp.groupby(self.agg_col).agg(self.agg_type)
            if ii == 0: 
                X_fin = X_temp.copy()
                continue
            X_fin = X_fin.merge(X_temp, left_index=True, right_index=True)
        return X_fin

class GroupbyImputer(BaseEstimator, TransformerMixin):
    """ impute features using multiple columns
    
    :param col: str col this class is applied to
    :param fromcols: list of columns to use for grouping
    :param agg_func: str aggregating methodology
    :param agg_type: column type
    """
    
    def __init__(self, col, fromcols, agg_func="mean", agg_type=int):
        self.col = col
        self.fromcols = fromcols
        if not isinstance(self.fromcols, list):
            self.fromcols = [self.fromcols]
        self.agg_type = agg_type
        self.agg_func = agg_func
        
    def fit(self, X, y=None):
        self.lookup = X.groupby(self.fromcols).agg(self.agg_func).astype(self.agg_type)
        new_index = self.lookup.index.tolist()
        new_index = list(map(agg_index, new_index))
        self.lookup.index = new_index
        self.lookup = self.lookup.to_dict()[self.col]
        
        self.global_fill = X[self.col].apply(self.agg_func) 
        return self
        
    def transform(self, X, y=None):
        X['temp'] = X[self.fromcols].apply(agg_index, axis=1)
        X[self.col] = np.where(X[self.col].isnull(), X['temp'].replace(self.lookup), X[self.col])
        X[self.col].fillna(self.global_fill, inplace=True)
        return X[self.col]
        
class Additional_feats(BaseEstimator, TransformerMixin):
    """Create some additional features 
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['SizeFam'] = X['SibSp'] + X['Parch'] + 1
        X['NoFamily'] = np.where(X['SizeFam'] == 1, 1, 0)
        return X

class FeatureSelection(BaseEstimator, TransformerMixin):
    """Some simple feature selection"""

    def __init__(self, threshold=0.0, **model_params):
        self.threshold = threshold
        self.model_params = model_params

    def model_(self):

        if self.model_params:
            xgb_regr = XGBClassifier(random_state=SEED, 
                                     seed = SEED, 
                                     **self.model_params)
        else:
            xgb_regr = XGBClassifier(
                n_estimators=1500,
                learning_rate=0.3,
                max_depth=5,
                subsample=0.9,
                min_samples_leaf=10,
                n_jobs=4,
                gamma=5,
                random_state=SEED,
                seed = SEED)

        return xgb_regr

    def fit(self, X, y):
        model = self.model_()
        model.fit(X, y)
        # feature_scores = model.regressor_._Booster.get_score(
            # importance_type="gain")
        feature_scores = {col: score for col,score in zip(X.columns, model.feature_importances_)}
        self.model = model
        self.kept_columns_ = [k for k, v in feature_scores.items()
                              if v > self.threshold]
        return self

    def transform(self, X, y=None):
        return X[self.kept_columns_]
