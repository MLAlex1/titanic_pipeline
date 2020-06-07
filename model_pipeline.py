import pandas as pd
import numpy as np

import pickle
import os
import argparse
import logging

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score

from utils.pipeline_utils import *

from xgboost import XGBClassifier

SEED = 256
file_dir = os.getcwd()

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

model = XGBClassifier(learning_rate=0.3, n_estimators=1500, 
					  max_depth=4, min_child_weight=10, subsample=0.85,
                      gamma=5, random_state=SEED, seed= SEED)

def features_engineering_():
    
    age = ['Age']
    imp_cols_age = ["Pclass", "Sex"]
    pipe_age = Pipeline([
                ('selector', ColumnSelector(key=age + imp_cols_age)),
                ('imputer', GroupbyImputer(age[0], imp_cols_age)),
                        ])
    
    other_num = ['Fare']
    pipe_other_num = Pipeline([
                    ('selector', ColumnSelector(key=other_num)),
                    ('imputer', PandasImputer(cols=other_num, strategy='most_frequent'))
                            ])
    
    cat_feats = ['Embarked', 'Sex']
    pipe_cat = Pipeline([
                        ('selector', ColumnSelector(key=cat_feats)),
                        ('imputer', PandasImputer(cols=cat_feats, strategy='most_frequent')),
                        ('encoder', EncodeByFrequency(cat_feats))
                        ])
    
    additional = ["SizeFam", "NoFamily"]
    pipe_additional = Pipeline([
                        ('selector', ColumnSelector(key=additional)),
                        ('imputer', PandasImputer(cols=additional, strategy='most_frequent'))
                        ])
    
    return [("age_pipe", pipe_age),
            ("other_num", pipe_other_num),
            ("pipe_cat", pipe_cat),
            ("pipe_additional", pipe_additional)]

def features_model():
    features = PandasFeatureUnion(features_engineering_())
    return Pipeline([('new_feats', Additional_feats()),
                    ('feat_union', features)])

def features_selection():
    return Pipeline([('select', FeatureSelection(threshold=0.0))])

def model_(model=model):
    return Pipeline([
        ('classifier', model)])

def model_pipeline():
    return Pipeline(
        features_model().steps +
        features_selection().steps +
        model_().steps)

def main():
	target = "Survived"
	parser = argparse.ArgumentParser(
	        formatter_class=argparse.ArgumentDefaultsHelpFormatter
	        )

	parser.add_argument('--input_dir', help='path of the input directory',
						 required=True)
	parser.add_argument('--output_dir', help='path of the output directory',
						 required=True)
	parser.add_argument('--search_params', help='do grid search or use default model',
						default=False)

	args = parser.parse_args()

	X_train = pd.read_csv(os.path.join(file_dir, args.input_dir, 'train.csv'))
	X_test = pd.read_csv(os.path.join(file_dir, args.input_dir, 'test.csv'))

	y_train = X_train[target].values
	#y_test = test[target].values

	xgb_model =  model_pipeline()

	logger.info("Fitting the model..")

	if args.search_params:
		logger.info("Searching params..")
		param_grid = {
	        "classifier__learning_rate": [0.1, 0.3],
	        "classifier__n_estimators": [1500, 2000],
	        "classifier__max_depth": [4, 5],
	    }
		cv = KFold(n_splits=4, shuffle=True, random_state=SEED)
		Search = GridSearchCV(xgb_model, cv=cv, param_grid=param_grid,
                              verbose=2, n_jobs=-1,
                              scoring='recall',
                              refit=True)
		Search.fit(X_train, y_train)
		mdl = Search.best_estimator_
		logger.info(Search.best_params_)
	else:
		mdl = xgb_model.fit(X_train, y_train.values)

	logger.info("Done")
	predictions = mdl.predict(X_test)

	logger.info("Saving model")
	with open(os.path.join(file_dir, args.output_dir, "titanic_model.pkl"), 'wb') as f:
		pickle.dump(mdl, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	main()