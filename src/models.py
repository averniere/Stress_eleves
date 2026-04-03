import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error as rmse
from scipy.stats import randint, uniform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import RANDOM_STATE, CV_FOLDS, C_GRID_LASSO

logger = logging.getLogger(__name__)

class StressModels:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}

    def train_logistic_regression(self):
        """
        Entraîne les modèles Multiclasse, OneVsRest et Lasso CV.
        Compare les coefficients et performances.
        """
        logger.info("Entraînement des régressions logistiques...")
        models = {}
        
        # Multiclasse
        model_multi = LogisticRegression(penalty=None, max_iter=1000, random_state=RANDOM_STATE)
        model_multi.fit(self.X_train, self.y_train)
        models['Multiclasse'] = model_multi
        
        # OneVsRest (Référence)
        model_ovr = OneVsRestClassifier(LogisticRegression(penalty=None, max_iter=1000, random_state=RANDOM_STATE))
        model_ovr.fit(self.X_train, self.y_train)
        models['OneVsRest'] = model_ovr
        
        # Lasso avec Validation Croisée (Fine-tuning de C)
        model_lasso_cv = OneVsRestClassifier(
            LogisticRegressionCV(
                penalty='l1', 
                Cs=C_GRID_LASSO, 
                cv=CV_FOLDS, 
                solver='liblinear', 
                scoring='roc_auc', 
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )
        model_lasso_cv.fit(self.X_train, self.y_train)
        models['Lasso_CV'] = model_lasso_cv
        
        # Log des variables sélectionnées par Lasso
        self._log_lasso_selection(model_lasso_cv)
        
        return models

    def _log_lasso_selection(self, model_lasso):
        """Log les variables mises à zéro par le Lasso """
        df_coef = pd.DataFrame(model_lasso.estimators_[0].coef_, columns=self.X_train.columns)
        vars_non_zero = (df_coef != 0).any(axis=0)
        logger.info(f"Lasso : {vars_non_zero.sum()} variables sélectionnées sur {len(vars_non_zero)}")
        logger.debug(f"Variables éliminées : {df_coef.columns[~vars_non_zero].tolist()}")

    def get_top_features_from_cart(self, n_top=5):
        """
        Entraîne un CART simple, extrait les importances, et retourne les noms des n_top variables.
        """
        cart = DecisionTreeRegressor(random_state=RANDOM_STATE)
        cart.fit(self.X_train, self.y_train)
    
        importances = cart.feature_importances_
        feature_names = self.X_train.columns
    
        df_imp = pd.DataFrame({'feature': feature_names, 
                               'importance': importances}).sort_values(by='importance', 
                                                                       ascending=False)
    
        top_features = df_imp['feature'].head(n_top).tolist()

        # A commenter éventuellement après
        print(f"--- Sélection automatique de variables (CART) ---")
        print(f"Top {n_top} variables : {top_features}")
        print(f"Importances associées : {df_imp.head(n_top)['importance'].values.round(3)}")
    
        return top_features

    def train_tree_models(self, X_train_sub, X_test_sub):
        """
        X_train_sub, X_test_sub : base de données dans laquelle on a pré-sélectionné les variables de plus grande importance.
        Entraîne CART, Random Forest et Gradient Boosting avec Fine-Tuning.
        """
        results = []
        
        # Random Forest avec fine-tuning
        param_dist_rf = {
            'n_estimators': randint(50, 150),
            'max_depth': [5, 10, 15, None],
            'min_samples_split': randint(2, 10)
        }
        rf = RandomForestRegressor(random_state=RANDOM_STATE, oob_score=True)
        rf_search = RandomizedSearchCV(rf, param_dist_rf, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        rf_search.fit(X_train_sub, self.y_train)
        best_rf = rf_search.best_estimator_
        
        predict_train_rf = best_rf.predict(X_train_sub)
        predict_test_rf = best_rf.predict(X_test_sub)

        rmse_test_rf = rmse(self.y_test, predict_test_rf)
        rmse_train_rf = rmse(self.y_train, predict_train_rf)

        results.append({
            'model_name': 'RandomForest_Optimised',
            'model' : best_rf,
            'rmse_test': rmse_test_rf,
            'rmse_train' : rmse_train_rf, 
            'params': rf_search.best_params_
        })

        # Gradient Boosting avec fine-tuning
        param_dist_gb = {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 8)
        }
        gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
        gb_search = RandomizedSearchCV(gb, param_dist_gb, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        gb_search.fit(X_train_sub, self.y_train)
        best_gb = gb_search.best_estimator_

        predict_train_gb = best_gb.predict(X_train_sub)
        predict_test_gb = best_gb.predict(X_test_sub)

        rmse_test_gb = rmse(self.y_test, predict_test_gb)
        rmse_train_gb = rmse(self.y_train, predict_train_gb)
        
        results.append({
            'model_name': 'GradientBoosting_Optimised',
            'model' : best_gb, 
            'rmse_test': rmse_test_gb,
            'rmse_train' : rmse_train_gb,
            'params': gb_search.best_params_
        })
        
        return results
