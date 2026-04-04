import os 
import pandas as pd
from src.config import DATA_URL
from src.features import prepare_datasets
from src.models import StressModels
from src.evaluation import plot_roc_curves_comparison, generate_performance_table


df = pd.read_csv(DATA_URL, sep=';').iloc[:, 1:]

X_train, X_test, y_train, y_test, scaler = prepare_datasets(df)

sm = StressModels(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
log_models = sm.train_logistic_regression()

# Visualisation ROC
plot_roc_curves_comparison(log_models, X_test, y_test)

# Entraînement des Arbres
important_vars = sm.get_top_features_from_cart()
tree_results = sm.train_tree_models()

# Tableau de résultats (Pour le site web)
df_perf = generate_performance_table(tree_results)