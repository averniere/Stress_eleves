# Exemple de script principal ou chunk Quarto
import pandas as pd
from src.config import DATA_URL
from src.features import prepare_datasets
from src.models import StressModels
from src.evaluation import plot_roc_curves_comparison, generate_performance_table

df = pd.read_csv(DATA_URL, sep=';').iloc[:, 1:]

X_train, X_test, y_train, y_test, scaler = prepare_datasets(df)

sm = StressModels()
log_models = sm.train_logistic_regression(X_train, y_train)
print("Modèles logistiques entraînés. Meilleur C Lasso:", log_models['Lasso_CV_Auto'].estimators_[0].C_[0])

# Visualisation ROC
plot_roc_curves_comparison(log_models, X_test, y_test)

# Entraînement Arbres (Besoin métier : performance pure)
important_vars = sm.get_top_features_from_cart(n_top = 5)
tree_results = train_tree_models(X_train, y_train, X_test, y_test, feature_subset=important_vars)

# 6. Tableau de résultats (Pour le site web)
df_perf = generate_performance_table(tree_results)
print(df_perf)