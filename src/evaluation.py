import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from src.config import FIGURES_DIR, TABLES_DIR
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(model, X_test, y_test, model_name, save=True):
    """Génère et sauvegarde la matrice de confusion."""
    y_pred = model.predict(X_test)
    labels = model.classes_ if hasattr(model, 'classes_') else None
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'Matrice de Confusion - {model_name}')
    
    if save:
        filepath = FIGURES_DIR / f"confusion_{model_name}.png"
        plt.savefig(filepath, dpi=300)
        print(f"Matrice sauvegardée : {filepath}")
    
    plt.show()

def plot_roc_curves_comparison(models_dict, X_test, y_test, save=True):
    """Compare les courbes ROC de plusieurs modèles (OneVsRest requis pour multiclasse)."""
    plt.figure(figsize=(10, 8))
    
    # Binarisation des labels pour ROC multiclasse
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            
            # Calcul ROC moyenne (Macro)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # ROC par classe
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # ROC Macro Average
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
            plt.plot(fpr["macro"], tpr["macro"], label=f'{name} (AUC Macro = {roc_auc["macro"]:.2f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Comparaison des Courbes ROC (Macro Average)')
    plt.legend(loc="lower right")
    
    if save:
        filepath = FIGURES_DIR / "roc_comparison.png"
        plt.savefig(filepath, dpi=300)
        print(f"ROC sauvegardée : {filepath}")
    
    plt.show()

def generate_performance_table(tree_results, save=True):
    """Crée un DataFrame propre des performances des modèles arbres/boosting."""
    data = []
    for res in tree_results:
        data.append({
            'Modèle': res['model_name'],
            'RMSE Train': res['rmse_train'],
            'RMSE Test': res['rmse_test'],
            'Meilleurs Paramètres': str(res['best_params'])
        })
    
    df_res = pd.DataFrame(data)
    
    if save:
        filepath = TABLES_DIR / "performance_trees.csv"
        df_res.to_csv(filepath, index=False)
        print(f"Tableau sauvegardé : {filepath}")
        
    return df_res