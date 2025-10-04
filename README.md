# Breast-Cancer-detection
Achieved ROC AUC Score: 0.9580740255486406
Mean Squared Error:0.03508771929824561
RMSE: 0.1873
Accuracy Score: 0.9649
Precision Score: 0.9589
F1 Score: 0.9722
Cross-Validated F1 Score: 0.9694 ± 0.0180Visualized feature importances and confusion matrix for interpretability  Used cross-validation to ensure robustness  Modular, readable code with clear metrics and plots
# CODE:- 
#Random Forest Classifier
import pandas as pd 
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, r2_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
load_data = load_breast_cancer()
X = load_data.data
y = load_data.target
X_train, X_test,y_train,  y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_binary = [1 if p>0.5 else 0 for p in y_pred]
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
f1_score_predict = f1_score(y_test, y_pred_binary)
cv_scores =  cross_val_score(model, X, y, cv = 5, scoring = 'f1')
sns.heatmap(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred_binary, name='Predicted')), annot=True, fmt='d', cmap='Blues')
def plot():
    x_label = "Malignant"
    y_label = "Benign"
    plt.plot([0,1] , [1,0], marker = 'o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Malignant vs Benign")
    plt.grid(True)
    plt.show()
plot()
importances = model.feature_importances_
features = load_data.feature_names
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.show()
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")
print(f"Mean Squared Error:{mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_binary):.4f}")
print(f"Precision Score: {precision_score(y_test, y_pred_binary):.4f}")
print(f"F1 Score: {f1_score_predict:.4f}")
print(f"Cross-Validated F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
