import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../../data/avg_robot/DemoDataset_1Robot.csv')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Labels

# 1. Correlation-based Feature Selection
correlations = X.apply(lambda x: x.corr(data['goal_status'].astype('category').cat.codes))

# 2. Mutual Information
selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X, y)
mutual_info_scores = selector.scores_

# 3. Feature Importance from RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
importance_scores = clf.feature_importances_

# Combine the scores into a DataFrame for visualization
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': correlations,
    'Mutual Information': mutual_info_scores,
    'Random Forest Importance': importance_scores
}).sort_values(by='Random Forest Importance', ascending=False)

print(feature_scores)
