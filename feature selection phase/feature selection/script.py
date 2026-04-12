import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.multitest import  multipletests

# --------------------- DATA PREPARATION ------------------- #


train_df = pd.read_csv('train_merged.tsv', sep='\t')
val_df = pd.read_csv('validation_merged.tsv', sep='\t')

labels_df = pd.read_csv('Train_clinical.tsv', sep='\t')


reg_info = train_df[['Chromosome', 'Start', 'End', 'Nclone']].copy()

names_of_regions = ("chr" + train_df['Chromosome'].astype(str) + ":" + train_df['Start'].astype(str) + "-" + train_df['End'].astype(str))

no_array_cols = ['Chromosome', 'Start', 'End', 'Nclone']

x_train_raw = train_df.drop(columns = no_array_cols).copy()
x_train_raw.index = names_of_regions

x_train_final = x_train_raw.T.copy()

x_val_raw = val_df.drop(columns = no_array_cols).copy()
x_val_raw.index = names_of_regions

x_val_final = x_val_raw.T.copy()

y_train = labels_df.set_index('Sample').copy()

same_samples = x_train_final.index.intersection(y_train.index)
print(len(same_samples))

x_train_final = x_train_final.loc[same_samples]

y_train = y_train.loc[same_samples, 'Subgroup']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)


# --------------------- KRUSKAL-WALLIS ------------------- #


p_value = []
h_stats = []

for feature in x_train_final.columns:
    val = x_train_final[feature]

    group1 = val[y_train == label_encoder.classes_[0]]
    group2 = val[y_train == label_encoder.classes_[1]]
    group3 = val[y_train == label_encoder.classes_[2]]

    stat, p = kruskal(group1, group2, group3)
    h_stats.append(stat)
    p_value.append(p)

kruskal_wallis_rslts = pd.DataFrame({'Region': x_train_final.columns, 'H stat': h_stats, 'p value': p_value})

rejected, accepted , _, _ = multipletests(kruskal_wallis_rslts['p value'], alpha = 0.05, method = 'bonferroni')

kruskal_wallis_rslts['p value Bonferroni'] = accepted
kruskal_wallis_rslts['significant'] = rejected

significant_from_KW = kruskal_wallis_rslts.loc[kruskal_wallis_rslts['significant'], "Region"].tolist()


x_train_after_kw = x_train_final[significant_from_KW].copy()
x_val_after_kw = x_val_final[significant_from_KW].copy()

# --------------------- ELASTIC NET ------------------- #

pipeline = Pipeline([('scaler', StandardScaler()),('clf', LogisticRegression(penalty='elasticnet',
                                                                             solver='saga',
                                                                             max_iter=10000,
                                                                             multi_class='multinomial',))])

grid_params = {'clf__C':[0.01, 0.1, 1, 10], 'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(estimator = pipeline, param_grid=grid_params, scoring = 'accuracy',cv=cv, n_jobs=-1, refit = True)

grid.fit(x_train_final, y_encoded)

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV accuracy:")
print(grid.best_score_)

best_model = grid.best_estimator_

cfs = best_model.named_steps['clf'].coef_
no_zero = np.abs(cfs).sum(axis=0) > 1e-8

final_features = x_train_final.columns[no_zero]

final_features_df = pd.DataFrame({"Regions": final_features})

train_pred = best_model.predict(x_train_final)
train_pred_labels = label_encoder.inverse_transform(train_pred)

val_pred = best_model.predict(x_val_final)
val_pred_labels = label_encoder.inverse_transform(val_pred)

val_preds_df = pd.DataFrame({"Sample": x_val_final.index, "Predicted": val_pred_labels})

print('Total features:', x_train_final.shape[1])
print('Selected features from ElasticNet:', len(final_features))

cfs_df = pd.DataFrame(cfs.T, index=x_train_final.columns, columns=label_encoder.classes_)

cfs_df['importance'] = np.abs(cfs_df).sum(axis=1)

cfs_selected = cfs_df.loc[final_features].sort_values(by='importance', ascending=False)

print(cfs_selected.head(20))

top_features = cfs_selected.head(20)

top_features['importance'].plot(kind='bar', figsize = (10,5))
plt.title('Most important region selected from ElasticNet')
plt.ylabel('Importance')
plt.show()

print("\nClassification report from training:")
print(classification_report(y_train, train_pred_labels))

val_preds_df.to_csv('validation predictions.csv', index=False)

cfs_selected.to_csv('selected coefficients.csv')

top_features.to_csv('top features.csv')

cfs_df.to_csv('all coefficients.csv')

cm = confusion_matrix(y_train, train_pred_labels, labels=label_encoder.classes_)
print("\nConfusion matrix:")
print(pd.DataFrame(cm, index = label_encoder.classes_, columns = label_encoder.classes_))

selected_mask = names_of_regions.isin(final_features)

train_EN = train_df[selected_mask].copy()
val_EN = val_df[selected_mask].copy()

train_EN.to_csv('training EN.csv', index=False)
val_EN.to_csv('validation EN.csv', index=False)

gs_sum = pd.DataFrame({
    "best_C":[grid.best_params_['clf__C']],
    "best_l1_ratio":[grid.best_params_['clf__l1_ratio']],
    "best_cv_accuracy":[grid.best_score_],
    "n_selected_features":[len(final_features)],
    "total_features_before_selection":[x_train_final.shape[1]],
    "solver":["saga"],
    "penalty":["elasticnet"],
    "max_iterations": [10000],
    "scaler": ['StandardScaler'],
    "cv_folds": [5]

})

gs_sum.to_csv("Best of Grid Search CV.csv", index=False)

gs_all_results = pd.DataFrame(grid.cv_results_)
gs_all_results = gs_all_results[[
    "param_clf__C",
    "param_clf__l1_ratio",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]].sort_values(by="rank_test_score")

gs_all_results.to_csv("All results of Grid Search CV.csv", index=False)



