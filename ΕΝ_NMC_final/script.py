import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#----------------  DATA INPUT ----------------#

train_df = pd.read_csv('train_merged.tsv', sep='\t')
labels_df = pd.read_csv('Train_clinical.tsv', sep='\t')

reg_info =train_df[['Chromosome', 'Start', 'End', 'Nclone']].copy()

names_of_region = (
    "chr" +train_df['Chromosome'].astype(str) +":" +
    train_df['Start'].astype(str) + "-" + train_df['End'].astype(str)
)

no_array_cols = ['Chromosome', 'Start', 'End', 'Nclone']

x_train_raw =train_df.drop(columns = no_array_cols).copy()
x_train_raw.index = names_of_region

x_train_final = x_train_raw.T.copy()

y_train = labels_df.set_index('Sample')['Subgroup'].copy()

same_samples =x_train_final.index.intersection(y_train.index)
x_train_final = x_train_final.loc[same_samples]
y_train = y_train.loc[same_samples]

X = x_train_final.copy()
y=y_train.copy()

#----------------  ENCODE LABELS ----------------#

lencoder = LabelEncoder()

y_encoded = pd.Series(lencoder.fit_transform(y),index=y.index, name='target')

print("Classes:", lencoder.classes_)

#----------------  HYPERPARAMETER GRID ----------------#

C_grid = [0.01, 0.1, 1, 10]
l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
top_k_grid = [5,10, 20, 50]

param_grid =list(ParameterGrid({
    'C': C_grid,
    'l1_ratio': l1_ratio_grid,
    'top_k': top_k_grid,
}))

#----------------  REPEATED NESTED CV SETTINGS ----------------#

outter_seeds = [1,2,3]

all_runs_results = []
all_outter_results = []
all_inner_results = []

#----------------  REPEATED NESTED CV ----------------#

for run_id, outter_seed in enumerate(outter_seeds, start=1):

    print("Run:", run_id)

    outter_cv =StratifiedKFold(n_splits=5, shuffle = True,random_state=outter_seed)

    outter_results = []
    inner_results = []
    selected_features_by_fold = {}

    for outter_fold,(outter_train_idx, outter_test_idx) in enumerate(outter_cv.split(X, y_encoded), start =1):

        print("outter fold:", outter_fold)

        x_outter_train = X.iloc[outter_train_idx].copy()
        x_outter_test = X.iloc[outter_test_idx].copy()
        y_outter_train = y_encoded.iloc[outter_train_idx].copy()
        y_outter_test = y_encoded.iloc[outter_test_idx].copy()

        # ----------------  INNER CV ----------------#

        inner_cv = StratifiedKFold(n_splits=5, shuffle = True,random_state=outter_seed)

        best_inner_score = -np.inf
        best_inner_std = None
        best_params = None

        for params in param_grid:
            inner_fold_scores =[]

            for inner_train_idx, inner_val_idx in inner_cv.split(x_outter_train, y_outter_train):

                x_inner_train =x_outter_train.iloc[inner_train_idx].copy()
                x_inner_val =x_outter_train.iloc[inner_val_idx].copy()
                y_inner_train = y_outter_train.iloc[inner_train_idx].copy()
                y_inner_val = y_outter_train.iloc[inner_val_idx].copy()

                # ----------------  STEP 1: FIT ELASTIC NET ----------------#
                # ---------------- LOGISTIC REGRESSION ON INNER CV ----------------#

                scaler = StandardScaler()
                x_inner_train_scaled = scaler.fit_transform(x_inner_train)
                x_inner_val_scaled = scaler.transform(x_inner_val)

                elastic_net_model = LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    max_iter=10000,
                    C = params['C'],
                    l1_ratio = params['l1_ratio'],
                    random_state=outter_seed,
                    multi_class = 'multinomial'
                )

                elastic_net_model.fit(x_inner_train_scaled, y_inner_train)

                # ----------------  STEP 2: RANK FEATURES BY ABS COEFFICIENTS ----------------#

                cfs = elastic_net_model.coef_
                importance = np.abs(cfs).sum(axis = 0)
                importance_df =pd.DataFrame({
                    'feature': x_inner_train.columns,
                    'importance': importance,
                }).sort_values(by='importance', ascending=False)

                selected_features = importance_df.head(params['top_k'])['feature'].tolist()

                # ----------------  STEP 3: TRAIN NMC ON TOP K FEATURES ----------------#

                x_inner_train_selected = pd.DataFrame(
                    x_inner_train_scaled,
                    index = x_inner_train.index,
                    columns = x_inner_train.columns,
                )[selected_features]

                x_inner_val_selected = pd.DataFrame(
                    x_inner_val_scaled,
                    index=x_inner_val.index,
                    columns=x_inner_val.columns,
                )[selected_features]

                nmc = NearestCentroid()
                nmc.fit(x_inner_train_selected, y_inner_train)

                y_inner_pred =nmc.predict(x_inner_val_selected)
                inner_acc =accuracy_score(y_inner_val, y_inner_pred)
                inner_fold_scores.append(inner_acc)

            mean_inner_acc = np.mean(inner_fold_scores)
            std_inner_acc = np.std(inner_fold_scores)

            inner_results.append({
                'run_id': run_id,
                'outter_seed': outter_seed,
                'outter_fold': outter_fold,
                'C': params['C'],
                'l1_ratio': params['l1_ratio'],
                'top_k': params['top_k'],
                'mean_inner_accuracy': mean_inner_acc,
                'std_inner_accuracy': std_inner_acc,
            })

            if mean_inner_acc >best_inner_score:
                best_inner_score = mean_inner_acc
                best_inner_std = std_inner_acc
                best_params = params

        print('Best inner params:', best_params)
        print('Best inner accuracy:', round(best_inner_score,4))

        # ----------------  REFIT BEST PIPELINE ON FULL OUTTER TRAIN ----------------#

        scaler = StandardScaler()
        x_outter_train_scaled = scaler.fit_transform(x_outter_train)
        x_outter_test_scaled = scaler.transform(x_outter_test)

        best_elastic_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=10000,
            C=best_params['C'],
            l1_ratio=best_params['l1_ratio'],
            random_state=outter_seed,
            multi_class='multinomial'
        )

        best_elastic_model.fit(x_outter_train_scaled, y_outter_train)

        best_cfs = best_elastic_model.coef_
        best_importance = np.abs(best_cfs).sum(axis = 0)

        best_importance_df = pd.DataFrame({
            'feature': x_outter_train.columns,
            'importance': best_importance,
        }).sort_values(by='importance', ascending=False)

        best_selected_features = best_importance_df.head(best_params['top_k'])['feature'].tolist()
        selected_features_by_fold[f'run_{run_id}_outter_fold_{outter_fold}'] = best_selected_features

        x_outter_train_selected = pd.DataFrame(
            x_outter_train_scaled,
            index = x_outter_train.index,
            columns = x_outter_train.columns,
        )[best_selected_features]

        x_outter_test_selected = pd.DataFrame(
            x_outter_test_scaled,
            index=x_outter_test.index,
            columns=x_outter_test.columns,
        )[best_selected_features]

        # ----------------  NMC ON FULL OUTTER TRAIN / TEST ON OUTTER TEST ----------------#

        final_nmc = NearestCentroid()
        final_nmc.fit(x_outter_train_selected, y_outter_train)

        y_outter_pred = final_nmc.predict(x_outter_test_selected)
        outter_acc = accuracy_score(y_outter_test, y_outter_pred)

        print("Outter accuracy:", round(outter_acc,4))

        y_outter_test_labels =lencoder.inverse_transform(y_outter_test)
        y_outter_pred_labels =lencoder.inverse_transform(y_outter_pred)

        cm = confusion_matrix(y_outter_test_labels, y_outter_pred_labels, labels = lencoder.classes_)
        cm_df = pd.DataFrame(cm, index = lencoder.classes_, columns = lencoder.classes_)

        report_dict = classification_report(y_outter_test_labels, y_outter_pred_labels, output_dict=True, zero_division=0)

        report_df = pd.DataFrame(report_dict).transpose()

        coef_df = pd.DataFrame(best_cfs.T, index = x_outter_train.columns
                               , columns = lencoder.classes_)

        coef_df['importance'] = np.abs(coef_df[lencoder.classes_]).sum(axis = 1)
        coef_selected_df =coef_df.loc[best_selected_features].sort_values(by='importance', ascending=False)

        pred_df = pd.DataFrame({
            'sample': x_outter_test.index,
            'true_label': y_outter_test_labels,
            'predicted_label': y_outter_pred_labels,
        })

        pred_df.to_csv(f'run{run_id}_outter_fold_{outter_fold}_predictions_ENLR_NMC.csv', index=False)
        best_importance_df.to_csv(f'run{run_id}_outter_fold_{outter_fold}_feature_ranking_ENLR_NMC.csv', index=False)
        coef_df.to_csv(f'run{run_id}_outter_fold_{outter_fold}_all_coefficients_ENLR_NMC.csv')
        coef_selected_df.to_csv(f'run{run_id}_outter_fold_{outter_fold}_selected_coefficients_ENLR_NMC.csv')
        cm_df.to_csv(f'run{run_id}_outter_fold_{outter_fold}_confusion_matrix_ENLR_NMC.csv')
        report_df.to_csv(f'run{run_id}_outter_fold_{outter_fold}_classification_report_ENLR_NMC.csv')


        outter_results.append({
            'run_id': run_id,
            'outter_seed': outter_seed,
            'outter_fold': outter_fold,
            'best_C': best_params['C'],
            'best_l1_ratio': best_params['l1_ratio'],
            'best_top_k': best_params['top_k'],
            'best_inner_accuracy': best_inner_score,
            'best_inner_std': best_inner_std,
            'outter_test_accuracy': outter_acc,
            'n_selected_features': len(best_selected_features),
        })


    # ----------------  SUMMARY PER ITERATION ----------------#

    outter_results_df =pd.DataFrame(outter_results)
    inner_results_df =pd.DataFrame(inner_results)

    iter_mean_outter_acc =outter_results_df['outter_test_accuracy'].mean()
    iter_std_outter_acc =outter_results_df['outter_test_accuracy'].std()

    outter_results_df.to_csv(f'run{run_id}_nested_cv_outter_results_ENLR_NMC.csv', index=False)
    inner_results_df.to_csv(f'run{run_id}_nested_cv_inner_results_ENLR_NMC.csv', index=False)

    all_runs_results.append({
        'run_id': run_id,
        'outter_seed': outter_seed,
        'mean_outter_acc': iter_mean_outter_acc,
        'std_outter_acc': iter_std_outter_acc,
    })

    all_outter_results.append(outter_results_df)
    all_inner_results.append(inner_results_df)

# ----------------  FINAL SUMMARY ACROSS ITERATIONS ----------------#

all_runs_df =pd.DataFrame(all_runs_results)
all_outter_results_df = pd.concat(all_outter_results, ignore_index=True)
all_inner_results_df = pd.concat(all_inner_results, ignore_index=True)

all_runs_df.to_csv('all_runs_summary_ENLR_NMC.csv', index=False)
all_outter_results_df.to_csv('all_outter_results_ENLR_NMC.csv', index=False)
all_inner_results_df.to_csv('all_inner_results_ENLR_NMC.csv', index=False)






