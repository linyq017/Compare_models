import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # Import seaborn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from lce import LCEClassifier
import pickle
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
import csv

# Define your main folder where you want to store the results
main_folder = '/workspace/data/SGU/SFSI/SFSI/XBG10x_akermark_7class/'

# Check if the main folder exists
if not os.path.exists(main_folder):
    # If it doesn't exist, create it and any missing parent folders
    os.makedirs(main_folder)
    print(f"Main folder '{main_folder}' was created.")
else:
    print(f"Main folder '{main_folder}' already exists.")

# Generate a timestamp to create a unique subfolder
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
sub_folder = os.path.join(main_folder, timestamp)
os.makedirs(sub_folder, exist_ok=True)  # Create the subfolder
print('Created subfolder.')

# Import training data
akermark = pd.read_csv('/workspace/data/SGU/SFSI/SFSI/MASTER_TRAIN.csv', sep= ',', decimal = '.')
akermark = akermark.loc[:, ~akermark.columns.isin(['Unnamed: 0'])]

print('Read input data.')
# Define the number of iterations
num_iterations = 100

# Create dataframes to store results
results_train = pd.DataFrame()
results_test = pd.DataFrame()
precision_recall_fscore_support_train_df = pd.DataFrame()
precision_recall_fscore_support_test_df = pd.DataFrame()

# Create a variable to store the best MCC score
best_mcc_score = -1  # Initialize with a value that guarantees replacement

# Create a variable to store the best model
best_model = None

target_names = ['coarse sed AGR', 'coarse sed Forest', 'fine sed AGR', 'fine sed Forest', 'peat', 'rock', 'till']
# Create dictionaries to store lists of the individual class precision, recall, f1 scores, and support for each class in each iteration
classwise_precision_per_iteration = {class_name: [] for class_name in target_names}
classwise_recall_per_iteration = {class_name: [] for class_name in target_names}
classwise_f1_per_iteration = {class_name: [] for class_name in target_names}
classwise_support_per_iteration = {class_name: [] for class_name in target_names}


# Read random seeds from the text file
with open('/workspace/data/SGU/SFSI/SFSI/LCE10x_akermark_7class/20231122170432/random_seeds.txt', 'r') as file:
    random_seeds = [int(line.strip()) for line in file]

# Loop for multiple iterations
for iteration in range(num_iterations):
    random_seed = random_seeds[iteration]  # Get the seed for this iteration
    
    print(f'Iteration {iteration + 1} of {num_iterations} using seed: {random_seed}')

    # Perform train-test split
    train, test = train_test_split(akermark, test_size=0.2, stratify=akermark['GENERAL_TX'], random_state=random_seed)

    x_train = train.loc[:, [c for c in train.columns if c not in ['GENERAL_TX']]]
    le = LabelEncoder()
    y_train = le.fit_transform(train.loc[:, "GENERAL_TX"])
    d_train = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    
    # Define the two classes you want to subsample
    subsample_classes = ['coarse sed AGR','fine sed AGR']

    # Define the proportion to keep for the subsample classes
    subsample_proportion = 0.102  # 10.2%

    # Create an empty DataFrame for the subsampled test set
    subsampled_test_set = pd.DataFrame(columns=akermark.columns)

    # Iterate through the classes in the test set
    for class_name in test['GENERAL_TX'].unique():
        if class_name in subsample_classes:
            class_data = test[test['GENERAL_TX'] == class_name]
            num_samples_to_select = max(1, int(len(class_data) * subsample_proportion))
            selected_samples = class_data.sample(n=num_samples_to_select, random_state=42)
            subsampled_test_set = pd.concat([subsampled_test_set, selected_samples])
        else:
            intact_class_data = test[test['GENERAL_TX'] == class_name]
            subsampled_test_set = pd.concat([subsampled_test_set, intact_class_data])

    # Create the final test set containing subsampled and intact samples
    test = subsampled_test_set
    test['Geomorphon'] = test['Geomorphon'].astype('int64') # Rearranging test set messed up Geomorphon datatypes
    # Split test into x and y
    x_test = test.loc[:, [c for c in test.columns if c not in ['GENERAL_TX']]]
    y_test = le.fit_transform(test.loc[:, "GENERAL_TX"])
    d_test = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
    print('Length of test set: ', len(test))
    print('Value Counts based on GENERAL_TX:',test['GENERAL_TX'].value_counts())
    print('Data split and label encoding complete. Model training starts')

    # Declare parameters obtained from hyperparameter tuning using bayesian optimization
    params = {'lambda': 3.6797431147409454, 'alpha': 0.16377015271779502, 'booster': 'gbtree', 'max_depth': 11, 'eta': 0.24755202760689674, 'gamma': 1.1552990227831699, 
    'colsample_bytree': 0.8356557257338416, 'min_child_weight': 5,'num_class': 7, 'verbosity':1, 'eval_metric':["mlogloss","merror"]}
    # Initialize and fit the model
    xgb_model = xgb.train(params, d_train)
    y_train_pred = xgb_model.predict(d_train)
    y_test_pred = xgb_model.predict(d_test)
    
    print('Model fitting and prediction complete.')
    
    # Calculate evaluation metrics for training and testing
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    cohen_kappa_train = cohen_kappa_score(y_train, y_train_pred)
    cohen_kappa_test = cohen_kappa_score(y_test, y_test_pred)
    f1_train_w = f1_score(y_train, y_train_pred, average='weighted')
    f1_test_w = f1_score(y_test, y_test_pred, average='weighted')
    f1_train_uw = f1_score(y_train, y_train_pred, average=None)
    f1_test_uw = f1_score(y_test, y_test_pred, average=None)
    mcc_train = matthews_corrcoef(y_train, y_train_pred)
    mcc_test = matthews_corrcoef(y_test, y_test_pred)
    #precision_recall_fscore_support_train = precision_recall_fscore_support(y_train, y_train_pred, average = None)
    #precision_recall_fscore_support_test = precision_recall_fscore_support(y_test, y_test_pred, average = None)
    classification_report_test = classification_report(y_test, y_test_pred, target_names=target_names, digits=3, output_dict=True)

    precision_test, recall_test, f1_score_test, support_test = precision_recall_fscore_support(y_test, y_test_pred, average=None)


    # Assuming target_names is a list of class names
    for i, class_name in enumerate(target_names):
        classwise_precision_per_iteration[class_name].append(precision_test[i])
        classwise_recall_per_iteration[class_name].append(recall_test[i])
        classwise_f1_per_iteration[class_name].append(f1_score_test[i])
        classwise_support_per_iteration[class_name].append(support_test[i])

    # Create dataframes for this iteration's results
    iteration_results_train = pd.DataFrame({
        'Train Accuracy': [train_accuracy],
        'Cohen\'s Kappa (Train)': [cohen_kappa_train],
        'F1 Score (Train) weighted': [f1_train_w],
        'MCC (Train)': [mcc_train],
    })

    # Add separate columns for each class's unweighted F1 score
    for i, class_name in enumerate(target_names):
        iteration_results_train[f'F1_unweighted - {class_name}'] = f1_train_uw[i]

    # Create dataframes for this iteration's results
    iteration_results_test = pd.DataFrame({
        'Test Accuracy': [test_accuracy],
        'Cohen\'s Kappa (Test)': [cohen_kappa_test],
        'F1 Score (Test) weighted': [f1_test_w],
        'MCC (Test)': [mcc_test],
    })

    # Add separate columns for each class's unweighted F1 score
    for i, class_name in enumerate(target_names):
        iteration_results_test[f'F1_unweighted - {class_name}'] = f1_test_uw[i]

    # Concatenate the iteration results to the overall results dataframes
    results_train = pd.concat([results_train, iteration_results_train], ignore_index=True)
    results_test = pd.concat([results_test, iteration_results_test], ignore_index=True)


    print('Results dataframes updated.')

    # Check if the current model has a better MCC score
    if mcc_test > best_mcc_score:
        best_mcc_score = mcc_test
        best_model = xgb_model  # Update the best model with the current model

# Save the best model with pickle
pickle_file_path = os.path.join(sub_folder, 'best_model.pkl')
with open(pickle_file_path, 'wb') as model_file:
    pickle.dump(best_model, model_file)
print('Pickle dumped.')

# Define a path for the seed file within the subfolder
seed_path = os.path.join(sub_folder, 'random_seeds.txt')

# Save the random seeds to a text file in the subfolder
with open(seed_path, 'w') as seed_file:
    seed_file.write('\n'.join(map(str, random_seeds)))
print('Random seeds saved.')

# Save the results to CSV
results_train_path = os.path.join(sub_folder, 'XGB_train_metrics.csv')
results_train.to_csv(results_train_path, index=False)
results_test_path = os.path.join(sub_folder, 'XGB_test_metrics.csv')
results_test.to_csv(results_test_path, index=False)
#precision_recall_fscore_support_test_path = os.path.join(sub_folder, 'XGB_precision_recall_fscore_support_test.csv')
#precision_recall_fscore_support_test.to_csv(precision_recall_fscore_support_test_path, index=False)

print('Results saved to CSV.')

# Create box plots to visualize the distribution of evaluation metrics for training and testing
plt.figure(figsize=(10, 5))

# Training metrics
plt.subplot(1, 2, 1)
plt.ylim(0.4, 0.8)
plt.boxplot([results_train['Cohen\'s Kappa (Train)'], results_train['F1 Score (Train) weighted'], results_train['MCC (Train)']],
            labels=['Cohen\'s Kappa (Train)', 'F1 Score (Train) weighted', 'MCC (Train)'])
plt.title('Training Metrics')

# Testing metrics
plt.subplot(1, 2, 2)
plt.ylim(0.4, 0.8)
plt.boxplot([results_test['Cohen\'s Kappa (Test)'], results_test['F1 Score (Test) weighted'], results_test['MCC (Test)']],
            labels=['Cohen\'s Kappa (Test)', 'F1 Score (Test) weighted', 'MCC (Test)'])
plt.title('Testing Metrics')

plt.tight_layout()

# Save the plots as images
box_plot_path = os.path.join(sub_folder, 'XGB10x_metrics_box_plots.png')
plt.savefig(box_plot_path)
print('Boxplot saved.')

## Create DataFrames from dictionaries
df_f1 = pd.DataFrame(classwise_f1_per_iteration)
df_precision = pd.DataFrame(classwise_precision_per_iteration)
df_recall = pd.DataFrame(classwise_recall_per_iteration)
df_support = pd.DataFrame(classwise_support_per_iteration)

# Specify file paths for saving CSV files
csv_f1_path = os.path.join(sub_folder,'classwise_f1_scores.csv')
csv_precision_path = os.path.join(sub_folder,'classwise_precision_scores.csv')
csv_recall_path = os.path.join(sub_folder,'classwise_recall_scores.csv')
csv_support_path = os.path.join(sub_folder,'classwise_support_values.csv')

# Save DataFrames to CSV files
df_f1.to_csv(csv_f1_path, index=False)
df_precision.to_csv(csv_precision_path, index=False)
df_recall.to_csv(csv_recall_path, index=False)
df_support.to_csv(csv_support_path, index=False)

print(f"Results saved to {csv_f1_path}, {csv_precision_path}, {csv_recall_path}, and {csv_support_path}.")

# Plot box plots for class-wise F1 scores
plt.figure(figsize=(10, 6))
df_f1.boxplot(rot=45, sym='k+', grid=False)
plt.title('Class-wise F1 Scores Box Plot')
plt.ylabel('F1 Score')
plt.xlabel('Class Name')
plt.tight_layout()
plt.show()

# Save the figure to a PNG file
box_plot_path = os.path.join(sub_folder, 'unweighted_f1_score_box_plot.png')
plt.savefig(box_plot_path)
print('Classwise f1 score box plot saved')
