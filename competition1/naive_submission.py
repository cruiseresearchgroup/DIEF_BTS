import numpy as np
import pandas as pd
import zipfile

# -----------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------
# This script demonstrates how to prepare a submission file for the "Brick by Brick 2024" competition.
# 
# STEPS:
# 1. Extract filenames from a ZIP file containing test data (.pkl files).
# 2. Use the training labels file (train_y) to determine the expected output columns (the prediction targets).
# 3. Generate a sample submission file, filled with random predictions, to illustrate the required output format.
# 4. Save the sample submission file as a compressed CSV (.csv.gz).
#
# NOTE:
# The random predictions are only placeholders. In an actual submission, replace them with your model's predictions.
#
# For more information on the dataset and required files, visit:
#   https://www.aicrowd.com/challenges/brick-by-brick-2024/dataset_files
#
# Once your submission file is ready, you can submit it at:
#   https://www.aicrowd.com/challenges/brick-by-brick-2024/submissions/new
#
# Make sure to accept the competition's terms and conditions before submitting.
# -----------------------------------------------------------

# -----------------------------------------------------------
# DEFINE FILE PATHS
# -----------------------------------------------------------
# Ensure that the data files (ZIP of test inputs and the train_y file) are placed in the `data` folder.
zip_file_path = 'data/test_X_v0.1.0.zip'    # Path to the ZIP file containing the test data (.pkl files)
train_y_path = 'data/train_y_v0.1.0.csv'    # Path to the training target CSV file
submission_file_path = 'data/sample_submission_v0.1.0'  # Path to save the submission file (compressed CSV)
submission_file_extensions = '.csv.gz'  # Path to save the submission file (compressed CSV)

# -----------------------------------------------------------
# EXTRACT TEST FILENAMES FROM ZIP FILE
# -----------------------------------------------------------
# The test ZIP file typically contains files named like: test_X/<filename>.pkl
# We need a list of these .pkl filenames without the "test_X/" prefix.

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Get the complete list of files inside the ZIP.
    file_list = zip_ref.namelist()

print(f"Found {len(file_list)} files in the test_X ZIP archive.")

# Filter the list to only include '.pkl' files and remove the leading directory prefix.
filenames = [f.replace("test_X/", "") for f in file_list if f.endswith('.pkl')]

# -----------------------------------------------------------
# LOAD TRAINING TARGET DATAFRAME TO DETERMINE OUTPUT COLUMNS
# -----------------------------------------------------------
# The train_y CSV contains:
# - 'filename'
# - 'brick_class_label'
# - A number of target columns that we need to predict for the test set.
df_train_y = pd.read_csv(train_y_path, index_col=0)

(f"Loaded training target data with shape: {df_train_y.shape}")

# Extract all columns except 'filename' and 'brick_class_label'—these are our target columns.
expected_columns = list(df_train_y.columns)
filtered_columns = sorted([col for col in expected_columns if col not in ['filename', 'brick_class_label']])

# Assert that we have the expected number of target columns.
# The competition expects a total of 94 target columns.
assert len(filtered_columns) == 94, f"Expected 94 target columns, found {len(filtered_columns)}"

# -----------------------------------------------------------
# GENERATE SAMPLE PREDICTIONS
# -----------------------------------------------------------
# To demonstrate the submission format, we predictions for each test sample.
# Replace this logic with your model's predictions in a real scenario.

num_files = len(filenames)
# num_files = 315721
num_targets = len(filtered_columns)

# Create a numpy array of shape [num_files, num_targets] with random floats in [0,1].

# ZEROS
sample_predictions_zero = np.zeros((num_files, num_targets)).astype(np.float16)

# ONES
sample_predictions_ones = np.ones((num_files, num_targets)).astype(np.float16)

# RANDOM UNIFORM
sample_predictions_RaUn = np.random.random((num_files, num_targets)).astype(np.float16)

# MODE
# find the mode of the training data
a_count = (df_train_y.values[:,3:]==1).sum(axis=0)
i_mode = np.argmax(a_count)
# set everything to zero except the mode column
sample_predictions_mode = np.zeros((num_files, num_targets)).astype(np.float16)
sample_predictions_mode[:,i_mode] = 1

# RANDOM PROPORTIONAL
sample_predictions_RaPr = np.random.random((num_files, num_targets)).astype(np.float16)
dftrn = df_train_y.values
proportions = (dftrn==1).sum(0)/dftrn.shape[0]
l_tst_H_proportional = []
for i in range(num_targets):
    icolumn = np.random.rand(num_files)
    icolumn = icolumn/2
    icolumn = icolumn + proportions[i]/2
    l_tst_H_proportional.append(icolumn)
sample_predictions_RaPr = np.stack(l_tst_H_proportional).T

# -----------------------------------------------------------
# CREATE THE SUBMISSION DATAFRAME
# -----------------------------------------------------------
# Construct a DataFrame with the same structure as the expected submission.
# It must contain a 'filename' column plus one column for each target.

df_sample_zero = pd.DataFrame(sample_predictions_zero, columns=filtered_columns)
df_sample_ones = pd.DataFrame(sample_predictions_ones, columns=filtered_columns)
df_sample_RaUn = pd.DataFrame(sample_predictions_RaUn, columns=filtered_columns)
df_sample_mode = pd.DataFrame(sample_predictions_mode, columns=filtered_columns)
df_sample_RaPr = pd.DataFrame(sample_predictions_RaPr, columns=filtered_columns)
df_sample_zero.insert(0, 'filename', filenames)  # Insert 'filename' as the first column.
df_sample_ones.insert(0, 'filename', filenames)  # Insert 'filename' as the first column.
df_sample_RaUn.insert(0, 'filename', filenames)  # Insert 'filename' as the first column.
df_sample_mode.insert(0, 'filename', filenames)  # Insert 'filename' as the first column.
df_sample_RaPr.insert(0, 'filename', filenames)  # Insert 'filename' as the first column.

# -----------------------------------------------------------
# SAVE THE SUBMISSION FILE
# -----------------------------------------------------------
# Save as a compressed CSV (gzip) without the index column.
df_sample_zero.to_csv(submission_file_path+'_zero'+submission_file_extensions, index=False, compression='gzip')
df_sample_ones.to_csv(submission_file_path+'_ones'+submission_file_extensions, index=False, compression='gzip')
df_sample_RaUn.to_csv(submission_file_path+'_RaUn'+submission_file_extensions, index=False, compression='gzip')
df_sample_mode.to_csv(submission_file_path+'_mode'+submission_file_extensions, index=False, compression='gzip')
df_sample_RaPr.to_csv(submission_file_path+'_RaPr'+submission_file_extensions, index=False, compression='gzip')
print(f"Sample submission file saved.")

# -----------------------------------------------------------
# SUMMARY & NEXT STEPS
# -----------------------------------------------------------
# After running this script, you will have:
# - A file named `sample_submission_v0.1.0.csv.gz` in the `data` folder.
# - This file contains:
#   * A 'filename' column listing each test .pkl file.
#   * 94 columns of predictions filled with random values.
#
# In a real submission:
# - Replace the random predictions with your model's predictions.
# - Ensure your predictions match the expected order and format.
# - Check the competition page for additional requirements and instructions.
#
# Once ready, submit your .csv.gz file at:
#   https://www.aicrowd.com/challenges/brick-by-brick-2024/submissions/new
#
# Good luck!
# -----------------------------------------------------------
