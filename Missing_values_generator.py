import pandas as pd
import numpy as np
import os

def generate_missing_values(data, missing_rate, missing_type):
    # Reshape the data
    data_reshaped = data.copy()
    n, t, d = data_reshaped.shape
    data_reshaped = data_reshaped.reshape(n * t, d)

    num_indices = round(n*t*missing_rate)

    if missing_type == 'MCAR':
        current_missing_rate = 0
        missing_rate2 = missing_rate

        for p in range(data_reshaped.shape[1]):
            missing_indices = np.random.choice(range(0, n*t), num_indices, replace=False)
            data_reshaped[missing_indices, p] = np.nan
            #data_reshaped_copy = data_reshaped.copy()
            #data_reshaped_copy[missing_indices, p] = np.nan
            #non_nan_values = data_reshaped_copy[:, p][~np.isnan(data_reshaped_copy[:, p])]


    elif missing_type == 'MAR':
        counter = 0
        # Generate missing indices for MAR iteratively

        current_missing_rate = 0
        missing_rate2 = missing_rate

        while current_missing_rate < missing_rate and missing_rate2 < 1.0:
            observed_variables = np.random.choice(data_reshaped.shape[1], 3, replace=False)
            missing_variables = np.setdiff1d(range(data_reshaped.shape[1]), observed_variables)
            counter += 1
            if counter > 100:
                missing_rate2 += 0.05
                counter = 0

            for q in observed_variables:
                low_val = np.nanpercentile(data_reshaped[:, q], missing_rate2 / 2)
                high_val = np.nanpercentile(data_reshaped[:, q], 100 - (missing_rate2 / 2))
                indices = np.where((data_reshaped[:, q] <= low_val) | (data_reshaped[:, q] >= high_val))[0]

                indices_array = np.array(indices)
                if(len(indices_array) != 0):
                    for j in missing_variables:
                        data_reshaped_copy = data_reshaped.copy()
                        data_reshaped_copy[indices_array, j] = np.nan
                        non_nan_values = data_reshaped_copy[:, j][~np.isnan(data_reshaped_copy[:, j])]
                        distinct_values = np.unique(non_nan_values)

                        if len(distinct_values) >= 2:
                            data_reshaped[indices_array, j] = np.nan
                        else:
                            continue

                        current_missing_rate = np.sum(np.isnan(data_reshaped)) / (n * t * d)
                        if current_missing_rate >= missing_rate:
                            break


    elif missing_type == 'MNAR':
        # Generate missing indices for MNAR
        counter = 0
        # Generate missing indices for MAR iteratively

        current_missing_rate = 0
        missing_rate2 = missing_rate

        while current_missing_rate < missing_rate and missing_rate2 < 1.0:
            counter += 1
            if counter > 100:
                missing_rate2 += 0.05
                counter = 0

            for p in range(data_reshaped.shape[1]):
                low_val = np.nanpercentile(data_reshaped[:, p], missing_rate / 2)
                high_val = np.nanpercentile(data_reshaped[:, p], 100 - (missing_rate / 2))
                indices = np.where((data_reshaped[:, p] <= low_val) | (data_reshaped[:, p] >= high_val))[0]
                indices_array = np.array(indices)
                if (len(indices_array) != 0):
                    data_reshaped_copy = data_reshaped.copy()
                    data_reshaped_copy[indices_array, p] = np.nan
                    non_nan_values = data_reshaped_copy[:, p][~np.isnan(data_reshaped_copy[:, p])]
                    distinct_values = np.unique(non_nan_values)

                    if len(distinct_values) >= 1:
                        data_reshaped[indices_array, p] = np.nan
                    else:
                        continue

                current_missing_rate = np.sum(np.isnan(data_reshaped)) / (n * t * d)
                if current_missing_rate >= missing_rate:
                    break

    data_missing = data_reshaped.reshape(n, t, d)

    return data_missing

folder_path = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/train_groundtruth_no_na'
#folder_path = 'D:/mimic_imputation/dacmi_challenge_code_and_data/data/test/test_groundtruth_no_na'
#output_folder = 'D:/CATSI/data/test_MCAR_80'
output_folder = 'D:/CATSI/data/train_MCAR_80'
missing_rate = 0.8
missing_type = 'MCAR'


for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df = df.set_index('CHARTTIME')  # Make sure 'CHARTTIME' is the index

        df = df.replace('NA', np.nan)
        df.dropna(inplace=True)
        # output_path = os.path.join(ground_truth_folder, filename)
        # df.to_csv(output_path, index=True)
        data_array = df.values.reshape((len(df), 1, -1))
        missing_data = generate_missing_values(data_array, missing_rate, missing_type)
        df_with_missing = pd.DataFrame(missing_data.reshape(len(df), -1), columns=df.columns)
        df_with_missing.index = df.index
        missing_values_count = df_with_missing.isna().sum().sum()
        df_with_missing = df_with_missing.fillna('NA')

        output_path = os.path.join(output_folder, filename)
        df_with_missing.to_csv(output_path, index=True)
        print(missing_values_count)
        print(missing_values_count / (len(df) * 13))

# df = pd.read_csv('D:/mimic_imputation/dacmi_challenge_code_and_data/data/train_groundtruth/1.csv')
# df = df.set_index('CHARTTIME')  # Make sure 'CHARTTIME' is the index
#
# df = df.replace('NA', np.nan)
# data_array = df.values.reshape((len(df),1, -1))
#
# missing_rate = 0.2
# missing_type = 'MNAR'

# missing_data = generate_missing_values(data_array, missing_rate, missing_type)
# df_with_missing = pd.DataFrame(missing_data.reshape(len(df), -1), columns=df.columns)
# df_with_missing.index = df.index
# missing_values_count = df_with_missing.isna().sum().sum()
# df_with_missing = df_with_missing.fillna('NA')


# df_with_missing.to_csv('data_with_missing_values.csv', index=True)
# print(missing_values_count)
# print(missing_values_count/(len(df)*13))