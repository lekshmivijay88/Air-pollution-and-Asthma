import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer

class ProcessData:

    @staticmethod
    def combine_data_files_from_all_sources():
        source_folder_names = ["KCL", "AURN", "AQE", "MISC"]
        borough_names = ["Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "City of London",
                         "City of Westminster", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
                         "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow",
                         "Islington", "Kensington and Chelsea", "Kingston", "Lambeth", "Lewisham", "Merton", "Newham",
                         "Redbridge", "Richmond", "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest",
                         "Wandsworth"]
        combined_dir = os.getcwd() + "\\data\\Combined"
        if not os.path.exists(combined_dir):
            os.mkdir(combined_dir)
        count = 0
        for borough_name in borough_names:
            combined_df = pd.DataFrame()
            for folder_name in source_folder_names:
                file_path = Path(os.getcwd() + "\\data\\" + folder_name + "\\" + borough_name + ".csv")
                if file_path.is_file():
                    temp_df = pd.read_csv(file_path)
                    if temp_df.columns[0] == "":
                        temp_df.drop([0], inplace=True)
                    temp_df["borough"] = borough_name
                    combined_df = pd.concat([combined_df, temp_df])
            combined_filename = Path(os.getcwd() + "\\data\\Combined\\" + borough_name + ".csv")
            combined_df.to_csv(combined_filename, index=False)
            count = count + 1
        print("Saved combined file count: ", count)

    @staticmethod
    def create_resampled_data_file():
        dir_path = os.getcwd() + "\\data\\Processed"
        dir_list = os.listdir(dir_path)
        full_df = pd.DataFrame()
        count = 0
        for file in dir_list:
            if file.endswith(".csv"):
                df = pd.read_csv(dir_path + "\\" + file)
                df["date"] = pd.to_datetime(df['date'], errors="raise", format="%Y/%m/%d %H:%M")
                df.set_index("date", drop=True, inplace=True)
                df_resampled = df.resample('D').agg("mean")
                df_resampled["borough"] = file.split(".csv")[0]
                df_resampled.reset_index(inplace=True)
                full_df = pd.concat([full_df, df_resampled])
                count = count + 1

        resampled_filename = os.getcwd() + "\\data\\ResampledDataForEDA.csv"
        full_df.to_csv(resampled_filename, index=False)
        print("Saved resampled file count: ", count)

    @staticmethod
    def create_missing_value_file_for_combined_files():
        dir_path = os.getcwd() + "\\data\\Combined"
        ratio_df = pd.DataFrame()
        counter = 0
        dir_list = os.listdir(dir_path)
        for file in dir_list:
            if file.endswith(".csv"):
                df = pd.read_csv(dir_path + "\\" + file)
                df["date"] = pd.to_datetime(df['date'], errors="raise", format="%d/%m/%Y %H:%M")
                df.set_index("date", drop=True, inplace=True)
                df_resampled = df.resample('H').agg("mean")
                df_resampled.reset_index(inplace=True)
                df = df_resampled
                df.drop("date", axis=1, inplace=True)
                temp_df = pd.DataFrame()
                temp_df["files"] = [file]

                for colname in df.columns.values:
                    temp_df[colname] = [(df[colname].isnull().sum()/len(df[colname])*100)]

                ratio_df = pd.concat([ratio_df, temp_df])
                counter = counter + 1

        ratio_df.to_csv(os.getcwd() + "\\data\\MissingRatioCombinedFiles.csv")
        print(f"Saved missing value ratio file for {counter} files")

    @staticmethod
    def handle_outliers_and_missing_values():
        file_list = ["Barking and Dagenham", "Bexley", "Camden", "City of Westminster", "Ealing", "Greenwich",
                     "Haringey", "Harrow", "Hillingdon", "Kensington and Chelsea", "Lewisham", "Richmond", "Southwark",
                     "Tower Hamlets"]
        processed_dir = os.getcwd() + "\\data\\Processed"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)
        dir_path = os.getcwd() + "\\data\\Combined\\"
        count = 0
        for borough in file_list:
            file_path = dir_path + borough + ".csv"
            df = pd.read_csv(file_path)
            df = df[["date", "borough", "nox", "no2", "no", "o3", "pm10", "pm2.5", "ws", "wd", "air_temp"]]

            df["date"] = pd.to_datetime(df['date'], errors="raise", format="%d/%m/%Y %H:%M")
            df.set_index("date", drop=True, inplace=True)
            df_resampled = df.resample('H').agg("mean")
            df_resampled.reset_index(inplace=True)
            df = df_resampled
            for column in df.columns:
                if df[column].dtype == 'float':
                    if column == "air_temp":
                        df[column] = np.where((df[column] > -20), (df[column] * 1.8) + 32, np.nan)

                    df[column] = np.where((df[column] > 0), df[column], np.nan)
                    Q3, Q1 = np.nanpercentile(df[column], [75, 25])
                    IQR = Q3 - Q1
                    UL = Q3 + 1.5 * IQR
                    LL = Q1 - 1.5 * IQR
                    df[column] = np.where((df[column] > UL) | (df[column] < LL), np.nan, df[column])
                    print(f"Missing value for {column} : {(df[column].isnull().sum() / len(df[column]) * 100)}")

            df_temp = df.drop("date", axis=1)
            df.rename(columns={"pm2.5": "pm25"}, inplace=True)
            knn = KNNImputer()
            df_arr = knn.fit_transform(df_temp)
            df1 = pd.DataFrame(df_arr, columns=["nox", "no2", "no", "o3", "pm10", "pm2.5", "ws", "wd", "air_temp"])
            print("Negative values: ", (df1 < 0).values.any(), "*****", df1.shape)

            count = count + 1
            df1["borough"] = borough
            df1["date"] = df["date"]

            df1 = df1.round(2)
            df1.to_csv(file_path.replace("Combined", "Processed"), index=False)
        print(f"Saved {count} processed files.")

    @staticmethod
    def create_single_file_for_eda():
        dir_path = os.getcwd() + "\\data\\Processed\\"
        full_df = pd.DataFrame()
        count = 0
        for file in os.listdir(dir_path):
            file_path = dir_path + file
            temp_df = pd.read_csv(file_path)
            full_df = pd.concat([full_df, temp_df])
            count = count + 1

        print(f"Writing single data file for {count} boroughs.")
        print(full_df.shape)
        full_df.to_csv(os.getcwd() + "\\data\\DataForEDA.csv", index=False)

    def create_final_data_file(self):
        self.combine_data_files_from_all_sources()
        self.create_missing_value_file_for_combined_files()
        self.handle_outliers_and_missing_values()
        self.create_single_file_for_eda()
        self.create_resampled_data_file()
        self.remove_skewness()

    @staticmethod
    def read_data():
        file = os.getcwd() + "\\data\\ResampledDataForEDA_final.csv"
        csv_data = pd.read_csv(file)
        return csv_data

    @staticmethod
    def preprocess_data(df):
        dataframe = df.drop(["date"], axis=1)
        # Label Encoding
        encoder = preprocessing.LabelEncoder()
        dataframe["borough"] = encoder.fit_transform(dataframe["borough"])
        return dataframe

    @staticmethod
    def normalise_data(df):
        df_scaled = df.drop(["date", "borough"], axis=1)
        cols = df_scaled.columns.values
        print(cols)
        df_scaled = StandardScaler().fit_transform(df_scaled)
        df_scaled = pd.DataFrame(df_scaled, columns=cols)

        cat_col = df["borough"]
        enc_output = pd.get_dummies(cat_col)
        df_result = pd.concat([df_scaled, enc_output], axis=1)
        print("after enc:", df_result.shape)
        df_result = df_result.round(2)
        return df_result

    @staticmethod
    def create_yearly_sample_data():
        file = os.getcwd() + "\\data\\DataForEDA.csv"
        df = pd.read_csv(file, low_memory=True)
        df["date"] = pd.to_datetime(df['date'], errors="raise", format="%d/%m/%Y %H:%M")
        df.set_index("date", drop=True, inplace=True)
        df_resampled = df.groupby("borough").resample('Y').agg("sum")
        df_resampled.reset_index(inplace=True)
        resampled_filename = os.getcwd() + "\\data\\YearlyResampledData.csv"
        print("Writing yearly resampled file: ", resampled_filename)
        print(df_resampled.head(3))
        df_resampled.to_csv(resampled_filename, index=False)

    @staticmethod
    def remove_skewness():
        file = os.getcwd() + "\\data\\ResampledDataForEDA.csv"
        csv_data = pd.read_csv(file)
        df = csv_data.drop(["date", "borough"], axis=1)

        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')
        for i in df.columns:
            array = np.array(df[i]).reshape(-1, 1)
            x = qt.fit_transform(array)
            df[i] = x

        df["date"] = csv_data["date"]
        df["borough"] = csv_data["borough"]
        filename = os.getcwd() + "\\data\\ResampledDataForEDA_final.csv"
        df.to_csv(filename, index=False)
