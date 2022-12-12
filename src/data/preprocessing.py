import argparse
import logging
from pprint import pprint

import numpy as np
import pandas as pd
import os
from pandas_profiling import ProfileReport
from pathlib import Path
# from featurewiz import Groupby_Aggregator
from typing import Union


class DataProcessor:
    """
    DataProcessor class helps for reading, processing, and writing data from the Rossmann Store Sales
    dataset. It is used to create the data files that are used by the model.
    """

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.store_data = None
        self.processed_train_data = None
        self.processed_test_data = None
        self.logger = logging.getLogger(__name__)

    def read_data(self,
                  train_data_path: str,
                  test_data_path: str,
                  store_data_path: str,
                  verbose: bool = False,
                  report_output_path: str = None) -> None:
        """
        Read raw data into DataProcessor.
        :param train_data_path: Path to train data.
        :param test_data_path: Path to test data.
        :param store_data_path: Path to store data.
        :param verbose: Whether to print out data information.
        :param report_output_path: Path to output report to.
        """

        self.logger.info(
            f"Reading data from {train_data_path} , {test_data_path} and {store_data_path}."
        )

        for data_path in [train_data_path, test_data_path, store_data_path]:
            self.path_checker(data_path,
                              mkdir=False,
                              extension=".csv",
                              is_file=True)

        try:
            self.train_data = pd.read_csv(train_data_path, low_memory=False)
            self.test_data = pd.read_csv(test_data_path, low_memory=False)
            self.store_data = pd.read_csv(store_data_path, low_memory=False)

            if verbose:
                term_size = os.get_terminal_size()

                print("=" * term_size.columns)

                self.logger.info("Train Data:")
                print(self.train_data.head())
                print("=" * term_size.columns)

                self.logger.info("Dataframe shape: {}".format(
                    self.train_data.shape))
                print("=" * term_size.columns)

                self.logger.info("Dataframe columns:")
                pprint(self.train_data.columns)
                print("=" * term_size.columns)

                self.logger.info("Dataframe info:")
                print(self.train_data.info())
                print("=" * term_size.columns)

                self.logger.info("Dataframe describe:")
                print(self.train_data.describe())
                print("=" * term_size.columns)

                self.logger.info("Dataframe null values:")
                print(self.train_data.isnull().sum())
                print("=" * term_size.columns)

                self.logger.info("Dataframe duplicates:")
                print(self.train_data.duplicated().sum())
                print("=" * term_size.columns)

                self.logger.info("Dataframe unique values:")
                print(self.train_data.nunique())
                for column in self.train_data:
                    print(
                        f"{column} column unique values: {self.train_data[column].unique()}"
                    )
                print("=" * term_size.columns)

                self.logger.info("Dataframe value counts:")
                print(self.train_data.value_counts())
                print("=" * term_size.columns)
                print("─" * term_size.columns)
                print("=" * term_size.columns)

                self.logger.info("Store Data:")
                print(self.store_data.head())
                print("=" * term_size.columns)

                self.logger.info("Store info:")
                print(self.store_data.info())
                print("=" * term_size.columns)

                self.logger.info("Store describe:")
                print(self.store_data.describe())
                print("=" * term_size.columns)

                self.logger.info("Store null values:")
                print(self.store_data.isnull().sum())
                print("=" * term_size.columns)

            self.logger.info(u"\u2713 Data successfully read.")

        except FileNotFoundError:
            self.logger.error("File not found. Please check the path.")

    def process_data(self,
                     verbose: bool = False,
                     report_output_path: str = None) -> None:
        """
        Processes raw data into processed data.
        :param verbose: Whether to print out data information.
        :param report_output_path: Path to output report to.
        """

        # Checks if the data has been read successfully.
        for data in [self.train_data, self.test_data, self.store_data]:
            assert data is not None, "Data is not loaded."

        self.logger.info("Processing data.")

        # Column names of store data that contain null values.
        na_cols = [
            "Promo2SinceWeek", "Promo2SinceYear", "CompetitionOpenSinceYear",
            "CompetitionOpenSinceMonth"
        ]

        # Replaces null values with 0.
        for na_col in na_cols:
            self.store_data[na_col] = self.store_data[na_col].fillna(
                self.store_data[na_col].mean()).astype(int)

        # Replaces null values in CompetitionDistance column with mean of column.
        self.store_data["CompetitionDistance"].fillna(
            self.store_data["CompetitionDistance"].mean(), inplace=True)

        # CompetitionOpenSinceYear and CompetitionOpenSinceMonth must be renamed to year and month in order to be
        # converted to datetime.
        self.store_data.rename(columns={
            "CompetitionOpenSinceYear": "YEAR",
            "CompetitionOpenSinceMonth": "MONTH"
        },
                               inplace=True)
        # Converts CompetitionOpenSinceYear and CompetitionOpenSinceMonth to datetime.
        self.store_data["CompetitionOpenSinceDate"] = pd.to_datetime(
            self.store_data[["YEAR", "MONTH"]].assign(DAY=1))

        # Drops columns that are not needed.
        self.store_data.drop(["YEAR", "MONTH"], axis=1, inplace=True)

        # Replace null values in PromoInterval column with "0".
        self.store_data["PromoInterval"].fillna("0", inplace=True)

        # Merge train and store data.
        self.processed_train_data = self.train_data.merge(self.store_data,
                                                          on="Store",
                                                          how="inner")
        self.processed_test_data = self.test_data.merge(self.store_data,
                                                        on="Store",
                                                        how="inner")

        promo_interval_dict = {
            "0": 0,
            "Jan,Apr,Jul,Oct": 1,
            "Feb,May,Aug,Nov": 2,
            "Mar,Jun,Sept,Dec": 3
        }

        store_type_dict = {"a": 1, "b": 2, "c": 3, "d": 4}

        # Places processed data variables and iterates over it to get a shallow copy
        df_list = [self.processed_train_data, self.processed_test_data]

        for i, df in enumerate(df_list):
            processed_data = df.copy()
            # # Removes rows if store is closed on a given day because there is no sales for only training data.
            # if i == 0:
            #    processed_data = processed_data[processed_data["Open"] == 1]
            # Removes whole Open column because it is not needed anymore.
            # Reminder: inplace cannot be used because processed_data is a copy of self.processed_*_data.
            # processed_data = processed_data.drop("Open", axis=1)
            processed_data["Open"] = processed_data["Open"].fillna(1)

            # Converts CompetitionOpenSinceDate column to months.
            processed_data["CompetitionOpenSinceMonth"] = (
                pd.to_datetime(processed_data["Date"]) -
                processed_data["CompetitionOpenSinceDate"]).dt.days // 30

            # Converts Date column to datetime for easier manipulation.
            processed_data["Year"] = pd.DatetimeIndex(
                processed_data["Date"]).year
            processed_data["Season"] = pd.DatetimeIndex(
                processed_data["Date"]).month % 12 // 3 + 1
            processed_data["Month"] = pd.DatetimeIndex(
                processed_data["Date"]).month
            processed_data["WeekOfYear"] = pd.DatetimeIndex(
                processed_data["Date"]).isocalendar().week.values
            processed_data["Day"] = pd.DatetimeIndex(
                processed_data["Date"]).day

            # Creates new column for whether the store is open on a public holiday.
            processed_data["IsOpenOnPublicHoliday"] = np.where(
                processed_data["StateHoliday"].isin(["a", "b", "c"]), 1, 0)

            # Removes StateHoliday column because it is not needed anymore.
            processed_data = processed_data.drop("StateHoliday", axis=1)

            # Applies mapping to PromoInterval, StoreType, and Assortment columns and converts them to integers.
            processed_data["PromoInterval"] = processed_data[
                "PromoInterval"].map(promo_interval_dict).astype(int)
            processed_data["StoreType"] = processed_data["StoreType"].map(
                store_type_dict).astype(int)
            processed_data["Assortment"] = processed_data["Assortment"].map(
                store_type_dict).astype(int)

            # Resets index.
            processed_data = processed_data.reset_index(drop=True)

            # Removes Date column because it is not needed anymore.
            processed_data = processed_data.drop("Date", axis=1)

            # Removes CompetitionOpenSinceDate column because it is not needed anymore.
            processed_data = processed_data.drop("CompetitionOpenSinceDate",
                                                 axis=1)

            try:
                # Removes Customers column because it is not provided on test data.
                processed_data = processed_data.drop("Customers", axis=1)
            except KeyError:
                pass

            df_list[i] = processed_data

        self.processed_train_data, self.processed_test_data = df_list

        # Checks if there are any null values in the processed data.
        assert self.processed_train_data.isnull().sum().sum(
        ) == 0, "Null values still exist in processed train data."
        assert self.processed_test_data.isnull().sum().sum(
        ) == 0, "Null values still exist in processed test data."
        """
        gba_categoricals = ["Store", "DayOfWeek"]
        gba_aggregates = ["mean"]
        gba_numberics = ["Customers"]

        gba = Groupby_Aggregator(
            categoricals=gba_categoricals,
            aggregates=gba_aggregates,
            numerics=gba_numberics
        )

        gba_train = gba.fit_transform(self.processed_train_data)
        gba_train.drop(["Customers", "DayOfWeek"], axis=1, inplace=True)

        gba_test = gba.transform(self.processed_test_data)
        gba_test.drop("DayOfWeek", axis=1, inplace=True)

        self.processed_train_data = gba_train
        self.processed_test_data = gba_test
        """

        self.logger.info(u"\u2713 Data successfully processed.")

        if verbose:
            term_size = os.get_terminal_size()

            self.logger.info("Processed train data:")
            print(self.processed_train_data.head())
            print("=" * term_size.columns)

            self.logger.info("Processed test data:")
            print(self.processed_test_data.head())
            print("=" * term_size.columns)

            if report_output_path is None:
                report_output_path = Path(
                    "../../reports/processed_data_report.html")
                report_output_path.parent.mkdir(parents=True, exist_ok=True)

            profile_report_title = "Dataset Profiling Report"
            # Create a pandas profiling report.
            profile = ProfileReport(
                self.processed_train_data,
                title=profile_report_title,
                html={"style": {
                    "full_width": True
                }},
            )
            profile.to_file(output_file=str(report_output_path))

            self.logger.info(
                u"\u2713 {} successfully created and saved to {}".format(
                    profile_report_title, report_output_path))

    def write_data(self,
                   processed_train_data_path: str = None,
                   processed_test_data_path: str = None) -> None:
        """
        Write processed data to csv files.
        :param processed_train_data_path: Path to write the processed train data.
        :param processed_test_data_path: Path to write the processed test data.
        """

        assert processed_train_data_path is not None, "Processed train data path is not specified."
        assert processed_test_data_path is not None, "Processed test data path is not specified."

        self.path_checker(Path(processed_train_data_path).parent, mkdir=True)
        self.path_checker(Path(processed_test_data_path).parent, mkdir=True)

        self.logger.info("Writing data.")

        self.processed_train_data.to_csv(processed_train_data_path,
                                         index=False)
        self.processed_test_data.to_csv(processed_test_data_path, index=False)
        self.logger.info(u"\u2713 Data successfully written.")

    def path_checker(self,
                     data_path: Union[str, Path],
                     mkdir: bool = False,
                     extension: str = None,
                     is_file: bool = False) -> None:
        """Check if data path exists and has the correct extension."""

        data_path = Path(data_path)

        if is_file:
            assert data_path.exists(), f"Data path {data_path} is not a file."

        if extension is not None:
            assert data_path.suffix == extension, f"Data path {data_path} does not end with {extension}."

        if mkdir and not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {data_path}")


if __name__ == "__main__":
    # Creates argument parser.
    parser = argparse.ArgumentParser()

    # Adds input arguments to the parser.
    parser.add_argument("--input_train",
                        default="./data/raw/train.csv",
                        help="Path to raw train data.",
                        type=str)
    parser.add_argument("--input_test",
                        default="./data/raw/test.csv",
                        help="Path to raw test data.",
                        type=str)
    parser.add_argument("--input_store",
                        default="./data/raw/store.csv",
                        help="Path to raw store data.",
                        type=str)

    # Adds output arguments to the parser.
    parser.add_argument("--output_train",
                        default="./data/processed/train.csv",
                        help="Path to save processed train data.",
                        type=str)
    parser.add_argument("--output_test",
                        default="./data/processed/test.csv",
                        help="Path to save processed test data.",
                        type=str)

    # Parses arguments.
    args = parser.parse_args()

    # Creates data processor object.
    data_processor = DataProcessor()
    # Reads data based on the input arguments.
    data_processor.read_data(args.input_train,
                             args.input_test,
                             args.input_store,
                             verbose=True)
    # Processes data.
    data_processor.process_data(verbose=False)
    # Writes data based on the output arguments.
    data_processor.write_data(args.output_train, args.output_test)
