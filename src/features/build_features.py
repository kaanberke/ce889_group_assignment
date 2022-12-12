from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class FeatureConstructor(object):
    def __init__(self,
                 data: pd.DataFrame,
                 target: str = None,
                 cat_features: list = None,
                 cont_features: list = None) -> None:
        self.data = data
        self.target = target
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.embedding_sizes = None

        assert self.data is not None, "Data is not provided"
        assert self.cat_features is not None or self.cont_features is not None, "At least one of the features should " \
                                                                                "be specified "

        if self.cat_features is None:
            self.cat_features = [
                col for col in self.data.columns
                if col not in self.cont_features and col != self.target
            ]

        if self.cont_features is None:
            self.cont_features = [
                col for col in self.data.columns
                if col not in self.cat_features and col != self.target
            ]

        assert len(self.cat_features) + len(self.cont_features) + 1 == len(
            self.data.columns), "Features are not specified correctly"

    def construct(self) -> pd.DataFrame:
        # Copies data to avoid changing the original one
        df = self.data.copy()

        # Construct categorical features
        df, categorical_code_dict = self._construct_cat_features(df)

        # for col in self.cat_features:
        #     code_dict = categorical_code_dict[col]
        #     code_fillna_value = len(code_dict)
        #     df[col] = df[col].map(code_dict).fillna(code_fillna_value).astype(np.int64)

        if self.target is not None:
            df[self.target] = df[self.target].astype(np.float32)

        # Checks the number of categorical feature sizes
        cat_sizes = [len(df[col].unique()) for col in self.cat_features]
        # print(f"Number of unique values in categorical features: {cat_sizes}")

        # Creates embedding sizes for categorical features
        self.embedding_sizes = [(size, min(50, (size + 1) // 2))
                                for size in cat_sizes]
        # print(f"Embedding sizes: {self.embedding_sizes}")

        # Normalizes continuous features
        df = self.normalize(df)

        df[self.cat_features] = df[self.cat_features].apply(
            LabelEncoder().fit_transform)

        return df, categorical_code_dict

    def _construct_cat_features(self, df: pd.DataFrame) -> tuple:
        categorical_code_dict = {}
        for col in df.columns:
            if col in self.cat_features:
                df[col] = df[col].astype("category")
                df, categorical_code_dict = self._encode_categorical(
                    df, col, categorical_code_dict)
        return df, categorical_code_dict

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        scaler.fit(df[self.cont_features])
        df[self.cont_features] = scaler.transform(df[self.cont_features])
        df[self.cont_features] = df[self.cont_features].astype(np.float32)
        return df

    @staticmethod
    def split(df: pd.DataFrame) -> tuple:
        train, val = train_test_split(df, test_size=0.25)
        val, test = train_test_split(val, test_size=0.2)
        return train, val, test

    @staticmethod
    def _encode_categorical(df, col, categorical_code_dict):
        if col not in categorical_code_dict:
            col_dict = {}
            for i, cat in enumerate(df[col].cat.categories):
                col_dict[cat] = i
            categorical_code_dict[col] = col_dict
        df[col] = df[col].map(categorical_code_dict[col])
        return df, categorical_code_dict
