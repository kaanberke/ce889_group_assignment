#%% LIBRARIES
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model

from src.features.build_features import FeatureConstructor

#%% CONSTANTS
MODEL_NO = 13

CAT_FEATURES = [
    "Store",
    "Promo",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "DayOfWeek",
    "Season",
    "IsOpenOnPublicHoliday",
]

CONT_FEATURES = [
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    # "Customers_by_Store_mean",
    # "Customers_by_DayOfWeek_mean"
]

#%% DATA LOADING AND PREPROCESSING

train_df = pd.read_csv("./data/processed/test.csv")

# Feature Engineering
fc = FeatureConstructor(data=train_df,
                        cat_features=CAT_FEATURES,
                        cont_features=CONT_FEATURES)

df, categorical_code_dict = fc.construct()

id_column = df["Id"].values.reshape(-1, 1)
X = df.drop(columns=["Id"], axis=1).values
X = np.expand_dims(X, axis=-1)


#%% MODEL LOADING

model = load_model(f"./models/model_{MODEL_NO}.hdf5")

y_pred = model.predict(X, verbose=1)

y_pred_denormalized = (np.exp(y_pred) - 1) * 0.985

submission = np.concatenate((id_column, y_pred_denormalized), axis=1).astype(int)
submission_df = pd.DataFrame(submission, columns=["Id", "Sales"])
submission_df.to_csv(f"./data/submission_{MODEL_NO}.csv", index=False)
