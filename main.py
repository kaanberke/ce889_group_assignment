# %% LIBRARIES
import logging
import os

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO,
                    datefmt="%H:%M:%S")

logging.info("LIBRARIES BEING LOADED...")

from src.features.build_features import FeatureConstructor
from src.utils import scheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.run_functions_eagerly(True)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    LearningRateScheduler
)
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    Conv1D,
    MaxPooling1D,
    Dropout,
    Flatten
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.metrics import RootMeanSquaredError

logging.info("LIBRARIES IS LOADED SUCCESSFULLY \u2713")

# %% CONSTANTS
TARGET = "Sales"
EPOCHS = 1_000
BATCH_SIZE = 2048
LEARNING_RATE = 1E-3

CAT_FEATURES = [
    "Store",
    "Promo",
    "Open",
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

logging.info("CONSTANT VARIABLES IS SET SUCCESSFULLY \u2713")

# %% Data Loading and Preprocessing

logging.info("DATA IS BEING LOADED AND PREPROCESSING...")

train_df = pd.read_csv("./data/processed/train.csv")

# Feature Engineering
fc = FeatureConstructor(data=train_df,
                        target=TARGET,
                        cat_features=CAT_FEATURES,
                        cont_features=CONT_FEATURES)

df, categorical_code_dict = fc.construct()
"""
mms = MinMaxScaler(y_range[0], y_range[1])
"""
"""
y_min = 0
y_max = df[TARGET].max() * 1.2
"""

# normalize = lambda x: (x - y_min)/(y_max - y_min)
# denormalize = lambda x: x * (y_max - y_min) + y_min

# df[TARGET] = df[TARGET].apply(normalize)
# Denormalization: (np.exp(y)-1)

# Normalizes target column
df[TARGET] = df[TARGET].apply(np.log1p)

logging.info("DATA IS LOADED AND PREPROCESSED SUCCESSFULLY \u2713")

# %% Data split and reshaping

# Splits data into train, validation and test sets
df_train, df_val, df_test = fc.split(df)

# Reshapes data into 3D array
X_train = df_train.drop(TARGET, axis=1).values
X_train = np.expand_dims(X_train, axis=-1)
y_train = df_train[TARGET].values

X_val = df_val.drop(TARGET, axis=1).values
X_val = np.expand_dims(X_val, axis=-1)
y_val = df_val[TARGET].values

X_test = df_test.drop(TARGET, axis=1).values
X_test = np.expand_dims(X_test, axis=-1)
y_test = df_test[TARGET].values

logging.info("DATA IS SPLITTED INTO TRAIN, VAL AND TEST SUCCESSFULLY \u2713")

# %% Model Building

logging.info("MODEL IS BEING BUILT...")

# Asks for model number to be built
model_no = None
saved_models = os.listdir("./models")
while True:
    model_no = input("Please enter the model number: ")
    if not model_no.isdigit():
        print("Please enter a valid model number.")
    elif model_no in [
        i.split("_")[1].split(".")[0] for i in saved_models if i.endswith(".hdf5")
    ]:
        print("Model number already exists. Please try again.")
    else:
        model_no = int(model_no)
        break

# Defines the input layer
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# # Defines the first convolutional layer
x = Conv1D(128, 3, padding="same", activation="relu")(inputs)
x = MaxPooling1D(2, padding="same")(x)

# # Defines the second convolutional layer
# x = Conv1D(128, 3, padding="same", activation="relu")(x)
# x = MaxPooling1D(2, padding="same")(x)

# Defines the first LSTM layer
x = LSTM(128, return_sequences=True)(x)

# Defines the second LSTM layer
x = LSTM(64, return_sequences=False)(x)

# Defines the third LSTM layer
# x = LSTM(32, return_sequences=False)(x)

# Defines the flatten layer
# x = Flatten()(x)

# # Defines the first dense layer
# x = Dense(512, activation="relu", kernel_initializer="uniform")(x)
# x = Dropout(0.5)(x)
#
# # Defines the second dense layer
# x = Dense(1024, activation="relu", kernel_initializer="uniform")(x)

# Defines the output layer
predictions = Dense(1)(x)

# Builds the model
model = Model(inputs=inputs, outputs=predictions, name=f"model_{model_no}")

# Defines the early stopping callback
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=100)

# Defines the model checkpoint callback
checkpoint_filepath = f"./models/model_{model_no}.hdf5"
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                            save_weights_only=False,
                                            monitor="val_loss",
                                            mode="min",
                                            save_best_only=True)

# Defines the tensorboard callback
tensorboard_callback = TensorBoard(log_dir="./logs")

learning_rate_scheduler = LearningRateScheduler(scheduler)

# Concatenates callbacks
callbacks = [
    early_stopping_callback,
    model_checkpoint_callback,
    tensorboard_callback,
    learning_rate_scheduler
]

# Defines the optimizer
rmsprop = RMSprop(learning_rate=LEARNING_RATE)
adam = Adam(learning_rate=LEARNING_RATE)
sgd = SGD(learning_rate=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)

# Creates Root Mean Squared Error object to use as a metric
root_mean_squared_error = RootMeanSquaredError(
    name="root_mean_squared_error", dtype=None
)

# Compiles the model
model.compile(optimizer=rmsprop,
              loss="mean_squared_error",
              metrics=[
                  "mean_absolute_error",
                  root_mean_squared_error
                  ]
)

# Builds the model in order to see the summary
model.build(X_train)

logging.info("MODEL IS BUILT SUCCESSFULLY \u2713")

# %% Model architecture summary

# Prints the model summary
model.summary()

# %% Model training

logging.info("MODEL IS BEING STARTED TO TRAIN...")

# Trains the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    initial_epoch=0,
)

# %% Model evaluation

# Evaluates the model on test set
test_results = model.evaluate(X_test, y_test, batch_size=128)

# %% Model Details and Results

# Saves the model details and results as json file
model_details = {
    "EPOCHS": EPOCHS,
    "STOPPED_EPOCH": early_stopping_callback.stopped_epoch,
    "BEST_VAL_LOSS": f"{early_stopping_callback.best:.5f}",
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "TEST_LOSS": f"{test_results[0]:.5f}",
    "TEST_MAE": f"{test_results[1]:.5f}",
    "TEST_RMSE": f"{test_results[2]:.5f}",
    "OPTIMIZER": str(model.optimizer.get_config()),
    "MODEL_ARCHITECTURE": json.loads(model.to_json())
}

with open(f"./reports/json/model_{model_no}.json", "w") as f:
    json.dump(model_details, f, indent=4)

# %% Model Testing

# Loads the best model
model = load_model(f"./models/model_{model_no}.hdf5")

# Predicts the test set
y_pred = model.predict(X_test)

# Denormalizes the predicted values and the test values
y_test_denormalized = (np.exp(y_test) - 1)
y_pred_denormalized = (np.exp(y_pred) - 1)

# Prints the test results
print(y_test_denormalized[:5])
print(y_pred_denormalized[:5][:, 0])

# %% Model Visualization

# Plot training and validation loss
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.legend()
plt.savefig(f"./reports/figures/model_{model_no}_loss.png")
plt.show()

# Plot training and validation mean absolute error
plt.plot(history.history["mean_absolute_error"], label="Training MAE")
plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")
plt.legend()
plt.savefig(f"./reports/figures/model_{model_no}_MAE.png")
plt.show()
