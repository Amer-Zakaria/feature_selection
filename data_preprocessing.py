from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def data_preprocessing(csv_url: str = "data.csv") -> SplitData:
    df = pd.read_csv(csv_url)

    # STEP 1: Clean the dataset (accept only numerical values)
    df_numeric = df.select_dtypes(include=["number"])

    # STEP 2: Split into data and the target column(first column)
    target = df_numeric.iloc[:, 0]  # What we're going to predict
    data = df_numeric.iloc[:, 1:]  # What we're going to use to predict

    # STEP 3: Remove rows with missing Target Values first
    mask = ~np.isnan(target)
    data = data[mask]
    target = target[mask]

    # STEP 4: Splitting the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25, random_state=42
    )

    # STEP 5: Impute NAN values
    imputer = KNNImputer()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Convert X_train and X_test back to DataFrames with original column names
    X_train = pd.DataFrame(X_train_imputed, columns=data.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=data.columns)

    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
