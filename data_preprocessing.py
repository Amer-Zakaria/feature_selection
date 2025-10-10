import pandas as pd
from dataclasses import dataclass
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from collections import Counter
import numpy as np


@dataclass
class SplitData:  # a class to hold the data and its training features
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list
    scaler: object = None


def data_preprocessing(
    csv_url: str = "data.csv",  # default file name & path
    test_size: float = 0.25,  # proportion of data to test (rest is training)
    use_scaling: bool = True,  # a statistical scaling of data to enable faster alg performance
    scaling_method: str = "robust",  #  "standard" "robust" or "minmax"
    use_pca: bool = False,  # PCA=Principal Component Analysis (reduces number of vars to make analysis faster)
    pca_variance: float = 0.95,  # bigger variance leads to less reduction of variables
) -> SplitData:

    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print("error reading csv file: ", e)
        raise

    ###sellect numerical features
    df_numeric = df.select_dtypes(include=["number"])
    if len(df_numeric.columns) == 0:
        raise ValueError("error: no numerical features found in the dataset")

    ###remove constant columns
    numeric_columns = df_numeric.columns[df_numeric.nunique() > 1]
    df_numeric = df_numeric[numeric_columns]

    ### split into target and features
    target = df_numeric.iloc[:, 0]  # TARGET IS ALWAYS THE 1ST COLUMN #
    data = df_numeric.iloc[:, 1:]

    ###remove rows with missing target values
    mask = ~np.isnan(target)
    data = data[mask]
    target = target[mask]

    ###check if we can use stratification
    can_stratify = target.nunique() < 10

    ##sort values based on variance (for easier analysis)
    variances = data.var().sort_values(ascending=False)
    selected_features = variances.head(len(data.columns)).index
    data = data[selected_features]

    ###split data with conditional stratification
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=test_size, stratify=target
        )
        print("used stratified sampling")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=test_size
        )
        print("used random sampling (no stratification)")

    ###choose imputation strategy based on data size
    if len(X_train) > 8000:  # Use simpler imputation for very large datasets
        imputer = SimpleImputer(strategy="median")
        print("Using median imputation for large dataset")
    else:
        n_neighbors = min(
            5, max(2, len(X_train) // 20)
        )  # to ensure at least 2 neighbors
        imputer = KNNImputer(n_neighbors=n_neighbors)
        print("Using knn imputation with ", n_neighbors, " neighbors")

    ###impute missing values
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    ###apply PCA for dimensionality reduction if it's requested
    if use_pca and X_train_imputed.shape[1] > 2:
        ###ensure we have enough samples for pca
        n_components = min(X_train_imputed.shape[0] - 1, X_train_imputed.shape[1])
        if n_components > 1:
            pca = PCA(n_components=min(pca_variance, n_components))
            X_train_imputed = pca.fit_transform(X_train_imputed)
            X_test_imputed = pca.transform(X_test_imputed)
            print(
                "PCA reduced features to ",
                X_train_imputed.shape[1],
                " components explaining ",
                pca.explained_variance_ratio_.sum(),
                " of variance ",
            )
            feature_names = [f"PC {i+1}" for i in range(X_train_imputed.shape[1])]
        else:
            feature_names = data.columns.tolist()
            print("PCA skipped because of insufficient samples or features")
    else:
        feature_names = data.columns.tolist()

    ## apply scaling
    scaler = None
    if use_scaling and len(X_train_imputed) > 0:
        if scaling_method == "standard":
            scaler = StandardScaler()  # use if not sure what to choose
        elif scaling_method == "robust":
            scaler = RobustScaler()  # better for many outliers
        else:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()

        X_train_imputed = scaler.fit_transform(X_train_imputed)
        X_test_imputed = scaler.transform(X_test_imputed)
        print(" applied ", scaling_method, " scaling")

    ###convert back to dataframes
    X_train_df = pd.DataFrame(X_train_imputed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_imputed, columns=feature_names)

    print(
        "Final dataset: ",
        X_train_df.shape[0],
        " training, ",
        X_test_df.shape[0],
        " test samples",
    )
    print("features: ", {X_train_df.shape[1]})

    if can_stratify:
        print("Class distribution in training: ", dict(Counter(y_train)))
        print("Class distribution in test: ", {dict(Counter(y_test))})

    return SplitData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )
