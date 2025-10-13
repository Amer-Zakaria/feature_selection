"""   NOTE: ensure that the target feature is the first column after the indexing column """
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from typing import List
from collections import Counter
import time

@dataclass
class SplitData:            ## a class to hold the data and its training features
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list
    scaler: object = None

def data_preprocessing(
    csv_url: str = "data.csv", #default file name & path
    test_size: float = 0.25,    #proportion of data to test (rest is training)
    use_scaling: bool = True,    #a statistical scaling of data to enable faster alg performance
    scaling_method: str = "robust",  #  "standard" or "robust" or "minmax"   
    use_pca: bool = False,       #PCA=Principal Component Analysis (reduces number of vars to make analysis faster)
    pca_variance: float = 0.95   #bigger variance leads to less reduction of variables
) -> SplitData:
    
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print("error reading csv file: ", e)
        raise

    ###sellect numerical features
    df_numeric = df.select_dtypes(include=["number"])
    if len(df_numeric.columns) == 0: raise ValueError("error: no numerical features found in the dataset")
    
    ###remove constant columns
    numeric_columns = df_numeric.columns[df_numeric.nunique() > 1]
    df_numeric = df_numeric[numeric_columns]

    ### split into target and features
    target = df_numeric.iloc[:, 1] # TARGET IS ALWAYS THE 1ST COLUMN #
    data = df_numeric.iloc[:, 2:]

    ###remove rows with missing target values
    mask = ~np.isnan(target)
    data = data[mask]
    target = target[mask]

     ###check if we can use stratification
    can_stratify = (target.nunique() < 10)

    ##sort values based on variance (for easier analysis)
    variances = data.var().sort_values(ascending=False)
    selected_features = variances.index.tolist()
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
    if len(X_train) > 8000:  #Use simpler imputation for very large datasets
        imputer = SimpleImputer(strategy='median')
        print("Using median imputation for large dataset")
    else:
        n_neighbors = min(5, max(2, len(X_train) // 20))  #to ensure at least 2 neighbors
        imputer = KNNImputer(n_neighbors=n_neighbors)
        print("Using knn imputation with ",n_neighbors, " neighbors")

    ###impute missing values
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    ###apply PCA for dimensionality reduction if it's requested
    
    if use_pca and X_train_imputed.shape[1] > 2:
        n_components = min(X_train_imputed.shape[0] - 1, X_train_imputed.shape[1])
        if n_components > 1:
            pca = PCA(n_components=min(pca_variance, n_components))
            X_train_imputed = pca.fit_transform(X_train_imputed)
            X_test_imputed = pca.transform(X_test_imputed)
            print("PCA reduced features to ", X_train_imputed.shape[1],
                  " components explaining ", pca.explained_variance_ratio_.sum(), " of variance ")
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
            scaler = StandardScaler() #use if not sure what to choose
        elif scaling_method == "robust":
            scaler = RobustScaler()  #better for many outliers
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        X_train_imputed = scaler.fit_transform(X_train_imputed)
        X_test_imputed = scaler.transform(X_test_imputed)
        print(" applied ",scaling_method," scaling")

    ###convert back to dataframes
    X_train_df = pd.DataFrame(X_train_imputed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_imputed, columns=feature_names)

    print("Final dataset: ",X_train_df.shape[0]," training, ",X_test_df.shape[0]," test samples")
    print("features: ",{X_train_df.shape[1]})
    

    return SplitData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler
    )



###Recursive  Feature Elimination(RFE) method

@dataclass
class RFEResults:
    selected_features: List[str]
    feature_ranking: pd.DataFrame
    n_features_selected: int
    train_rmse: float
    test_rmse: float

def perform_rfe(
    split_data,
    n_features_to_select: int = 1, ###  default num features to keep in the end  ###
    step: int = 1                   ###  default num features to remove at a time for speed  ###
) -> RFEResults:
    
    X_train, X_test, y_train, y_test = (
        split_data.X_train, split_data.X_test, split_data.y_train, split_data.y_test
    )
    
    ###use linear regressionm
    estimator = LinearRegression() ### choose another regrassion (like logistic) if needed ###
    
    ###"Recursive Feature Elimination" method (model-based but classical method)
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=step
    )
    
    rfe.fit(X_train, y_train)
    
    ###what are the selected features remaining
    selected_mask = rfe.support_
    selected_features = X_train.columns[selected_mask].tolist()
    
    ###feature rankings
    feature_ranking = pd.DataFrame({'feature': X_train.columns,'ranking': rfe.ranking_,'selected': rfe.support_}).sort_values('ranking')
    
    ###  calculate performance metrics (R mean squared estimation)
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    estimator.fit(X_train_selected, y_train)
    y_pred_train = estimator.predict(X_train_selected)
    y_pred_test = estimator.predict(X_test_selected)

    train_rmse = mean_squared_error(y_train, y_pred_train)
    test_rmse = mean_squared_error(y_test, y_pred_test)
 
    print(f"Original features: {X_train.shape[1]} ")
    print(f"Selected features: {n_features_to_select}")
    print(f"Step size: {step} ")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print("\nSELECTED FEATURES:")
    for i, feature in enumerate(selected_features, 1):
        print(feature)
    
    print(f"\nTOP 10 FEATURE RANKINGS:")
    top_rankings = feature_ranking.head(10)
    for _, row in top_rankings.iterrows():
        status = "SELECTED" if row['selected'] else "eliminated"
        print(f"  {row['ranking']:2d}. {row['feature']} ({status})")
    
    return RFEResults(
        selected_features=selected_features,
        feature_ranking=feature_ranking,
        n_features_selected=n_features_to_select,
        train_rmse=train_rmse,
        test_rmse=test_rmse
    )

def get_top_features(rfe_results: RFEResults, top_n: int = 0) -> pd.DataFrame:
    
    if top_n == 0: ##not specified
        top_n = rfe_results.n_features_selected
    
    top_features = rfe_results.feature_ranking.head(top_n)
    return top_features[['feature', 'ranking', 'selected']]

if __name__ == "__main__":
    split_data = data_preprocessing("data.csv") ##change if needed##

    Rstart = time.perf_counter()
    rfe_results = perform_rfe(
        split_data=split_data,
        n_features_to_select=10,
        step=1  ## chosse num features to remove at a time for speed (bigger ==> faster) but keep it less than n_features ##
    )
    Rend = time.perf_counter()
    
    X_train_best = split_data.X_train[rfe_results.selected_features]
    X_test_best = split_data.X_test[rfe_results.selected_features]
    
    print("\nRFE method time taken: ",Rend-Rstart," seconds\n\n")

