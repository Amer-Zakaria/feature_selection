from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import time

@dataclass
class SplitData:

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series   

def data_preprocessing(csv_url:str="data.csv") -> SplitData:
    df = pd.read_csv(csv_url)

    ##keep only numeric columnss
    df_numeric = df.select_dtypes(include=["number"])

    target_name = df_numeric.columns[1]                 ## DATA IS ALWAYS THE FIRST COLUMN (after the index column)
    target = df_numeric[target_name]  
    data   = df_numeric.drop(columns=target_name) 

    ##to remove rows where the target is NaN
    mask = ~np.isnan(target)
    target = target[ mask ]
    data   = data[ mask ]

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25, random_state=42
    )

    ##impute missing values to help analysis
    imputer = KNNImputer()
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=data.columns)
    X_test = pd.DataFrame( imputer.transform(X_test), columns=data.columns )
    return SplitData( X_train=X_train , X_test=X_test , y_train=y_train , y_test=y_test )

def Pearson (df, target) -> pd.Series:
    
    df = df.reset_index(drop=True)
    target = target.reset_index(drop=True)

    ##removerows with NaNs
    aligned = pd.concat([df, target.rename("target")], axis=1).dropna()

    if aligned.empty:
        return pd.Series(0.0, index=df.columns)
    
    cov = aligned.cov(ddof=0)
    cov_target = cov[ "target" ].drop("target")
    std = aligned.std(ddof=0)
    std_target = std["target"]
    std_features = std.drop( "target")
    corr = cov_target/(std_features*std_target) 
    return corr.fillna(0.0).abs().sort_values(ascending=False)  

def distance_correlation(x, y):
    
    
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    ###constant variables mean no correlation
    if np.var(x)==0 or np.var(y)==0:
        return 0  

    a = x - np.mean(x)
    b = y - np.mean(y)

    ###distance matrices using pairwise euclidean distance
    a_dist = np.abs(a[ :,None ]-a[ None,: ])  
    b_dist = np.abs(b[ :,None ]-b[ None,: ]) 

    ###distance covariance and variance
    dcov_ab = np.sum(a_dist*b_dist)/(n**2  )

    dcov_aa = np.sum(a_dist*a_dist)/(n** 2)
    dcov_bb = np.sum(b_dist*b_dist)/(n** 2)

     ##Distance correlation
    if dcov_aa > 0 and dcov_bb > 0:
        return dcov_ab / np.sqrt(dcov_aa * dcov_bb)
    else:
        return 0 # this means no variance

def Distance(df, series) -> pd.Series:
    if len(df) != len(series):
        raise ValueError("data and target aren't equal in  length ")

    df = df.reset_index(drop=True)
    series = series.reset_index(drop=True)

    corr_dict = {}
    for col in df.columns:
        corr_dict[col] = distance_correlation(df[col], series)

    return pd.Series(corr_dict).sort_values(ascending=False)
        # here there is no need to do abs() because distasnce correlation is from 0 to 1

if __name__ == "__main__":
    split = data_preprocessing()

    X = pd.concat([split.X_train, split.X_test])

    target = pd.concat([split.y_train, split.y_test])
    # pearson
    Pstart = time.perf_counter()   
    PearsonResults = Pearson(X, target)
    Pend = time.perf_counter() 

    print("\n\n Elapsed time in seconds using Pearson's correlation:", Pend-Pstart)
    print("\nPearson's best 10 predictive facotrs:\n", PearsonResults[0:10])
    # distance
    Dstart = time.perf_counter()   
    DistanceResults = Distance(X, target)
    Dend = time.perf_counter()

    print("\n\nElapsed time in seconds using distance correlation:", Dend-Dstart)
    print("\nDC's best 10 predictive facotrs:\n", DistanceResults[0:10])
