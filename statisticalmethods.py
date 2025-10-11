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

def data_preprocessing(csv_url: str = "data.csv") -> SplitData:
    df = pd.read_csv(csv_url)

    ### clean the dataset and accept only numerical values
    df_numeric = df.select_dtypes(include=["number"])
 
    ### Split into data and the target column(if the first column isn't the target change the "0")
    target = df_numeric.iloc[:,0]  ###  what we're going to predict
    data = df_numeric.iloc[:,1:]  ###  what we're going to analyze

    ### Remove rows with missing Target Values first
    mask = ~np.isnan(target)
    
    data=data[mask]
    target=target[mask]

    ###Splitting the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.25,random_state=42) 

    ### Impute any nan values
    imputer = KNNImputer()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    ### Convert X_train and X_test back to DataFrames with original column names
    X_train = pd.DataFrame(X_train_imputed, columns=data.columns)
    X_test = pd.DataFrame(X_test_imputed, columns=data.columns)


    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def Pearson(df, series):
    if len(df) != len(series):
        raise ValueError("data and target aren't equal in length")
  
    # calculate mean and deviation
    series_mean = series.mean()
    series_std = series.std() 

    if series_std == 0:
        return pd.Series(0, index=df.columns)
    
    CorrelationResults = {}
    
    #calculate correlations
    for column in df.columns:  
        column_mean = df[column].mean()
        column_std = df[column].std()
        
        if column_std>0:
            # calculate covariance
            covariance = ((df[column] - column_mean) * (series - series_mean)).mean()
            correlation = covariance / (column_std * series_std)
        else:   
            correlation = 0  

        CorrelationResults[column] = correlation
 # return the values sorted from best correlataion to worst for comparison
 # since big negative numbers also mean good correlation we return the absolute vals
    return pd.Series(CorrelationResults).abs().sort_values(ascending=False)

def distance_correlation(x, y):
    
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    #constant variables mean no correlation
    if np.var(x) == 0 or np.var(y) == 0:
        return 0  

    # centering each variable
    a = x - np.mean(x)
    b = y - np.mean(y)

    # distance matrices using pairwise euclidean distance
    a_dist = np.abs(a[:,None]-a[None,:])  
    b_dist = np.abs(b[:,None]-b[None,:]) 

    #Distance covariance
    dcov_ab = np.sum(a_dist * b_dist) / (n**2)
    #Distance variance
    dcov_aa = np.sum(a_dist * a_dist) / (n** 2)
    dcov_bb = np.sum(b_dist * b_dist) / (n** 2)

     #Distance correlation
    if dcov_aa > 0 and dcov_bb > 0:
        return dcov_ab / np.sqrt(dcov_aa * dcov_bb)
    else:
        return 0 # this means no variance

def Distance(df, series):

    if len(df)!=len(series):
        raise ValueError("data and target aren't equal in length")

    CorrelationResults = {}
    
    # Iterate through each column in the DataFrame
    for column in df.columns:
        correlation = distance_correlation(df[column], series)
        CorrelationResults[column] = correlation

    return pd.Series(CorrelationResults).sort_values(ascending=False)
        # here there is no need to do abs() because distance correlation is from 0 to 1

# Example usage
if __name__ == "__main__":
    data = data_preprocessing()
    X = pd.concat([data.X_train, data.X_test])
    
    df = pd.DataFrame(X)
    
    targetName = 'Pass/Fail' ## you can change the name of the target feature here ##
    target = df[targetName]
    
    # calculate correlation functions and the time they took
    Pstart = time.perf_counter()   
    PearsonResults = Pearson(df.drop(columns=[targetName]), target)
    Pend = time.perf_counter() 

    print("\n\nElapsed time in seconds using Pearson's correlation:", Pend-Pstart)
    print("\nPearson's best 10 predictive facotrs:\n", PearsonResults[0:10])
    
    Dstart = time.perf_counter()   
    DistanceResults = Distance(df.drop(columns=[targetName]), target)
    Dend = time.perf_counter()

    print("\n\nElapsed time in seconds using distance correlation:", Dend-Dstart)
    print("\nDC's best 10 predictive facotrs:\n", DistanceResults[0:10])