import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
):

    # Scalling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    train_std = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    test_std = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # training the model on the training set
    model = LinearRegression()
    model.fit(train_std, y_train.values.ravel())

    # evaluation on the testing set
    y_pred = model.predict(test_std)
    y_true = y_test.values.ravel()
    mse = mean_squared_error(y_true, y_pred)

    return mse
