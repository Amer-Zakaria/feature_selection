from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
):
    # training the model on the training set
    model = LinearRegression()
    model.fit(X_train, y_train.values.ravel())

    # evaluation on the testing set
    y_pred = model.predict(X_test)
    y_true = y_test.values.ravel()
    mse = mean_squared_error(y_true, y_pred)

    return mse
