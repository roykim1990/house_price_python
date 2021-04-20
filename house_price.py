import pandas
import pickle
import numpy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# modelop.init
def begin():
    global lasso_model
    global standard_scaler
    global train_encoded_columns

    # load pickled Lasso linear regression model
    lasso_model = pickle.load(open("lasso.pickle", "rb"))
    # load pickled standard scaler
    standard_scaler = pickle.load(open("standard_scaler.pickle", "rb"))
    # load train_encoded_columns
    train_encoded_columns = pickle.load(open("train_encoded_columns.pickle", "rb"))


# modelop.score
def action(data):
    # converting data into dataframe with some checks
    if isinstance(data, pandas.DataFrame):
        df = data
    else:
        if isinstance(data, list):
            df = pandas.DataFrame(data)
        else:
            df = pandas.DataFrame([data])

    # dictionary to convert values in certain columns
    generic = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0}

    # the only features that the model trained on
    predictive_features = [
        "FullBath",
        "1stFlrSF",
        "TotalBsmtSF",
        "BsmtQual",
        "GarageArea",
        "GarageCars",
        "KitchenQual",
        "ExterQual",
        "GrLivArea",
        "OverallQual",
    ]

    # set aside ground truth to later re-append to dataframe
    ground_truth = df["SalePrice"]

    # limiting features to just the ones the model needs
    df = df[predictive_features]

    # imputing missing values
    for col in predictive_features:
        df.loc[:, col] = df.loc[:, col].fillna("None")

    # converting categorical values from certain features into numerical
    for col in ["BsmtQual", "KitchenQual", "ExterQual"]:
        df.loc[:, col] = df[col].map(generic)

    # scale inputs
    df_ss = standard_scaler.transform(df)

    # generate predictions and rename columns
    df.loc[:, "prediction"] = numpy.round(numpy.expm1(lasso_model.predict(df_ss)), 2)
    df.loc[:, "ground_truth"] = ground_truth

    # MOC expects the action function to be a "yield" function
    yield df.to_dict(orient="records")


# modelop.metrics
def metrics(data):
    # converting data into dataframe with some checks
    if isinstance(data, pandas.DataFrame):
        df = data
    else:
        if isinstance(data, list):
            df = pandas.DataFrame(data)
        else:
            df = pandas.DataFrame([data])

    y = df["ground_truth"]
    y_preds = df["prediction"]

    yield {
        "MAE": mean_absolute_error(y, y_preds),
        "RMSE": mean_squared_error(y, y_preds) ** 0.5,
        "R2": r2_score(y, y_preds),
    }
