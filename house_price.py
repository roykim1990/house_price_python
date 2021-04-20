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

    # set aside ground truth to later re-append to dataframe
    ground_truth = df["SalePrice"]

    # dictionaries to convert values in certain columns
    generic = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0}
    fireplace_quality = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    garage_finish = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}

    # imputations
    df.loc[:, "GarageYrBlt"] = df.loc[:, "GarageYrBlt"].fillna(df["YearBuilt"])
    for col in ["GarageFinish", "BsmtQual", "FireplaceQu"]:
        df.loc[:, col] = df.loc[:, col].fillna("None")
    # the rest of NaNs will be filled with 0s - end model only uses numerical features
    for col in df.columns:
        df[col] = df[col].fillna(0)

    # converting categorical values from certain features into numerical
    for col in ["BsmtQual", "KitchenQual", "ExterQual"]:
        df.loc[:, col] = df[col].map(generic)
    df.loc[:, "GarageFinish"] = df["GarageFinish"].map(garage_finish)
    df.loc[:, "FireplaceQu"] = df["FireplaceQu"].map(fireplace_quality)

    # feature engineering
    f = lambda x: bool(1) if x > 0 else bool(0)
    df["eHasPool"] = df["PoolArea"].apply(f)
    df["eHasGarage"] = df["GarageArea"].apply(f)
    df["eHasBsmt"] = df["TotalBsmtSF"].apply(f)
    df["eHasFireplace"] = df["Fireplaces"].apply(f)
    df["eHasRemodeling"] = df["YearRemodAdd"] - df["YearBuilt"] > 0
    df["eTotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["eTotalBathrooms"] = (
        df["FullBath"]
        + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"]
        + (0.5 * df["BsmtHalfBath"])
    )
    df["eOverallQual_TotalSF"] = df["OverallQual"] * df["eTotalSF"]

    # limiting features to just the ones the model needs
    df = df[train_encoded_columns]

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
