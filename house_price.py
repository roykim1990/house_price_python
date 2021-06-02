import pandas
import pickle
import numpy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


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


# modelop.train
def train(data):
    # load data
    df = pandas.DataFrame(data)

    # set aside ground truth to later re-append to dataframe
    y_train = df["SalePrice"]

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
    f = lambda x: 1 if x > 0 else 0
    df["eHasPool"] = df["PoolArea"].apply(f)
    df["eHasGarage"] = df["GarageArea"].apply(f)
    df["eHasBsmt"] = df["TotalBsmtSF"].apply(f)
    df["eHasFireplace"] = df["Fireplaces"].apply(f)
    df["eHasRemodeling"] = (df["YearRemodAdd"] - df["YearBuilt"] > 0).astype(int)
    df["eTotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["eTotalBathrooms"] = (
        df["FullBath"]
        + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"]
        + (0.5 * df["BsmtHalfBath"])
    )
    df["eOverallQual_TotalSF"] = df["OverallQual"] * df["eTotalSF"]

    # final list of encoded columns
    train_encoded_columns = [
        "eOverallQual_TotalSF",
        "OverallQual",
        "eTotalSF",
        "GrLivArea",
        "ExterQual",
        "KitchenQual",
        "GarageCars",
        "eTotalBathrooms",
        "BsmtQual",
        "GarageArea",
        "TotalBsmtSF",
        "GarageFinish",
        "YearBuilt",
        "eHasGarage",
        "TotRmsAbvGrd",
        "eHasRemodeling",
        "FireplaceQu",
        "MasVnrArea",
        "eHasFireplace",
        "eHasBsmt",
    ]

    # saving the final list of encoded columns
    pickle.dump(train_encoded_columns, open("train_encoded_columns.pickle", "wb"))

    # choosing only the final list of encoded columns
    X_train = df[train_encoded_columns]

    # standard scale data and pickle scaler
    standard_scaler = StandardScaler()
    X_train_ss = standard_scaler.fit_transform(X_train)
    pickle.dump(standard_scaler, open("standard_scaler.pickle", "wb"))

    # apply log to distribution of y-values
    y_train_log = numpy.log1p(y_train)

    # train and pickle model artifact
    lasso = LassoCV(max_iter=1000)
    lasso.fit(X_train_ss, y_train_log)
    pickle.dump(lasso, open("lasso.pickle", "wb"))

    yield


# modelop.score
def action(data):
    # turning data into a dataframe
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
    # for local testing, we use "return" to visualize the output
    # yield df.to_dict(orient='records')
    return df.to_dict(orient="records")


# modelop.metrics
def metrics(data):
    # converting data into dataframe
    df = pandas.DataFrame(data)

    y = df["ground_truth"]
    y_preds = df["prediction"]

    output_metrics = {
        "MAE": mean_absolute_error(y, y_preds),
        "RMSE": mean_squared_error(y, y_preds) ** 0.5,
        "R2": r2_score(y, y_preds),
    }

    # MOC expects the metrics function to be a "yield" function
    # yield output_metrics
    return output_metrics
