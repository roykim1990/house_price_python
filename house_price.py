import pandas
import pickle
import numpy


# modelop.init
def begin():
    global lasso_model
    global train_encoded_columns

    # load pickled Lasso linear regression model
    lasso_model = pickle.load(open("lasso.pickle", "rb"))
    # load train_encoded_columns
    train_encoded_columns = pickle.load(open("train_encoded_columns.pickle", "rb"))


# modelop.score
def action(data):
    # Turn data into DataFrame
    df = pandas.DataFrame([data])

    predictive_features = [
        "MSSubClass",
        "MSZoning",
        "LotFrontage",
        "LotArea",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YearRemodAdd",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "MasVnrArea",
        "ExterQual",
        "ExterCond",
        "Foundation",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinSF1",
        "BsmtFinType2",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "Heating",
        "HeatingQC",
        "CentralAir",
        "Electrical",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "KitchenQual",
        "TotRmsAbvGrd",
        "Functional",
        "Fireplaces",
        "FireplaceQu",
        "GarageType",
        "GarageYrBlt",
        "GarageFinish",
        "GarageCars",
        "GarageArea",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MiscVal",
        "MoSold",
        "YrSold",
        "SaleType",
        "SaleCondition",
    ]

    categorical_features = [
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "ExterQual",
        "ExterCond",
        "Foundation",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Heating",
        "HeatingQC",
        "CentralAir",
        "Electrical",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "SaleType",
        "SaleCondition",
    ]
    
    ground_truth = df['SalePrice']
    
    df = df[predictive_features]
    
    # imputing missing values
    for col in predictive_features:
        if col in categorical_features:
            df.loc[:, col] = df.loc[:, col].fillna("None")
        else:
            df.loc[:, col] = df.loc[:, col].fillna(0)

    # one-hot encode
    df = pandas.get_dummies(df, columns=categorical_features)

    # filling in any missing encoded columns with 0s
    for col in train_encoded_columns:
        if col not in df.columns:
            df[col] = numpy.zeros(df.shape[0])

    # restricting columns to only be final list of encoded columns
    df = df[train_encoded_columns]
    
    # saving prediction and ground_truth to original dataframe
    df["prediction"] = lasso_model.predict(df)
    df['ground_truth'] = ground_truth

    # MOC expects the action function to be a "yield" function
    yield df.to_dict(orient="records")


# modelop.metrics
def metrics(data):
    pass