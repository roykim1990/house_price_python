# modelop.slot.0: in-use
# modelop.slot.1: in-use


import pandas
import pickle
import numpy
import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


# modelop.init
def begin():
    
    global lasso_model
    global standard_scaler
    global train_encoded_columns

    # Load pickled Lasso linear regression model
    lasso_model = pickle.load(open("lasso.pickle", "rb"))
    # Load pickled standard scaler
    standard_scaler = pickle.load(open("standard_scaler.pickle", "rb"))
    # Load train_encoded_columns
    train_encoded_columns = pickle.load(open("train_encoded_columns.pickle", "rb"))

    logger.info(
        "'lasso.pickle', 'standard_scaler.pickle', and 'train_encoded_columns.pickle' files loaded to respective variables"
    )


# modelop.score
def action(data):
    
    # Turning data into a dataframe
    logger.info("Loading input record into a pandas.DataFrame")
    input_data = pandas.DataFrame([data])

    # Set aside ground truth to later re-append to dataframe
    ground_truth = input_data["SalePrice"]

    # Dictionaries to convert values in certain columns
    generic = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0}
    fireplace_quality = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    garage_finish = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}

    # Imputations
    logger.info("Conforming input dataset to be model-ready")
    input_data.loc[:, "GarageYrBlt"] = input_data.loc[:, "GarageYrBlt"].fillna(input_data["YearBuilt"])
    for col in ["GarageFinish", "BsmtQual", "FireplaceQu"]:
        input_data.loc[:, col] = input_data.loc[:, col].fillna("None")
    # The rest of NaNs will be filled with 0s - end model only uses numerical features
    for col in input_data.columns:
        input_data[col] = input_data[col].fillna(0)

    # Converting categorical values from certain features into numerical
    for col in ["BsmtQual", "KitchenQual", "ExterQual"]:
        input_data.loc[:, col] = input_data[col].map(generic)
    input_data.loc[:, "GarageFinish"] = input_data["GarageFinish"].map(garage_finish)
    input_data.loc[:, "FireplaceQu"] = input_data["FireplaceQu"].map(fireplace_quality)

    # Feature engineering
    f = lambda x: bool(1) if x > 0 else bool(0)
    input_data["eHasPool"] = input_data["PoolArea"].apply(f)
    input_data["eHasGarage"] = input_data["GarageArea"].apply(f)
    input_data["eHasBsmt"] = input_data["TotalBsmtSF"].apply(f)
    input_data["eHasFireplace"] = input_data["Fireplaces"].apply(f)
    input_data["eHasRemodeling"] = input_data["YearRemodAdd"] - input_data["YearBuilt"] > 0
    input_data["eTotalSF"] = input_data["TotalBsmtSF"] + input_data["1stFlrSF"] + input_data["2ndFlrSF"]
    input_data["eTotalBathrooms"] = (
        input_data["FullBath"]
        + (0.5 * input_data["HalfBath"])
        + input_data["BsmtFullBath"]
        + (0.5 * input_data["BsmtHalfBath"])
    )
    input_data["eOverallQual_TotalSF"] = input_data["OverallQual"] * input_data["eTotalSF"]

    # Limiting features to just the ones the model needs
    logger.info("Selecting columns that model is expecting")
    input_data = input_data[train_encoded_columns]

    # Scale inputs
    logger.info("Scaling data with pickled standard scaler")
    df_ss = standard_scaler.transform(input_data)

    # generate predictions and rename columns
    logger.info("Generating predictions with the model and appending onto DataFrame")
    input_data.loc[:, "prediction"] = numpy.round(numpy.expm1(lasso_model.predict(df_ss)), 2)
    input_data.loc[:, "SalePrice"] = ground_truth

    # MOC expects the action function to be a "yield" function
    yield input_data.to_dict(orient="records")


# modelop.metrics
def metrics(metrics_data):
    
    logger.info("metrics_data is of shape: %s", metrics_data.shape)

    logger.info("Grabbing relevant columns to calculate metrics")
    y = metrics_data["SalePrice"]
    y_preds = metrics_data["prediction"]

    logger.info("Computing MAE, RMSE, R2 scores")
    output_metrics = {
        "MAE": mean_absolute_error(y, y_preds),
        "RMSE": mean_squared_error(y, y_preds) ** 0.5,
        "R2": r2_score(y, y_preds),
    }
    
    logger.info("Metrics job complete!")

    # MOC expects the metrics function to be a "yield" function
    yield output_metrics


# modelop.train
def train(training_data):
    
    logger.info("training_data is of shape: %s", training_data.shape)
    
    # Set aside ground truth to later re-append to dataframe
    y_train = training_data["SalePrice"]

    # Dictionaries to convert values in certain columns
    generic = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0}
    fireplace_quality = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    garage_finish = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}

    # Imputations
    logger.info("Imputing Nulls")
    training_data.loc[:, "GarageYrBlt"] = training_data.loc[:, "GarageYrBlt"].fillna(training_data["YearBuilt"])
    for col in ["GarageFinish", "BsmtQual", "FireplaceQu"]:
        training_data.loc[:, col] = training_data.loc[:, col].fillna("None")
    # The rest of NaNs will be filled with 0s - end model only uses numerical features
    for col in training_data.columns:
        training_data[col] = training_data[col].fillna(0)

    # Converting categorical values from certain features into numerical
    logger.info("Converting categorical values to numerical values")
    for col in ["BsmtQual", "KitchenQual", "ExterQual"]:
        training_data.loc[:, col] = training_data[col].map(generic)
    training_data.loc[:, "GarageFinish"] = training_data["GarageFinish"].map(garage_finish)
    training_data.loc[:, "FireplaceQu"] = training_data["FireplaceQu"].map(fireplace_quality)

    # Feature engineering
    logger.info("Creating new features with feature engineering")
    f = lambda x: 1 if x > 0 else 0
    training_data["eHasPool"] = training_data["PoolArea"].apply(f)
    training_data["eHasGarage"] = training_data["GarageArea"].apply(f)
    training_data["eHasBsmt"] = training_data["TotalBsmtSF"].apply(f)
    training_data["eHasFireplace"] = training_data["Fireplaces"].apply(f)
    training_data["eHasRemodeling"] = (training_data["YearRemodAdd"] - training_data["YearBuilt"] > 0).astype(int)
    training_data["eTotalSF"] = training_data["TotalBsmtSF"] + training_data["1stFlrSF"] + training_data["2ndFlrSF"]
    training_data["eTotalBathrooms"] = (
        training_data["FullBath"]
        + (0.5 * training_data["HalfBath"])
        + training_data["BsmtFullBath"]
        + (0.5 * training_data["BsmtHalfBath"])
    )
    training_data["eOverallQual_TotalSF"] = training_data["OverallQual"] * training_data["eTotalSF"]

    # Final list of encoded columns
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

    # Saving the final list of encoded columns
    logger.info("Pickling final list of columns for model to predict with")
    
    # Pickle file should be written to outputDir/    
    with open("outputDir/train_encoded_columns.pickle", "wb") as columns_file:
        pickle.dump(train_encoded_columns, columns_file)

    # Choosing only the final list of encoded columns
    X_train = training_data[train_encoded_columns]
    
    # Standard scale data and pickle scaler
    standard_scaler = StandardScaler()
    X_train_ss = standard_scaler.fit_transform(numpy.array(X_train))
    logger.info("Pickling trained standard scaler object")
    
    # Pickle file should be written to outputDir/    
    with open("outputDir/standard_scaler.pickle", "wb") as scaler_file:
        pickle.dump(standard_scaler, scaler_file)
    
    # Apply log to distribution of y-values
    y_train_log = numpy.log1p(y_train)

    # Train and pickle model artifact
    logger.info("Fitting LASSO model")
    lasso = LassoCV(max_iter=1000)
    lasso.fit(X_train_ss, y_train_log)
    logger.info("Pickling trained LASSO model")

    # Pickle file should be written to outputDir/   
    with open("outputDir/lasso.pickle", "wb") as lasso_file:
        pickle.dump(lasso, lasso_file)

    logger.info("Training job complete!")
    pass