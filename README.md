# house_price_python
A sample data science project that uses a Lasso Linear Regression Python model to predict house price from the Ames Housing Data dataset. Specifically, this example is used to demonstrate the creating of ModelOp Center-compliant code.

## Assets:
- `lasso.pickle` is the trained model artifact.
- `train_encoded_columns.pickle` is a binarized list of final column names that the model will accept.
- `standard_scaler.pickle` is a `sklearn.preprocessing.StandardScaler` transformer object that is fit on the training data.
- The datasets used for **scoring** are `df_baseline.json` and `df_sample.json`. These datasets represent raw data that would first be run into a batch scoring job.
- The datasets used for **metrics** are `df_baseline_scored.json` and `df_sample_scored.json`. These datasets represent data that has gone through the scoring process, meaning that the data is already transformed into model-ready input and that the predictions for each row are stored in the `prediction` column. Furthermore, the `SalePrice` column contains the actual sale price.
- The dataset used for **training** is `house_price_data.csv`.
- The `input_schema.avsc` file is an AVRO-compliant json file that details the input schema, as needed for ModelOp Center Monitoring (out-of-the-box) functionality.


## Scoring Jobs

### Sample Inputs

Choose **one** of
 - `df_baseline.json`
 - `df_sample.json`

### Schema Checking

Schema Checking is **disabled** in model code (slots in-use).

### Sample Output

The output of the scoring job when the input data is `df_sample.json` is a JSONS file (one-line JSON records). Here are the first two output records:
```json
{"Id":564,"MSSubClass":50,"MSZoning":"RL","LotFrontage":66.0,"LotArea":21780,"Street":"Pave","Alley":null,"LotShape":"Reg","LandContour":"Lvl","Utilities":"AllPub","LotConfig":"Inside","LandSlope":"Gtl","Neighborhood":"Edwards","Condition1":"Norm","Condition2":"Norm","BldgType":"1Fam","HouseStyle":"1.5Fin","OverallQual":6,"OverallCond":7,"YearBuilt":1918,"YearRemodAdd":1950,"RoofStyle":"Gable","RoofMatl":"CompShg","Exterior1st":"Wd Sdng","Exterior2nd":"Wd Sdng","MasVnrType":"None","MasVnrArea":0.0,"ExterQual":"TA","ExterCond":"TA","Foundation":"BrkTil","BsmtQual":"Gd","BsmtCond":"TA","BsmtExposure":"Mn","BsmtFinType1":"Unf","BsmtFinSF1":0,"BsmtFinType2":"Unf","BsmtFinSF2":0,"BsmtUnfSF":1163,"TotalBsmtSF":1163,"Heating":"GasA","HeatingQC":"Ex","CentralAir":"Y","Electrical":"SBrkr","1stFlrSF":1163,"2ndFlrSF":511,"LowQualFinSF":0,"GrLivArea":1674,"BsmtFullBath":0,"BsmtHalfBath":0,"FullBath":2,"HalfBath":0,"BedroomAbvGr":4,"KitchenAbvGr":1,"KitchenQual":"TA","TotRmsAbvGrd":8,"Functional":"Typ","Fireplaces":1,"FireplaceQu":"Gd","GarageType":"Detchd","GarageYrBlt":1955.0,"GarageFinish":"Fin","GarageCars":2,"GarageArea":396,"GarageQual":"TA","GarageCond":"TA","PavedDrive":"N","WoodDeckSF":72,"OpenPorchSF":36,"EnclosedPorch":0,"3SsnPorch":0,"ScreenPorch":144,"PoolArea":0,"PoolQC":null,"Fence":null,"MiscFeature":null,"MiscVal":0,"MoSold":7,"YrSold":2008,"SaleType":"WD","SaleCondition":"Normal","SalePrice":185000}
{"Id":507,"MSSubClass":60,"MSZoning":"RL","LotFrontage":80.0,"LotArea":9554,"Street":"Pave","Alley":null,"LotShape":"IR1","LandContour":"Lvl","Utilities":"AllPub","LotConfig":"Inside","LandSlope":"Gtl","Neighborhood":"SawyerW","Condition1":"Norm","Condition2":"Norm","BldgType":"1Fam","HouseStyle":"2Story","OverallQual":8,"OverallCond":5,"YearBuilt":1993,"YearRemodAdd":1994,"RoofStyle":"Gable","RoofMatl":"CompShg","Exterior1st":"VinylSd","Exterior2nd":"VinylSd","MasVnrType":"BrkFace","MasVnrArea":125.0,"ExterQual":"Gd","ExterCond":"TA","Foundation":"PConc","BsmtQual":"Gd","BsmtCond":"TA","BsmtExposure":"No","BsmtFinType1":"GLQ","BsmtFinSF1":380,"BsmtFinType2":"Unf","BsmtFinSF2":0,"BsmtUnfSF":397,"TotalBsmtSF":777,"Heating":"GasA","HeatingQC":"Ex","CentralAir":"Y","Electrical":"SBrkr","1stFlrSF":1065,"2ndFlrSF":846,"LowQualFinSF":0,"GrLivArea":1911,"BsmtFullBath":0,"BsmtHalfBath":0,"FullBath":2,"HalfBath":1,"BedroomAbvGr":3,"KitchenAbvGr":1,"KitchenQual":"Gd","TotRmsAbvGrd":8,"Functional":"Typ","Fireplaces":1,"FireplaceQu":"TA","GarageType":"Attchd","GarageYrBlt":1993.0,"GarageFinish":"RFn","GarageCars":2,"GarageArea":471,"GarageQual":"TA","GarageCond":"TA","PavedDrive":"Y","WoodDeckSF":182,"OpenPorchSF":81,"EnclosedPorch":0,"3SsnPorch":0,"ScreenPorch":0,"PoolArea":0,"PoolQC":null,"Fence":null,"MiscFeature":null,"MiscVal":0,"MoSold":9,"YrSold":2006,"SaleType":"WD","SaleCondition":"Normal","SalePrice":215000}
```

## Metrics Jobs

Model code includes a metrics function used to compute `R2`, `RMSE`, and `MAE` metrics. The metrics function expectes a dataframe with at least the followwing columns: `prediction` (score) and `SalePrice` (ground truth).

### Sample Inputs

Choose **one** of
 - `df_baseline_scored.json`
 - `df_sample_scored.json`


## Training Jobs

Model Code includes a training function used to train a model binary, along with other dependencies (encoded columns, standard scaler).

### Sample Inputs

Use the following CSV file as input to the training Job:
 - `house_price_data.csv`

### Output Files

In order to be able to download the three pickle files that are written by the training function, add **all** of the following files as outputs to the training job:
 - `lasso.pickle`
 - `standard_scaler.pickle`
 - `train_encoded_columns.pickle`
