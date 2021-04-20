# house_price_python

A sample data science project that uses a Lasso Linear Regression Python model to predict house price from the Ames Housing Data dataset. Specifically, this example is used to show the how to make ModelOp Center-compliant code.


Assets:
- `lasso.pickle` is the trained model artifact.
- `train_encoded_columns.pickle` is a binarized list of final column names that the model will accept.
- `standard_scaler.pickle` is a `sklearn.preprocessing.StandardScaler` transformer object that is fit on the training data.  
- Sample inputs to the scoring function are included (`df_baseline.json` and `df_sample.json`)

The metrics function expects a DataFrame with at least the following two columns: `prediction` and `ground_truth` (actual values).
Sample inputs to the metrics function are included (`df_baseline_scored.json` and `df_sample_scored.json`)

The output of the scoring job when the input data is `df_sample.json` is a JSONS file (one-line JSON records). Here are the first three output records
```json
{"Id":564,"prediction":168353.66,"ground_truth":185000,"eOverallQual_TotalSF":17022,"OverallQual":6,"eTotalSF":2837,"GrLivArea":1674,"ExterQual":2,"KitchenQual":2,"GarageCars":2,"eTotalBathrooms":2.0,"BsmtQual":3,"GarageArea":396,"TotalBsmtSF":1163,"GarageFinish":3,"YearBuilt":1918,"eHasGarage":true,"TotRmsAbvGrd":8,"eHasRemodeling":true,"FireplaceQu":4,"MasVnrArea":0.0,"eHasFireplace":true,"eHasBsmt":true}
{"Id":507,"prediction":234576.12,"ground_truth":215000,"eOverallQual_TotalSF":21504,"OverallQual":8,"eTotalSF":2688,"GrLivArea":1911,"ExterQual":3,"KitchenQual":3,"GarageCars":2,"eTotalBathrooms":2.5,"BsmtQual":3,"GarageArea":471,"TotalBsmtSF":777,"GarageFinish":2,"YearBuilt":1993,"eHasGarage":true,"TotRmsAbvGrd":8,"eHasRemodeling":true,"FireplaceQu":3,"MasVnrArea":125.0,"eHasFireplace":true,"eHasBsmt":true}
{"Id":656,"prediction":121785.16,"ground_truth":88000,"eOverallQual_TotalSF":9702,"OverallQual":6,"eTotalSF":1617,"GrLivArea":1092,"ExterQual":2,"KitchenQual":2,"GarageCars":1,"eTotalBathrooms":1.5,"BsmtQual":2,"GarageArea":264,"TotalBsmtSF":525,"GarageFinish":1,"YearBuilt":1971,"eHasGarage":true,"TotRmsAbvGrd":6,"eHasRemodeling":false,"FireplaceQu":0,"MasVnrArea":381.0,"eHasFireplace":false,"eHasBsmt":true}
```
