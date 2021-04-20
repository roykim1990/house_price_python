# house_price_python

A Lasso Linear Regression Python model to predict house price.
The model was trained on the Ames Housing Data dataset.


Assets:
- `lasso.pickle` is the trained model artifact.
- `train_encoded_columns` is a binarized list of final column names that the model will accept.
- `standard_scaler` is a `sklearn.preprocessing.StandardScaler` transformer object that is fit on the training data.  
- Sample inputs to the scoring function are included (`df_baseline.json` and `df_sample.json`)

The metrics function expects a DataFrame with at least the following two columns: `prediction` and `ground_truth` (actual values).
Sample inputs to the metrics function are included (`df_baseline_scored.json` and `df_sample_scored.json`)

The output of the scoring job when the input data is `df_sample.json` is a JSONS file (one-line JSON records). Here are the first three output records
```json
{"Id":607,"prediction":148658.460602766,"ground_truth":152000,"FullBath":1,"1stFlrSF":1040,"TotalBsmtSF":1040,"BsmtQual":"Gd","GarageArea":576,"GarageCars":2,"KitchenQual":"Gd","ExterQual":"TA","GrLivArea":1040,"OverallQual":5}
{"Id":1211,"prediction":193698.3182707051,"ground_truth":189000,"FullBath":2,"1stFlrSF":1055,"TotalBsmtSF":1055,"BsmtQual":"Gd","GarageArea":462,"GarageCars":2,"KitchenQual":"Gd","ExterQual":"Gd","GrLivArea":1845,"OverallQual":6}
{"Id":493,"prediction":163408.0352850238,"ground_truth":172785,"FullBath":2,"1stFlrSF":728,"TotalBsmtSF":728,"BsmtQual":"Gd","GarageArea":429,"GarageCars":2,"KitchenQual":"TA","ExterQual":"Gd","GrLivArea":1456,"OverallQual":6}
```
