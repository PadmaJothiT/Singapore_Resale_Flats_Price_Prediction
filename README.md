# Singapore_Resale_Flats_Price_Prediction
Singapore Resale Flat Prices Predicting

### Singapore Resale Flats Price Prediction Project aims to predict the future sale price using the data from 1990 till now onwards.

## Techniques used in this project:
#### Pandas
#### Numpy
#### Seaborn
#### Matplolib.Pyplot
#### Scikit

## Steps involved in this Project
#### Data Collection - Data collected from public source from 1990 to 2024
#### Feature Engineering - Added additional columns using the extracted data and removing the unwanted columns that is duplicated as additional columns
#### Filling Null Values - Null values are presented in remaining_lease column filling those with median values
#### Finding Outliers and Skewness - Finding outliers using boxplot and skewness using distplot, removing outliers inter quartile range and removing skewness using numpy logarithm transform.
#### Encoding - Encoding categorical variables with Label Encoding for the columns like town,flat model and flat type.
#### Scaling - Scaling the all columns uding Standard Scaler
#### Model Evaluation - Model Evaluation three models has been evaluated for the accurate prediction of the sale price.
