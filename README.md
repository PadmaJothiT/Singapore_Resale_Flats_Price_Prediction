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
#### Data Collection - Data collected from the public source from 1990 to 2024.Collected data will be combined and merged to a single dataframe. Checking the data types and shape of the combined data.
#### Feature Engineering - Added additional columns using the extracted data and removing the unwanted columns that is duplicated as additional columns.Splitting the data to years and months and storey range column splitting into start storey range and end storey range and removing the original data.Changing the object data types columns into int data types as it contains the numeric values. 
#### Filling Null Values - Null values are presented in remaining_lease and block column filling those with median and mode values.
#### Unique values -  Exploring the unique values of abject data types columns as it contains the duplicate values.In that town, flat_type, street_name, flat_model are the object columns. Flat_type and Flat_model contains the duplicate values as the characters are in case sensitive.Changing the values to the common unique values.
#### Encoding - Encoding categorical variables with Label Encoding for the columns like town,flat model,flat type and street_name.
#### Finding Outliers and Skewness - Finding outliers using boxplot and skewness using distplot, removing outliers inter quartile range and removing skewness using numpy logarithm transform.
#### Scaling - Scaling the all columns uding Standard Scaler
#### Model Evaluation - Model Evaluation three models has been evaluated for the accurate prediction of the sale price.
