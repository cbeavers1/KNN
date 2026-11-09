import numpy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
import pandas as pd
import my_knn_classifier

pd.set_option('display.max_columns', None)

income_test_data = pd.read_csv(r"hw1-data/income.test.blind.csv")
income_data = pd.read_csv(r"hw1-data/income.train.5k.csv")
income_dev_data = pd.read_csv(r"hw1-data/income.dev.csv")
target = income_data['target']
target_dev = income_dev_data['target']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
scaler = MinMaxScaler().set_output(transform='pandas')

preprocessor = ColumnTransformer([
    ('num', scaler, ['age', 'hours']),
    ('cat', encoder, ['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])
])

preprocessor.fit(income_data)
processed = preprocessor.transform(income_data)
processed_dev = preprocessor.transform(income_dev_data)
processed_test = preprocessor.transform(income_test_data)

my_knn = my_knn_classifier.MyKNN(41)
my_knn.fit(processed, target)
predictions = my_knn.predict(processed_test)

income_test_data['target'] = predictions
income_test_data.to_csv('income.test.predicted.csv', index=False)



