import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv("student-mat.csv")

data = data[data['G3'] != 0]

X = data.drop(columns="G3")
y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50, random_state = 0)
X_y_train = X_train.copy(deep=True)
X_y_train['y'] = y_train

X_y_train_numeric = X_y_train.select_dtypes(include=['float64', 'int64'])
X_y_train_category = X_y_train.select_dtypes(exclude=['float64', 'int64'])

correlation_matrix = X_y_train_numeric.corr()
corr = correlation_matrix['y']
corr = corr[(corr >=0.1) | (corr <= -0.1)]

numeric_factors = ['age', 'Medu', "Fedu", "traveltime", "studytime", "failures", "goout", "absences", "G1"]

interesting_category_factors = ["Mjob", "Fjob", "schoolsup", "higher"]

mjob_target_encoder = TargetEncoder()
mjob_target_encoder.fit(X_y_train['Mjob'], X_y_train['y'])

fjob_target_encoder = TargetEncoder()
fjob_target_encoder.fit(X_y_train['Fjob'], X_y_train['y'])

schoolsup_target_encoder = TargetEncoder()
schoolsup_target_encoder.fit(X_y_train['schoolsup'], X_y_train['y'])

higher_target_encoder = TargetEncoder()
higher_target_encoder.fit(X_y_train['higher'], X_y_train['y'])

X_y_train['Mjob_encoded'] = mjob_target_encoder.transform(X_y_train['Mjob'])
X_y_train['Fjob_encoded'] = fjob_target_encoder.transform(X_y_train['Fjob'])
X_y_train['schoolsup_encoded'] = schoolsup_target_encoder.transform(X_y_train['schoolsup'])
X_y_train['higher_encoded'] = higher_target_encoder.transform(X_y_train['higher'])

factors_to_use = numeric_factors + ['Mjob_encoded', 'Fjob_encoded', 'schoolsup_encoded', 'higher_encoded']

X_train_encoded = X_y_train.copy(deep=True)
X_train_encoded = X_train_encoded[factors_to_use]

scaler = MinMaxScaler()
scaler.fit(X_train_encoded)

X_train_scale = scaler.transform(X_train_encoded)

model = LinearRegression()
model.fit(X_train_scale, y_train)
print(model.score(X_train_scale, y_train)) #оценка r^2

#сверху мы обучили модель на обучающей выборке, теперь аналогично категориям из обучающей делаем табличку на тестовую выборку.

X_test['y'] = y_test

X_test['Mjob_encoded'] = mjob_target_encoder.transform(X_test['Mjob'])
X_test['Fjob_encoded'] = fjob_target_encoder.transform(X_test['Fjob'])
X_test['schoolsup_encoded'] = schoolsup_target_encoder.transform(X_test['schoolsup'])
X_test['higher_encoded'] = higher_target_encoder.transform(X_test['higher'])

X_test_encoded = X_test[factors_to_use]

X_test_scale = scaler.transform(X_test_encoded)

predictions = model.predict(X_test_scale)

print("Predictions:", predictions)