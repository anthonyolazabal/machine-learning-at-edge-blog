import bentoml
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Dropout  # type: ignore
from keras.optimizers import RMSprop  # type: ignore

print("Checking data files to be imported")
for dirname, _, filenames in os.walk('./ProductionQuality/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("Importing raw data")
x_data = pd.read_csv('./ProductionQuality/data_X.csv', sep=',')
x_cols = list(x_data.columns)
x_data.columns = x_cols
print(x_data.shape)

print("Importing reference quality indexes")
y_train = pd.read_csv('./ProductionQuality/data_Y.csv', sep=',')
print(y_train.shape)

print("Importing sample quality output")
Y_submit = pd.read_csv('./ProductionQuality/sample_submission.csv', sep=',')
print(Y_submit.shape)

print("Merging raw data with reference quality indexes for train dataset")
train_df = x_data.merge(y_train, left_on='date_time', right_on='date_time')
print(train_df.shape)

print("Merging raw data with sample quality indexes for test dataset")
test_df = x_data.merge(Y_submit, left_on='date_time',
                       right_on='date_time').drop('quality', axis=1)
print(test_df.shape)

print("Checking integrity and consistency of train and test datasets")
assert train_df.shape[0] == y_train.shape[0]
assert test_df.shape[0] == Y_submit.shape[0]

print("Cleaning quality column on train dataset")
y = train_df['quality']
train_df.drop(['quality'], axis=1, inplace=True)

print("Cleaning datetime column on train and test dataset")
train_df.drop(['date_time'], axis=1, inplace=True)
test_df.drop(['date_time'], axis=1, inplace=True)

print("Splitting datasets")
X_train, X_test, y_train, y_test = train_test_split(
    train_df, y, test_size=0.33)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
test_df = scaler.transform(test_df)

model = Sequential()
model.add(Dense(17, activation='tanh', input_shape=(17,)))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(34, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

print(model.summary())

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['mse', 'mae'])
batch_size = 64
epochs = 30

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test mse:', score[0])
print('Test mae:', score[2])

bento_model = bentoml.keras.save_model("qualitycheck", model)
print(f"Bento model tag = {bento_model.tag}")

Y_submit['quality'] = model.predict(test_df)
Y_submit.to_csv('submission.csv', index=False)
