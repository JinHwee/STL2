import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# reading in the csv file that had been cleaned
df = pd.read_csv('./ExternalFiles/for_corelation.csv')

# selecting only the top 5 columns identified via Excel corelation
column_names = ['dttl', 'sttl', 'ackdat', 'tcprtt', 'dload', 'label']
df = df.loc[:, column_names]

# selecting all benign rows
benign_rows = df.loc[df['label'] == 0]

# selecting all malicious (attack) rows
attack_rows = df.loc[df['label'] == 1]

selected_benign_rows = benign_rows.sample(n=500)
selected_attack_rows = attack_rows.sample(n=500)

selected_benign_rows.to_csv('./Lab3/benign_500_rows.csv', index=False)
selected_attack_rows.to_csv('./Lab3/attack_500_rows.csv', index=False)

# train test split
x_names = ['dttl', 'sttl', 'ackdat', 'tcprtt', 'dload']
y_names = ['label']

x = df.loc[:, x_names]
y = df.loc[:, y_names]

x_train, x_test, y_train, y_test = train_test_split(x, y)
train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1)

train_data.to_csv('./Lab3/train_data.csv', index=False)
test_data.to_csv('./Lab3/test_data.csv', index=False)