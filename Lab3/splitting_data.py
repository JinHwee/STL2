import pandas as pd

# reading in the csv file that had been cleaned
df = pd.read_csv('./ExternalFiles/for_corelation.csv')

# selecting only the top 5 columns identified via Excel corelation
column_names = ['sttl', 'dttl', 'dload', 'ackdat', 'dmean', 'label']
df = df.loc[:, column_names]

# selecting all benign rows
benign_rows = df.loc[df['label'] == 0]

# selecting all malicious (attack) rows
attack_rows = df.loc[df['label'] == 1]

selected_benign_rows = benign_rows.sample(n=500)
selected_attack_rows = attack_rows.sample(n=500)

selected_benign_rows.to_csv('./Lab3/benign_500_rows.csv', index=False)
selected_attack_rows.to_csv('./Lab3/attack_500_rows.csv', index=False)
