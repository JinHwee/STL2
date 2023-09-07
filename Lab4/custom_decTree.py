# Answer for Last Question

import pandas as pd

# global variables

# reading in csv data using pandas
data = pd.read_csv('with_new_features.csv')
requiredColumns = ['URL_Length', 'Prefix/Suffix', 'iFrame', 'subDomain', 'DNS_Record', 'Have_At', 'Web_Forwards', 'Label']

dependent = ['Label']
independents = ['URL_Length', 'Prefix/Suffix', 'iFrame', 'subDomain', 'DNS_Record', 'Have_At', 'Web_Forwards']

# defining manual tree creation; legitimate = 0, phishing = 1
def manual_tree(row):
    if row['URL_Length'] <= 0.5:
        return 1
    else:
        if row['Prefix/Suffix'] <= 0.5:
            if row['iFrame'] <= 0.5:
                if row['DNS_Record'] <= 0.5:
                    return 0
                else:
                    return 1
            else:
                if row['subDomain'] <= 0:
                    return 0
                else:
                    if row['Web_Forwards'] <= 0.5:
                        return 0
                    else:
                        return 1
        else:
            if row['subDomain'] <= 0:
                return 0
            else:
                return 1

if __name__ == "__main__":
        # selecting only the columns that are required to build the decision tree
    requiredData = data.loc[:, requiredColumns]

    # select 30% of data as test set
    testSet = requiredData.sample(frac=0.3).reset_index(drop=True)

    ALL_requiredData = testSet.loc[:, independents]
    ALL_labels = list(testSet['Label'])
    
    predicted_labels = []
    for id in range(len(ALL_requiredData)):
        predicted_label = manual_tree(ALL_requiredData.loc[id, :])
        predicted_labels.append(predicted_label)
        
    # accuracy: (tp + tn) / (p + n)
    # precision tp / (tp + fp)
    # recall: tp / (tp + fn)
    tp, fp, tn, fn = 0, 0, 0, 0
    p = ALL_labels.count(0)
    n = ALL_labels.count(1)
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == ALL_labels[i] and predicted_labels[i] == 0:
            tp += 1
        elif predicted_labels[i] == ALL_labels[i] and predicted_labels[i] == 1:
            tn += 1
        elif predicted_labels[i] != ALL_labels[i] and ALL_labels[i] == 0:
            fn += 1
        elif predicted_labels[i] != ALL_labels[i] and ALL_labels[i] == 1:
            fp += 1
            
    print("Accuracy: {:0.3f}".format((tp + tn)/(p+n)))
    print("Precision: {:0.3f}".format(tp/(tp+fp)))
    print("Recall: {:0.3f}".format(tp/(tp+fn)))