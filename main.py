import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import metrics, model_selection
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from InputHelper import InputHelper


def main():
    print('starting program')
    # parse csv
    df = pd.read_csv('fake_job_postings.csv')

    # wrangle data
    df = tokenize_job_title_col(df)
    # create new column is_in_usa based off address column
    df['is_in_usa'] = df['location'].str.split(',').str[0] == 'US'
    df['is_in_usa'] = df['is_in_usa'].astype(bool)
    df['fraudulent'] = df['fraudulent'].astype(bool)
    # df.loc[df['is_in_usa'] == True, 'fraudulent'] = False

    # print(sum(df['fraudulent'] == True))
    # randomly make half of the jobs fraudulent
    # df['fraudulent'] = [random.getrandbits(3) for i in df.index]
    dfupdate=df.sample(frac=0.5)
    dfupdate['fraudulent']=1
    df.update(dfupdate)
    # print(sum(df['fraudulent'] == True))

    # create ML model
    X = df[['telecommuting', 'has_company_logo',
            'has_questions', 'employment_type', 'is_in_usa']]
    y = df['fraudulent']

    # evaluate accuracy
    model = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,test_size=.8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # print(accuracy)

    # get user input
    ih = InputHelper()
    input = [ih.remote, ih.has_logo, ih.has_logo,
             ih.employment_type, ih.is_in_usa]
    input = np.array(input)
    input = input.reshape(1, -1)
    y_pred = model.predict(input)

    answer = 'fake' if y_pred == 1 else 'real'
    print(answer)
    print('the job is ' + str(round(accuracy*100, 2)) + '% likely to be ' + answer)

    # visualize data
    sns.set()
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm, cmap='Greens',xticklabels=[0,1],yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # Z = df[['fraudulent','telecommuting', 'has_company_logo',
    #         'has_questions', 'employment_type', 'is_in_usa']]
    # Z['fraudulent'] = Z['fraudulent'].astype(bool)
    # Z['telecommuting'] = Z['telecommuting'].astype(bool)
    # Z['has_company_logo'] = Z['has_company_logo'].astype(bool)
    # Z['has_questions'] = Z['has_questions'].astype(bool)
    # Z['is_in_usa'] = Z['is_in_usa'].astype(bool)
    # sns.relplot(x='fraudulent',y='telecommuting', col='is_in_usa', data=df, kind='line')
    # plt.show()
    # sns.catplot(x='fraudulent',y='has_company_logo', col='is_in_usa', data=df, kind='bar')
    # plt.show()
    # sns.catplot(x='fraudulent',y='has_questions', col='is_in_usa', data=df, kind='bar')
    # plt.show()

def is_in_usa(row):
    if row['location'].split(',')[0] == 'US':
        val = 1
    else:
        val = 0


def tokenize_job_title_col(dataframe):
    dataframe['employment_type'].fillna(5, inplace=True)
    dataframe.replace({'employment_type': {
        'Full-time': 1,
        'Part-time': 2,
        'Contract': 3,
        'Temporary': 4,
        'Other': 5,
    }}, inplace=True)
    dataframe['employment_type'].astype(int)
    return dataframe


if __name__ == '__main__':
    main()
