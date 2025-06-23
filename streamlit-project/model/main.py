import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle



def create_model(data):
    X = data.drop(labels=['diagnosis'], axis=1)
    y = data['diagnosis']

    # scale the x for better model performance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # data split for train and test phases
    X_train, X_test , y_train , y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train , y_train)

    # test the model 
    y_pred = model.predict(X_test)
    print(f'The accuracy score of the model is {accuracy_score(y_test, y_pred)}')
    print(f'classification report is as follows\n{classification_report(y_test, y_pred)}')
    return model , scaler



def get_normalized_data():
    data = pd.read_csv('data/wisconsin_breast_cancer.csv')

    # drop the unqualified column
    data = data.drop(labels=['Unnamed: 32', 'id'], axis=1)

    # one hot encoding applied on column 'diagnosis'
    data['diagnosis'] = data['diagnosis'].str.lower()
    data['diagnosis'] = data['diagnosis'].map({'m':1 , 'b':0})

    return data




def main():
    data = get_normalized_data()
    model , scaler = create_model(data)

    with open('model/model.pkl', 'wb') as m:
        pickle.dump(model, m)
    
    with open('model/scale.pkl', 'wb') as s:
        pickle.dump(scaler, s)

if __name__ == '__main__':
    main()
    