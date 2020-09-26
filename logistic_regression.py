"""
Created on Fri Sep 25 15:29:16 2020

TITANIC DISASTER - LOGISTIC REGRESSION

@author: Marc
"""
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data and visualising
# Training data
train = pd.read_csv('titanic_train.csv')
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
plt.savefig('Plots/train_null_values.png')
plt.close()
sns.countplot(x = 'Survived', data = train, hue = 'Sex', palette = 'RdBu_r')
plt.savefig('Plots/train_survived_gender.png')
plt.close()
sns.countplot(x = 'Survived', data = train, hue = 'Pclass')
plt.savefig('Plots/train_survived_class.png')
plt.close()
sns.displot(train['Age'].dropna(), kde = False, bins = 30)
plt.savefig('Plots/train_passenger_ages.png')
plt.close()
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
plt.savefig('Plots/train_pclass_vs_age.png')
plt.close()
# Testing data
test = pd.read_csv('titanic_test.csv')
sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
plt.savefig('Plots/test_null_values.png')
plt.close()
sns.displot(test['Age'].dropna(), kde = False, bins = 30)
plt.savefig('Plots/test_passenger_ages.png')
plt.close()
sns.boxplot(x = 'Pclass', y = 'Age', data = test)
plt.savefig('Plots/test_pclass_vs_age.png')
plt.close()

# Function for imputing the age based on visualisations
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
# Filling the null values and dropping the Cabin column because it has too little information
# Train
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)
train.drop('Cabin', axis = 1, inplace = True)
train.dropna(inplace = True)
# Test
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis = 1)
test.drop('Cabin', axis = 1, inplace = True)
test.dropna(inplace = True)

# Encoding the Sex and Embarked columns
# Train
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
train = pd.concat([train, sex, embark], axis = 1)
# Test
sex = pd.get_dummies(test['Sex'], drop_first = True)
embark = pd.get_dummies(test['Embarked'], drop_first = True)
test = pd.concat([test, sex, embark], axis = 1)

# Saving the Name column for future reference
name = test['Name']

# Dropping the columns which are not useful to the logistic model
train.drop(['PassengerId', 'Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
test.drop(['PassengerId', 'Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)

# Splitting the data
X_train = train.drop('Survived', axis = 1)
Y_train = train['Survived']
X_test = test

# Creating and fitting the logistic model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# Showing the results
# In a dataset
frame = {'Name': name, 'Survived': predictions}
results = pd.DataFrame(frame)
# In a plot
sns.countplot(x = 'Survived', data = results)
plt.savefig('Plots/results_survived.png')
plt.close()
# Simple printing
for i in range(len(name)):
    print(name.iloc[i], ' -> ', end = '')
    if predictions[i] == 0:
        print('did not survive')
    else:
        print('did survive')
# end
