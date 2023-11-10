import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Allow printing more columns
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Do not show pandas warnings
pd.set_option('mode.chained_assignment', None)

# Load the dataset https://ourworldindata.org/grapher/birth-rate-vs-death-rate
df = pd.read_csv('../data/birth-rate-vs-death-rate.csv')

# Rename columns for easier access
df.rename({
    'Entity': 'country',
    'Code': 'country_code',
    'Year': 'year',
    'Birth rate - Sex: all - Age: all - Variant: estimates': 'birth_rate',
    'Death rate - Sex: all - Age: all - Variant: estimates': 'death_rate',
    'Population - Sex: all - Age: all - Variant: estimates': 'population',
    'Continent': 'continent',
}, inplace=True, axis=1)

# Create some outliers
df['birth_rate'] = df['birth_rate'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['death_rate'] = df['death_rate'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['year'] = df['year'].apply(lambda x: x*100 if np.random.randint(0, 1000) < 1 else x)
df['population'] = df['population'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)

# Print min and max values of columns before removing outliers
print("*"*100, "Before removing outliers", "*"*100)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

# Deal with outliers - Remove outliers from birth_rate, death_rate, year, population
df = df[df['birth_rate'] > 0]
df = df[df['death_rate'] > 0]
df = df[(df['year'] >= 1950) & (df['year'] <= 2021)]
df = df[df['population'] > 0]

# Print min and max values of columns
print("*"*100, "After removing outliers", "*"*100)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

# Count missing values in columns
print("*"*100, "Missing values", "*"*100)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

# Deal with missing values
# Remove columns with lots of missing values
df_with_continent = df[df['continent'].notnull()]
df.drop(columns=['continent'], inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

print("*"*100, "Missing values after removing them", "*"*100)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

# Print column types
print("*"*100, "Column types", "*"*100)
print(df.dtypes)

# Label encoding for continent
le = LabelEncoder()
df_with_continent['labelEncoding'] = le.fit_transform(df_with_continent['continent'])

print("*"*100, "Label encoding", "*"*100)
print(df_with_continent[['continent', 'labelEncoding']].head(10))

continents_df = df_with_continent['continent']

# Dummy (one-hot) encoding for continent
df_with_continent = pd.get_dummies(df_with_continent, columns=['continent'], prefix='', prefix_sep='')
df_with_continent['continent'] = continents_df

continents = list(df_with_continent['continent'].unique())
show_columns = ['continent'] + continents

print("*"*100, "Dummy encoding", "*"*100)
print(df_with_continent[show_columns].head(10))

# Remove columns that are not needed
df.drop(columns=['country_code'], inplace=True)

# Keep only top 5 countries
top_5_countries = df.groupby('country').sum().sort_values(by='population', ascending=False).head(5).index
df = df[df['country'].isin(top_5_countries)]

# Use Label encoding for country
le = LabelEncoder()
df['country'] = le.fit_transform(df['country'])

# Split dataset into X and y
X = df.drop(columns=['country'])
y = df['country']

# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

# Print dataset shapes
print("*"*100, "Dataset shapes", "*"*100)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")

# Plot histograms before scaling
X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms before scaling/standardizing')
plt.show()

# Print min and max values of columns
print("*"*100, "Before scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# Scale data
# scaler = MinMaxScaler()
# !!!!!
# X_train = scaler.fit_transform(X_train)
# X_valid = scaler.transform(X_valid)
# X_test = scaler.transform(X_test)

# Standardize data
scaler = StandardScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Plot histograms after scaling/standardizing
X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms after scaling/standardizing')
plt.show()

# Print min and max values of columns
print("*"*100, "After scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# Train MLP model to predict country
print("*"*100, "MLP", "*"*100)
print(f"Random accuracy: {1/len(y_train.unique())}")

clf = MLPClassifier(
    hidden_layer_sizes=(100, 100, 5, 6, 90),
    random_state=1,
    max_iter=10,
    validation_fraction=0.2,
    early_stopping=True,
    learning_rate='adaptive',
    learning_rate_init=0.001,
).fit(X_train, y_train)

# Predict on train set
y_pred = clf.predict(X_train)
print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred))
cm_train = confusion_matrix(y_train, y_pred)

# Predict on test set
y_pred = clf.predict(X_test)
print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred))
cm_test = confusion_matrix(y_test, y_pred)

# Create class names for confusion matrix
class_names = list(le.inverse_transform(clf.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

