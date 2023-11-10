# dataset: https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit
# Tensorflow Keras: https://www.tensorflow.org/guide/keras/sequential_model
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
df = pd.read_csv('../data/citrus.csv')

# Split the dataset into X and y
X = df.drop(columns=['name'])
y = df['name']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Train MLP model in Keras
model = Sequential()
model.add(Dense(24, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00000000000002), metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32)

# Evaluate the model
test_scores = model.evaluate(X_test, y_test, verbose=0)

print("*"*100, "Test accuracy", "*"*100)
print(f"Test accuracy: {test_scores[1]:.4f}")

# Plot confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5)

# Create class names for confusion matrix
class_names = df['name'].unique().tolist()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

# Plot loss and accuracy
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
