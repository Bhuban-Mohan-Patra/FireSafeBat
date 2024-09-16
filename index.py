import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
# from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from keras.layers import GRU, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, GRU, Dense, Conv1D, MaxPooling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('/content/Battery__.csv')
df.head(3)
from matplotlib import pyplot as plt
df['Cell-Nominal-Voltage-V'].plot(kind='hist', bins=20, title='Cell-Nominal-Voltage-V')
plt.gca().spines[['top', 'right',]].set_visible(False)
from matplotlib import pyplot as plt
df['Cell-Energy-Wh'].plot(kind='hist', bins=20, title='Cell-Energy-Wh')
plt.gca().spines[['top', 'right',]].set_visible(False)
from matplotlib import pyplot as plt
df['Cell-Capacity-Ah'].plot(kind='hist', bins=20, title='Cell-Capacity-Ah')
plt.gca().spines[['top', 'right',]].set_visible(False)
from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('Cell-Failure-Mechanism').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)
df1 = df.copy()
df1.drop(columns=['Cell-Description', 'Baseline-Total-Energy-Yield-kJ','Corrected-Total-Energy-Yield-kJ'], inplace=True)
df1.head(3)
df1['Cell-Failure-Mechanism']=df1['Cell-Failure-Mechanism'].replace(['Top Vent','Top and Bottom Vent','Top Vent Only - Bottom Vent Not Actuated','Top Vent and Bottom Rupture','Bottom Vent','Top Vent and Bottom Breach','Top Vent and Bottum Rupture'],['Ejection','Ejection','Ejection','Ejection','Ejection','Ejection','Ejection'])
df1['Cell-Failure-Mechanism'].value_counts()
from sklearn.preprocessing import LabelEncoder

# Define a dictionary to map categories to desired numerical values
mapping = {'Ejection': 1, 'No Ejection': 0}

# Initialize LabelEncoder
my_encoder = LabelEncoder()

# Use the map function to apply the custom mapping and then fit_transform
df1['Cell-Failure-Mechanism'] = df1['Cell-Failure-Mechanism'].map(mapping)
df1['Cell-Failure-Mechanism'] = my_encoder.fit_transform(df1['Cell-Failure-Mechanism'])

# Check the counts of encoded values
print(df1['Cell-Failure-Mechanism'].value_counts())
df2=df1.copy()
df2.drop(columns=['Pred-Temp'], inplace=True)
df2.head(3)
X = df2.drop(columns=['Cell-Failure-Mechanism']).values
y = df2['Cell-Failure-Mechanism'].values
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
from sklearn.preprocessing import StandardScaler

# Assuming X is your dataset
# Instantiate the scaler
scaler = StandardScaler()

# Fit the scaler to your dataset
X_train_transform = scaler.fit_transform(X_train)
X_test_transform = scaler.fit_transform(X_test)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', test_accuracy)
print('Test loss:', test_loss)
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score
# Calculate accuracy
y_pred_proba = model.predict(X_test_transform)
y_pred = (y_pred_proba > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate loss (already calculated)
loss, _ = model.evaluate(X_test_transform, y_test)
print('Loss:', loss)

# Calculate mean squared error (MSE) using predicted probabilities
mse = mean_squared_error(y_test, y_pred_proba)
print('Mean Squared Error (MSE):', mse)

# Calculate precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)
from sklearn.metrics import confusion_matrix

# Obtain predicted labels for the test set
y_pred = (model.predict(X_test_transform) > 0.5).astype(int)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Your confusion matrix
conf_matrix = np.array([[0, 6],
                        [0, 57]])

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()
# Calculate accuracy
y_pred_proba = model.predict(X_test_transform)
y_pred = (y_pred_proba > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate loss (already calculated)
loss, _ = model.evaluate(X_test_transform, y_test)
print('Loss:', loss)

# Calculate mean squared error (MSE) using predicted probabilities
mse = mean_squared_error(y_test, y_pred_proba)
print('Mean Squared Error (MSE):', mse)

# Calculate precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)


import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your actual values)
categories = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
model_scores = [0.90, 1.0, 0.95,  0.90476]

x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
rects = ax.bar(x, model_scores, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(categories)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects)

fig.tight_layout()

plt.show()
