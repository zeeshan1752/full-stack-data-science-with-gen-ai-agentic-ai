
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score
)

sns.set(style='white')

# Load Data
dataset = pd.read_csv('iris.csv')

# Clean Column Names
dataset.columns = [
    colname.replace('(cm)', '').strip().replace(' ', '_')
    for colname in dataset.columns
]

features_names = dataset.columns.tolist()[:4]

# Feature Engineering
dataset['sepal_length_width_ratio'] = (
    dataset['sepal_length'] / dataset['sepal_width']
)

dataset['petal_length_width_ratio'] = (
    dataset['petal_length'] / dataset['petal_width']
)

# Select Features
dataset = dataset[
    [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'sepal_length_width_ratio',
        'petal_length_width_ratio',
        'target'
    ]
]

# Split Data
train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44
)

# X_train, y_train, X_test, y_test
X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data.loc[:, 'target'].values.astype('int32')

X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data.loc[:, 'target'].values.astype('int32')

# Logistic Regression
logreg = LogisticRegression(
    C=0.0001,
    solver='lbfgs',
    max_iter=100
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)

f1_lr = f1_score(y_test, predictions_lr, average='micro')
prec_lr = precision_score(y_test, predictions_lr, average='micro')
recall_lr = recall_score(y_test, predictions_lr, average='micro')

# Logistic Regression Accuracy
train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

# Random Forest Regressor
rf_reg = RandomForestRegressor()

rf_reg.fit(X_train, y_train)

predictions_rf = rf_reg.predict(X_test)

# Convert Predictions to Class Labels
predictions_rf_class = np.round(predictions_rf).astype(int)

# Random Forest Metrics
f1_rf = f1_score(y_test, predictions_rf_class, average='micro')
prec_rf = precision_score(y_test, predictions_rf_class, average='micro')
recall_rf = recall_score(y_test, predictions_rf_class, average='micro')

# Random Forest Accuracy (R2 Score)
train_acc_rf = rf_reg.score(X_train, y_train) * 100
test_acc_rf = rf_reg.score(X_test, y_test) * 100

# Confusion Matrix Plotting Function
def plot_cm(cm, target_name, title="Confusion Matrix", normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)

        cm_display = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )
    else:
        cm_display = cm

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm_display, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_name is not None:
        tick_marks = np.arange(len(target_name))
        plt.xticks(tick_marks, target_name, rotation=45)
        plt.yticks(tick_marks, target_name)

    thresh = cm_display.max() / 1.5 if normalize else cm_display.max() / 2

    for i, j in itertools.product(
        range(cm_display.shape[0]),
        range(cm_display.shape[1])
    ):

        if normalize:
            text_value = "{:0.4f}".format(cm_display[i, j])
        else:
            text_value = "{:,}".format(int(cm_display[i, j]))

        plt.text(
            j,
            i,
            text_value,
            horizontalalignment="center",
            color="white" if cm_display[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel('True Label')

    plt.xlabel(
        'Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass
        )
    )

    plt.savefig('ConfusionMatrix.png', dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()


# Target Names
target_name = np.array([
    'setosa',
    'versicolor',
    'virginica'
])

# Plot Confusion Matrix
plot_cm(
    cm_lr,
    target_name,
    title="Confusion Matrix (Logistic Regression)",
    normalize=True
)

# Feature Importance Plotting
importances = rf_reg.feature_importances_

labels = dataset.columns[:-1]

feature_df = pd.DataFrame(
    list(zip(labels, importances)),
    columns=['feature', 'importance']
)

features = feature_df.sort_values(
    by='importance',
    ascending=False
)

# Create Feature Importance Plot
plt.figure(figsize=(12, 6))

ax = sns.barplot(
    x='importance',
    y='feature',
    data=features
)

ax.set_xlabel('Importance', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)
ax.set_title('Random Forest Feature Importances', fontsize=12)

plt.tight_layout()

# Save and Show Feature Importance Plot
plt.savefig('FeatureImportance.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()

# Saving Scores to a File
with open('scores.txt', 'w', encoding='utf-8') as score:

    score.write("Random Forest Train Var: %2.1f%%\n" % train_acc_rf)
    score.write("Random Forest Test Var: %2.1f%%\n" % test_acc_rf)

    score.write("F1 Score: %2.1f%%\n" % (f1_rf * 100))
    score.write("Recall Score: %2.1f%%\n" % (recall_rf * 100))
    score.write("Precision Score: %2.1f%%\n" % (prec_rf * 100))

    score.write("\n\n")

    score.write("Logistic Regression Train Var: %2.1f%%\n" % train_acc_lr)
    score.write("Logistic Regression Test Var: %2.1f%%\n" % test_acc_lr)

    score.write("F1 Score: %2.1f%%\n" % (f1_lr * 100))
    score.write("Recall Score: %2.1f%%\n" % (recall_lr * 100))
    score.write("Precision Score: %2.1f%%\n" % (prec_lr * 100))


print("Model training completed successfully!")
print("Generated output files:")
print("1. ConfusionMatrix.png")
print("2. FeatureImportance.png")
print("3. scores.txt")