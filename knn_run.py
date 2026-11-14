import pandas
import numpy
from knn import MyKNN
import matplotlib.pyplot as plt

wbdc = pandas.read_csv('wdbc.data.mb.csv', header=None)

# Split the features and labels of the entire data set.
features_df = wbdc.iloc[:,:30].to_numpy()
labels = wbdc.iloc[:,30].to_numpy()

# Choose random indices from the data frame using seed 42
# to separate the features and labels for both training and test data
numpy.random.seed(42)
train_indices = numpy.random.choice(len(features_df), int(len(features_df) * 0.7), replace=False)
test_indices = numpy.setdiff1d(numpy.arange(len(features_df)), train_indices)

train_features = features_df[train_indices]
train_labels = labels[train_indices]
test_features = features_df[test_indices]
test_labels = labels[test_indices]

# Initialize the predicted array with the dimensions required for the test
predicted = numpy.zeros((len(test_labels), 5))

# Loop through K values and store the predictions in separate columns
for i, k in enumerate(range(1, 10, 2)):
    my_knn = MyKNN(k)
    my_knn.normalize_fit(train_features, train_labels)
    
    predicted[:,i] = my_knn.predict(test_features, norm=True)


############# Data specific evaluation #######################

# Accuracy and confusion matrix construction
accuracy = numpy.mean(predicted == test_labels[:,numpy.newaxis], axis=0)

true_positive = numpy.sum((predicted == 1) & (test_labels[:,numpy.newaxis] == 1), axis=0)
true_negative = numpy.sum((predicted == -1) & (test_labels[:,numpy.newaxis] == -1), axis=0)
false_positive = numpy.sum((predicted == 1) & (test_labels[:,numpy.newaxis] == -1), axis=0)
false_negative = numpy.sum((predicted == -1) & (test_labels[:,numpy.newaxis] == 1), axis=0)

confusion = numpy.array([[true_positive, false_negative], [false_positive, true_negative]]).transpose(2, 0, 1)

# Terminal output display: Precision, Sensitivity, and Specificity 
precision = true_positive / (true_positive + false_positive)
tp_rate = true_positive / (true_positive + false_negative)
tn_rate = true_negative / (true_negative + false_positive)

for j, l in enumerate(range(1, 10, 2)):
    print(f"K = {l}")
    print(f"Precision: {precision[j]}")
    print(f"True Positive Rate: {tp_rate[j]}")
    print(f"True Negative Rate: {tn_rate[j]}")

# Subplots of confusion matrices in one figure
fig, axes = plt.subplots(1, 5)
for i in range(0, 5):
    axes[i].table(
        cellText=confusion[i],
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
        rowLabels=["P", "N"],
        colLabels=["P", "N"]
    )
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_box_aspect(1)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual", labelpad=15)
    axes[i].set_title(f"K = {2*i + 1} \n Accuracy: {accuracy[i]:.2f}")

fig.set_size_inches(20, 10)
plt.tight_layout()
plt.show(block=True)

    







