import numpy as np
from sklearn.metrics import accuracy_score, log_loss, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd

# Dry Bean Dataset taken from https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
data = pd.read_excel('Dry_Bean_Dataset.xlsx')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data set to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the dataset by computing mean and standard deviation
training_mean = np.mean(X_train, axis=0)
training_std = np.std(X_train, axis=0)

X_train_std = (X_train - training_mean) / training_std
X_test_std = (X_test - training_mean) / training_std

# Defining ten combinations of parameters for MLP Classification
# Tried relu, tanh activation functions, solver: adam and lbfgs with varying max iteration values
diff_params_combo = [
    {'hidden_layer_sizes' : (100,), 'activation' :'relu', 'solver' : 'adam', 'batch_size' : 'auto', 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 50, 'verbose' : 1},
    {'hidden_layer_sizes' : (100,), 'activation' :'tanh', 'solver' : 'adam', 'alpha' : 0.0001, 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 10},
    {'hidden_layer_sizes' : (100, 50), 'activation' :'relu', 'solver' : 'adam', 'batch_size' : 'auto', 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 50},
    {'hidden_layer_sizes' : (100,), 'activation' :'relu', 'solver' : 'adam', 'alpha' : 0.0001, 'batch_size' : 'auto', 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 50},
    {'hidden_layer_sizes' : (100,), 'activation' :'relu', 'solver' : 'adam', 'learning_rate' : 'adaptive', 'learning_rate_init' : 0.0001, 'max_iter' : 20, 'random_state' : 42, 'early_stopping' : False},
    {'hidden_layer_sizes' : (100, 50), 'activation' :'tanh', 'solver' : 'adam', 'alpha' : 0.0001, 'batch_size' : 'auto', 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 10},
    {'hidden_layer_sizes' : (100, 20), 'activation' :'relu', 'solver' : 'adam', 'batch_size' : 'auto', 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 80},
    {'hidden_layer_sizes' : (200,), 'activation' :'relu', 'solver' : 'adam', 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 60, 'random_state' : 42},
    {'hidden_layer_sizes' : (100,), 'activation' :'relu', 'solver' : 'lbfgs', 'alpha' : 0.0001, 'batch_size' : 'auto', 'learning_rate' : 'adaptive', 'learning_rate_init' : 0.0001, 'max_iter' : 10},
    {'hidden_layer_sizes' : (200, 100), 'activation' :'tanh', 'solver' : 'adam', 'batch_size' : 'auto', 'learning_rate' : 'constant', 'learning_rate_init' : 0.0001, 'max_iter' : 20, 'random_state' : 42}

]

# Initializing the lists to store metrics
training_metrics = []
test_metrics = []
epochs_list = []

for params in diff_params_combo:
    mlp_classifier = MLPClassifier(**params)

    mlp_classifier.fit(X_train_std, y_train)
    epochs_list.append(mlp_classifier.n_iter_)

    y_train_pred = mlp_classifier.predict(X_train_std)
    y_test_pred = mlp_classifier.predict(X_test_std)

    # Computing training data metrics
    accuracy_train = accuracy_score(y_train, y_train_pred)
    sensitivity_train = recall_score(y_train, y_train_pred, average='weighted')
    specificity_train = precision_score(y_train, y_train_pred, average='weighted')
    f1_score_train = f1_score(y_train, y_train_pred, average='weighted')
    log_loss_train = log_loss(y_train, mlp_classifier.predict_proba(X_train_std))

    # Computing test data metrics
    accuracy_test = accuracy_score(y_test, y_test_pred)
    sensitivity_test = recall_score(y_test, y_test_pred, average='weighted')
    specificity_test = precision_score(y_test, y_test_pred, average='weighted')
    f1_score_test = f1_score(y_test, y_test_pred, average='weighted')
    log_loss_test = log_loss(y_test, mlp_classifier.predict_proba(X_test_std))

    # Append the metrics to the list
    training_metrics.append({
        'Accuracy': accuracy_train,
        'Sensitivity': sensitivity_train,
        'Specificity': specificity_train,
        'F1 Score': f1_score_train,
        'Log Loss': log_loss_train
    })

    # Append the metrics to the list
    test_metrics.append({
        'Accuracy': accuracy_test,
        'Sensitivity': sensitivity_test,
        'Specificity': specificity_test,
        'F1 Score': f1_score_test,
        'Log Loss': log_loss_test
    })

# for i, epochs in enumerate(epochs_list):
#     print(f'Combination {i+1}: Number of epochs = {epochs}')
#
# for i, metrics in enumerate(zip(training_metrics, test_metrics)):
#     training_metrics, test_metrics = metrics
#     print(f'Combination {i + 1}:')
#     print('------------Training Metrics------------')
#     print('Training Metrics: ', training_metrics)
#     print('------------Test Metrics------------')
#     print('Test Metrics:', test_metrics)
#     print()


combinations = list(range(1, len(diff_params_combo) + 1))

# Convert training and test metrics to DataFrame
train_metrics_df = pd.DataFrame(training_metrics, index=combinations)
test_metrics_df = pd.DataFrame(test_metrics, index=combinations)

# Plot Accuracy for Training and Testing
plt.figure(figsize=(10, 5))
plt.plot(combinations, train_metrics_df['Accuracy'], marker='o', label='Training Accuracy', linestyle='--')
plt.plot(combinations, test_metrics_df['Accuracy'], marker='s', label='Testing Accuracy', linestyle='-')
plt.xlabel('Model Configuration')
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy Across Configurations')
plt.legend()
plt.grid(True)
plt.show()

# Plot F1 Score for Training and Testing
plt.figure(figsize=(10, 5))
plt.plot(combinations, train_metrics_df['F1 Score'], marker='o', label='Training F1 Score', linestyle='--')
plt.plot(combinations, test_metrics_df['F1 Score'], marker='s', label='Testing F1 Score', linestyle='-')
plt.xlabel('Model Configuration')
plt.ylabel('F1 Score')
plt.title('Training vs Testing F1 Score Across Configurations')
plt.legend()
plt.grid(True)
plt.show()

# Plot Log Loss for Training and Testing
plt.figure(figsize=(10, 5))
plt.plot(combinations, train_metrics_df['Log Loss'], marker='o', label='Training Log Loss', linestyle='--')
plt.plot(combinations, test_metrics_df['Log Loss'], marker='s', label='Testing Log Loss', linestyle='-')
plt.xlabel('Model Configuration')
plt.ylabel('Log Loss')
plt.title('Training vs Testing Log Loss Across Configurations')
plt.legend()
plt.grid(True)
plt.show()

# Plot Number of Epochs Required for Convergence
plt.figure(figsize=(10, 5))
plt.bar(combinations, epochs_list, color='blue', alpha=0.7)
plt.xlabel('Model Configuration')
plt.ylabel('Epochs to Converge')
plt.title('Epochs to Converge for Each Configuration')
plt.grid(True)
plt.show()

y_test_bin = label_binarize(y_test, classes=np.unique(y))

plt.figure(figsize=(10, 7))

for i, params in enumerate(diff_params_combo):
    # Train the model with given parameters
    mlp_classifier = MLPClassifier(**params)
    mlp_classifier.fit(X_train_std, y_train)

    # Get probability scores for test set
    y_test_proba = mlp_classifier.predict_proba(X_test_std)

    # Compute ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_test_proba.ravel())
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.plot(fpr, tpr, label=f'Model {i + 1} (AUC = {roc_auc:.2f})')

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], color='black', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for MLP Classifier Models')
plt.legend()
plt.grid(True)
plt.show()