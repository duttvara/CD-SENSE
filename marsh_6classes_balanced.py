import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.ensemble import(
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Upload csv
from google.colab import files
uploaded = files.upload()

data = pd.read_csv('celiac_disease_lab_data.csv')

from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Initial checks
print("Data shape:", data.shape)
print("Data types:\n", data.dtypes)

df = data.copy()

# Clean 'Marsh' column
df['Marsh'] = df['Marsh'].astype(str).str.lower().str.strip()
df['Marsh'] = df['Marsh'].replace({
    'marsh type 1': '1',
    'marsh type 2': '2',
    'marsh type 3a': '3a',
    'marsh type 3b': '3b',
    'marsh type 3c': '3c',
    'none': np.nan,
    'nan': np.nan,
    'marsh type 0': '0'
})
marsh_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3a': 3,
    '3b': 4,
    '3c': 5,
}
df['Marsh'] = df['Marsh'].map(marsh_mapping)

# Drop rows with NaN in 'Marsh'
df = df.dropna(subset=['Marsh'])
df['Marsh'] = df['Marsh'].astype(int)

# Clean feature columns and handle missing values
feature_cols = [
    'Diabetes Type',
    'Short_Stature',
    'Sticky_Stool',
    'Weight_loss',
    'IgA',
    'IgG'  # Include 'Marsh' as a feature for SMOTE
]

for col in feature_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()
    df[col].replace({'nan': np.nan, 'none': np.nan}, inplace=True)

# Drop rows with NaNs in important features
df = df.dropna(subset=feature_cols)

# Label encode categorical features
categorical_features = [
    'Diabetes Type',
    'Short_Stature',
    'Sticky_Stool',
    'Weight_loss',
]

le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# Convert numeric features to float
numeric_features = ['IgA', 'IgG']
for col in numeric_features:
    df[col] = df[col].astype(float)

# Map 'yes' and 'no' in 'Disease_Diagnose' to 1 and 0
df['Disease_Diagnose'] = df['Disease_Diagnose'].map({'yes': 1, 'no': 0})

# Define X (features) and y (target: 'Marsh')
X = df[feature_cols]
y = df['Marsh']  # Use 'Marsh' as the target

# Check the distribution of 'Marsh' before balancing
print("Original 'Marsh' distribution:")
print(y.value_counts())

# Identify categorical feature indices
categorical_indices = [feature_cols.index(col) for col in categorical_features] # Index of 'Marsh' in feature_cols

# Apply SMOTENC to balance the dataset for 'Marsh'
smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)


print("Resampled 'Marsh' distribution:")
print(np.bincount(y_resampled))

# Convert the resampled data back into a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['Marsh'] = y_resampled

# Check the shape of the resampled dataset
print("Resampled data shape:", resampled_data.shape)
print(resampled_data.tail(20))
diabetes_type_1_count = resampled_data[resampled_data['Diabetes Type'] == 1].shape[0]
print(f"Number of data points with 'Diabetes Type' == 1: {diabetes_type_1_count}")

print("Resampled 'Marsh' distribution:")
print(resampled_data['Marsh'].value_counts())

#gridsearchcv for hyperparamater tuning
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

feature_column = [
    'Diabetes Type',
    'Short_Stature',
    'Sticky_Stool',
    'Weight_loss',
    'IgA',
    'IgG',
]

X = resampled_data[feature_column]
y = resampled_data['Marsh']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.astype(int)
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter grids for each model
param_grids = {
    'SVC': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt']
    },
    'ExtraTreesClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt']
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt']
    },
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 300]
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1, 10]
    },
    'XGBClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 1.0]
    },
    'BaggingClassifier': {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.8, 1.0],
        'max_features': [0.5, 0.8, 1.0]
    },
    'GaussianNB': {}  # No hyperparameters to tune for Naive Bayes
}

#. results
results = {}

#  GridSearchCV
def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

# List of models to tune
models = {
    'SVC': SVC(),
    'RandomForestClassifier': RandomForestClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'XGBClassifier': XGBClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'GaussianNB': GaussianNB()
}

# Perform hyperparameter tuning for each model
for model_name, model in models.items():
    param_grid = param_grids.get(model_name, {})
    best_params, best_score = tune_model(model, param_grid, X_train_scaled, y_train)
    results[model_name] = {
        'Best Parameters': best_params,
        'Best Score': best_score
    }

# Print the results
for model_name, result in results.items():
    print(f"{model_name}: {result}")

X = resampled_data[feature_column]
y = resampled_data['Marsh']
y = y.astype(int)

models = {
    'Support Vector Classifier': SVC(C=100, gamma='scale', kernel='rbf'),
    'Random Forest': RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_split=5, n_estimators=50),
    'Extra Trees': ExtraTreesClassifier(max_depth=None, max_features='sqrt', min_samples_split=10, n_estimators=50),
    'Decision Trees': DecisionTreeClassifier(max_depth=None, max_features='sqrt', min_samples_split=2),
    'Logistic Regression': LogisticRegression(C=0.01, max_iter=100, penalty='l2', solver='saga'),
    'AdaBoost': AdaBoostClassifier(learning_rate=0.1, n_estimators=100),
    'XGBoost': XGBClassifier(learning_rate=0.2, max_depth=6, n_estimators=100, subsample=1.0),
    'Bagging Classifier': BaggingClassifier(max_features=0.8, max_samples=0.8, n_estimators=100),
    'Naive Bayes': GaussianNB(),
}



from tqdm import tqdm

feature_column = [
    'Diabetes Type',
    'Short_Stature',
    'Sticky_Stool',
    'Weight_loss',
    'IgA',
    'IgG',
]
X = resampled_data[feature_column]
y = resampled_data['Marsh']
y = y.astype(int)

results = []
n_iterations = 1000

# Loop through each model
for model_name, model in models.items():
    print(f"Processing model: {model_name}")
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []

    # Bootstrap iterations with progress bar
    for i in tqdm(range(n_iterations), desc=f'Bootstrapping {model_name}', leave=False):
        X_resampled, y_resampled = resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='weighted')
            elif hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test)
                auc = roc_auc_score(y_test, y_scores, multi_class='ovo', average='weighted')
            else:
                auc = np.nan
        except ValueError:
            auc = np.nan

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)

    # Compute mean and 95% CIs
    metrics_dict = {
        'Model': model_name,
        'Accuracy Mean': np.mean(accuracy_list),
        'Accuracy 95% CI': (
            np.percentile(accuracy_list, 2.5),
            np.percentile(accuracy_list, 97.5),
        ),
        'Precision Mean': np.mean(precision_list),
        'Precision 95% CI': (
            np.percentile(precision_list, 2.5),
            np.percentile(precision_list, 97.5),
        ),
        'Recall Mean': np.mean(recall_list),
        'Recall 95% CI': (
            np.percentile(recall_list, 2.5),
            np.percentile(recall_list, 97.5),
        ),
        'F1 Score Mean': np.mean(f1_list),
        'F1 Score 95% CI': (
            np.percentile(f1_list, 2.5),
            np.percentile(f1_list, 97.5),
        ),
        'AUC Mean': np.nanmean(auc_list),
        'AUC 95% CI': (
            np.nanpercentile(auc_list, 2.5),
            np.nanpercentile(auc_list, 97.5),
        ),
    }

    # Append
    results.append(metrics_dict)

# Create DataFrame
results_df = pd.DataFrame(results)
results_df = results_df[
    [
        'Model',
        'Accuracy Mean',
        'Accuracy 95% CI',
        'Precision Mean',
        'Precision 95% CI',
        'Recall Mean',
        'Recall 95% CI',
        'F1 Score Mean',
        'F1 Score 95% CI',
        'AUC Mean',
        'AUC 95% CI',
    ]
]
print("\nBootstrapped Model Performance Metrics:")
print(results_df)

X = resampled_data[feature_column]
y = resampled_data['Marsh']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

# Loop through each model and plot the confusion matrix
for model_name, model in models.items():
    print(f"Evaluating model: {model_name}")

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

from sklearn.inspection import permutation_importance

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate and visualize permutation importance for all models
for model_name, model in models.items():
    print(f"Processing model: {model_name}")

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Compute permutation importance
    perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42)

    # Check if the number of features matches the importances
    if len(feature_cols) != len(perm_importance.importances_mean):
        raise ValueError(f"Mismatch in feature names ({len(feature_cols)}) and importances ({len(perm_importance.importances_mean)}) "
                         f"for model: {model_name}.")

    # Create a DataFrame for importances
    perm_importances_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(perm_importances_df['Feature'], perm_importances_df['Importance'], color='skyblue', xerr=perm_importances_df['Std'])
    plt.xlabel("Mean Decrease in Accuracy")
    plt.ylabel("Feature")
    plt.title(f"Permutation Feature Importance ({model_name})")
    plt.gca().invert_yaxis()  # Invert y-axis
    plt.show()

from sklearn.metrics import roc_curve, auc

# Define models with probability enabled for SVC
models = {
    'Support Vector Classifier': SVC(C=100, gamma='scale', kernel='rbf',probability=True),
    'Random Forest': RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_split=5, n_estimators=50),
    'Extra Trees': ExtraTreesClassifier(max_depth=None, max_features='sqrt', min_samples_split=10, n_estimators=200),
    'Decision Trees': DecisionTreeClassifier(max_depth=None, max_features='sqrt', min_samples_split=5),
    'Logistic Regression': LogisticRegression(C=0.01, max_iter=100, penalty='l2', solver='saga'),
    'AdaBoost': AdaBoostClassifier(learning_rate=0.1, n_estimators=100),
    'XGBoost': XGBClassifier(learning_rate=0.2, max_depth=6, n_estimators=100, subsample=1.0),
    'Bagging Classifier': BaggingClassifier(max_features=0.8, max_samples=0.5, n_estimators=50),
    'Naive Bayes': GaussianNB(),
}

# Use df['Marsh'] and feature_cols for X, y
X = resampled_data[feature_column]
y = resampled_data['Marsh']
y = y.astype(int)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classes and binarize y_test for ROC calculations
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

# Train models and create a separate ROC plot for each model
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Fit the model

    # Check if model supports probability prediction
    if not hasattr(model, "predict_proba"):
        print(f"Skipping {model_name} as it does not support probability estimates.")
        continue

    # Initialize a new figure for each model's ROC plot
    plt.figure(figsize=(8, 6))

    # Compute ROC curve and AUC for each class in a One-vs-Rest manner
    for i, class_name in enumerate(classes):
        # Generate class-specific probabilities and binary test labels
        y_test_class = y_test_bin[:, i]
        y_score_class = model.predict_proba(X_test_scaled)[:, i]

        # Calculate ROC curve and AUC for each class
        fpr, tpr, _ = roc_curve(y_test_class, y_score_class)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve for each class
        plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")

    # Diagonal line for reference
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.5)

    # Plot settings
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves for {model_name}")
    plt.legend(loc="lower right", frameon=False, fontsize='small')
    plt.grid(visible=True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()

import shap  # SHAP package

# Define features (X) and target (y)
X = resampled_data[feature_column]  
y = resampled_data['Marsh']       

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, stratify=y)

my_model = XGBClassifier(learning_rate=0.2, max_depth=6, n_estimators=100, subsample=1.0)
my_model.fit(train_X, train_y)

explainer = shap.TreeExplainer(my_model)

# Calculate SHAP values
shap_values = explainer(val_X) # Automatically returns correct format

# Check if SHAP values have multiple outputs for multi-class
try:
    for i in range(len(shap_values.values[0])):  # Loop through classes
        print(f"Class {i}:")
        shap.summary_plot(shap_values[..., i].values, val_X, show=True)
except Exception as e:
    raise ValueError(f"Failed to process SHAP values for multi-class. Error: {e}")
