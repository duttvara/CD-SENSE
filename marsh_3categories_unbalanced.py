import pandas as pd
import numpy as np
!pip install scikit-learn==1.2.2  # Or any version below 1.6
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

data = pd.read_csv('/content/celiac_disease_lab_data.csv')

print("Data shape:", data.shape)
print("Data types:\n", data.dtypes)
df = data.copy()

print(df.duplicated().sum())
df.drop_duplicates(inplace=True)

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

# Define features
feature_cols = [
    'Diabetes Type',
    'Short_Stature',
    'Sticky_Stool',
    'Weight_loss',
    'IgA',
    'IgG',
]

for col in feature_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()
    df[col].replace({'nan': np.nan, 'none': np.nan}, inplace=True)

df = df.dropna(subset=feature_cols)
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

# Define X and y
X = df[feature_cols]
y = df['Marsh']
print("Unique values in 'Marsh':", y.unique())
print(data.shape)
print(df.shape)

def categorize_marsh(marsh):
    if marsh == 0:
        return 'Normal'  # No damage to villi
    elif marsh == 1 or marsh == 2:
        return 'Mild'  # Early/mild stage damage to villi
    else:
        return 'Severe'  # Advanced damage to villi


df['Marsh_Category'] = df['Marsh'].apply(categorize_marsh)

X = df[feature_cols]
y = df['Marsh_Category']

df.head()
df['Marsh_Category'].head()
# Define the mapping for Marsh_Category
category_mapping = {'Normal': 0, 'Mild': 1, 'Severe': 2}


df['Marsh_Category'] = df['Marsh_Category'].map(category_mapping)


print(df['Marsh_Category'].head())

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


X = df[feature_cols]
y = df['Marsh_Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

models = {
    'Support Vector Classifier': SVC(C=100, gamma='scale', kernel='rbf',probability=True),
    'Random Forest': RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_split=2, n_estimators=200),
    'Extra Trees': ExtraTreesClassifier(max_depth=30, max_features='auto', min_samples_split=5, n_estimators=100),
    'Decision Trees': DecisionTreeClassifier(max_depth=20, max_features='auto', min_samples_split=2),
    'Logistic Regression': LogisticRegression(C=0.1, max_iter=100, penalty='l1', solver='liblinear'),
    'AdaBoost': AdaBoostClassifier(learning_rate=1, n_estimators=100),
    'XGBoost': XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8),
    'Bagging Classifier': BaggingClassifier(max_features=1.0, max_samples=1.0, n_estimators=100),
    'Naive Bayes': GaussianNB(),
}

X = df[feature_cols]
y = df['Marsh_Category']

from tqdm import tqdm

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
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
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

    #Compute mean and 95% CIs
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


    results.append(metrics_dict)

# add to df
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

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


X = df[feature_cols]
y = df['Marsh_Category']


models = {
    'Support Vector Classifier': SVC(C=100, gamma='scale', kernel='rbf',probability=True),
    'Random Forest': RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_split=2, n_estimators=200),
    'Extra Trees': ExtraTreesClassifier(max_depth=30, max_features='auto', min_samples_split=5, n_estimators=100),
    'Decision Trees': DecisionTreeClassifier(max_depth=20, max_features='auto', min_samples_split=2),
    'Logistic Regression': LogisticRegression(C=0.1, max_iter=100, penalty='l1', solver='liblinear'),
    'AdaBoost': AdaBoostClassifier(learning_rate=1, n_estimators=100),
    'XGBoost': XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8),
    'Bagging Classifier': BaggingClassifier(max_features=1.0, max_samples=1.0, n_estimators=100),
    'Naive Bayes': GaussianNB(),
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


for model_name, model in models.items():
    print(f"Evaluating model: {model_name}")

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, model_name)
  
