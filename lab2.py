import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

# pre-processing

file_link = 'https://drive.google.com/file/d/1uwdV2Sfw6HTAAMUn_ODuqc1uLw_bzJ7L/view?usp=sharing'
id = file_link.split('/')[-2]
new = f'https://drive.google.com/uc?id={id}'
df = pd.read_csv(new)
df.head()

df = df.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
df = df.drop_duplicates()
df.dropna(subset=['Embarked'],inplace=True)
df
df['Age'].fillna(df['Age'].median(),inplace=True)
df
# Apply one-hot encoding before splitting the data
if 'Sex' in df.columns and 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
else:
    print("Columns 'Sex' or 'Embarked' not found in the DataFrame.")

X = df.drop(['Survived'],axis=1)
Y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)




class DecisionTree:
    def __init__(self, max_depth = None):
        ### Your Code Here ###
        self.max_depth = max_depth
        pass

    def fit(self, X, y, depth = 0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return {'class': np.bincount(y).argmax()}
        best_feature, best_threshold = self.find_best_split(X, y)

        left = X[:, best_feature] <= best_threshold
        right = ~left
        left_subset, left_labels = X[left], y[left]
        right_subset, right_labels = X[right], y[right]

        # Recursively build the tree
        left_node = self.fit(left_subset, left_labels, depth + 1)
        right_node = self.fit(right_subset, right_labels, depth + 1)

        # return a decision node with feature, threshold, left_node, and right_node
        return {'feature': best_feature, 'threshold': best_threshold,'left': left_node, 'right': right_node}

        pass

    def find_best_split(self, X, y):
        gini_best = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask
                left_gini = self.calculate_gini(y[left_mask])
                right_gini = self.calculate_gini(y[right_mask])
                total_gini = (len(y[left_mask]) / len(y)) * left_gini + (len(y[right_mask]) / len(y)) * right_gini

                if total_gini < gini_best:
                    gini_best = total_gini
                    best_feature = feature
                    best_threshold = thresh

        return best_feature, best_threshold

    #create another method to calculate gini impurity if needed
    def calculate_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        gini = 1
        total_samples = len(y)

        for count in counts:
            gini -= (count / total_samples) ** 2

        return gini

    def predict(self, node, x):
        # if node is a leaf node:
        #     return the class of the leaf node
        # else:
        #     if instance's feature value <= node's threshold:
        #         return predict(node's left_node, instance)
        #     else:
        #         return predict(node's right_node, instance)
        if 'class' in node:
            return node['class']
        else:
            if x[node['feature']] <= node['threshold']:
                return self.predict(node['left'], x)
            else:
                return self.predict(node['right'], x)
        pass


# converting Pandas dataframe to NumPy Array
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()


# Create the model and fit the training data
decision_tree = DecisionTree(max_depth=5)
final_tree = decision_tree.fit(X_train_np, y_train_np) # returns the information of the final tree in a suitable data structure

# Predictions on the test set
y_pred_dt_scr = np.array([decision_tree.predict(final_tree, instance) for instance in X_test_np])

# Calculate and print the accuracy using y_test and y_pred manually
# Hint: accuracy = number of correctly predicted datapoints / total datapoints
accuracy_dt_scr = 0

### Your Code Here ###

print("Decision Tree Classifier (from scratch) Accuracy:", accuracy_dt_scr)