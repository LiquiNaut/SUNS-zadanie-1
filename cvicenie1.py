from sklearn import datasets # https://scikit-learn.org/stable/
import pandas as pd # https://pandas.pydata.org/docs/reference/index.html#api
import plotly.graph_objects as go # https://plotly.com/
import matplotlib.pyplot as plt # https://matplotlib.org/

# Load the iris dataset
iris = datasets.load_iris()

# Create a 2D scatter plot - Matplotlib
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

# Show the plot
plt.show()

# Create a pandas DataFrame with the iris data
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

# Map the target values to the target names
iris_df['target_name'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

two_class_df = iris_df.copy()

# Merge versicolor and virginica into one class
two_class_df['target_name'] = two_class_df['target'].map({'setosa': 'setosa', 'versicolor': 'versicolor+virginica', 'virginica': 'versicolor+virginica'})

# plot_df = two_class_df
plot_df = iris_df

# Plotly
fig = go.Figure()

for target_name in plot_df['target_name'].unique():
    target_df = plot_df[plot_df['target_name'] == target_name]

    fig.add_trace(go.Scatter3d(
        x=target_df['sepal length (cm)'],
        y=target_df['sepal width (cm)'],
        z=target_df['petal length (cm)'],
        mode='markers',
        name=target_name,
    ))

fig.update_layout(
    scene=dict(
        xaxis=dict(title='sepal length (cm)'),
        yaxis=dict(title='sepal width (cm)'),
        zaxis=dict(title='petal length (cm)'),
    ),
)

fig.show()

# Logistic regression
from sklearn.linear_model import LogisticRegression

X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]
y = iris_df['target']

clf = LogisticRegression().fit(X, y)
print('Logistic regression: ', clf.score(X, y))

# MLP
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(1, 1), random_state=1, max_iter=1000).fit(X, y)
print('MLP: ', clf.score(X, y))

