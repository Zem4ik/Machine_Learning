import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def main():
    pandas.set_option('display.max_columns', None)
    data = pandas.read_csv("titanic.csv", index_col='PassengerId')
    print(data.keys())
    all_fields = ["Survived", "Pclass", "Fare", "Age", "Sex"]
    x_fields = ["Pclass", "Fare", "Age", "Sex"]
    y_field = "Survived"
    X = data[all_fields]
    X["Sex"] = X["Sex"].map({"male": 1, "female": 2})
    X = X.dropna()
    y = X[y_field]
    X = X[x_fields]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    importances = clf.feature_importances_
    pairs = list(zip(x_fields, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    answer = list(map(lambda pair: pair[0], pairs[:2]))
    answer = ' '.join(answer)
    print(answer)
    with open("2nd task.out", "w") as file:
        file.write(answer)


if __name__ == "__main__":
    main()
