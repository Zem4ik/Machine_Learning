import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def main():
    df = pandas.read_csv("wine.data", header=None)

    y = df[0].values
    X = df.drop([0], axis=1).values

    result = calculateScores(y, X)
    with open("1.out", "w") as file:
        file.write(str(result[0]))
    with open("2.out", "w") as file:
        file.write(str(result[1]))

    X = scale(X)
    result = calculateScores(y, X)
    with open("3.out", "w") as file:
        file.write(str(result[0]))
    with open("4.out", "w") as file:
        file.write(str(result[1]))


def calculateScores(y, X):
    scores = list()
    for k in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=kf)
        scores.append((k, score.mean()))
    return max(scores, key=lambda pair: pair[1])


if __name__ == "__main__":
    main()
