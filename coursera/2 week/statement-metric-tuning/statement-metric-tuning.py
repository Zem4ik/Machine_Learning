import numpy
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def main():
    (data, target) = load_boston(return_X_y=True)
    data = scale(data)
    result = calculate_scores(data, target)
    print(result)
    with open("1.out", "w") as file:
        file.write(str(result[0]))


def calculate_scores(data, target):
    scores = list()
    for p in numpy.linspace(1, 10, 200):
        clf = KNeighborsRegressor(weights='distance', p=p)
        score = cross_val_score(clf, data, target, cv=kf)
        scores.append((p, score.mean()))
    return max(scores, key=lambda pair: pair[1])


if __name__ == '__main__':
    main()
