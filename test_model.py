from sklearn.datasets import load_breast_cancer
from sklearn.datasets import california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
from gradient_boosting import GradientBooster
from loss_functions import AbsoluteLoss, BinomialDevianceLoss, HingeLoss


def test_abs_loss():
    X, y = california_housing.fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    regressor = GradientBooster(n_iter=100, loss_function=AbsoluteLoss())
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)    
    print(f"Test Mean absolute error: {mean_absolute_error(y_pred, y_test):.4f}")


def test_binomial_loss():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = GradientBooster(n_iter=100, loss_function=BinomialDevianceLoss())
    clf.fit(X_train, y_train)

    y_pred = (clf.predict(X_test) > 0.5).astype(int)
    print(f"Test Accuracy: {accuracy_score(y_pred, y_test):.4f}")


def test_hinge_loss():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = GradientBooster(n_iter=100, loss_function=HingeLoss())
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_pred, y_test):.4f}")


def main():
    test_abs_loss()
    test_binomial_loss()
    test_hinge_loss()


if __name__ == "__main__":
    main()
