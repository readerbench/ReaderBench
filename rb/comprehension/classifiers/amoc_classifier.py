from sklearn.model_selection import train_test_split
from rb.comprehension.classifiers.loader import load_file
from sklearn.ensemble import RandomForestClassifier


def random_forest_classification(filepath):
    X, y = load_file(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state=0, verbose=2)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(acc)
    print(clf.feature_importances_)
    return acc
