from sklearn.ensemble import RandomForestClassifier


def build_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=314,
        n_jobs=-1,
        max_features="sqrt",
    )
