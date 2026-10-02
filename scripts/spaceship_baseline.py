"""Spaceship Titanic — baseline model.

Pipeline:
  1. Load train/test CSVs.
  2. Feature engineering: split Cabin, extract group size from PassengerId,
     total onboard spend.
  3. Preprocess: median/most-frequent imputation + one-hot encoding.
  4. Random Forest with 5-fold cross-validation to estimate leaderboard score.
  5. Fit on the full training set and write submission.csv.

Run:  .venv/bin/python spaceship_baseline.py
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cabin is "deck/number/side" (side is P=port, S=starboard)
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    # PassengerId is "group_member": e.g. "0002_01" -> group 0002, member 01
    df["GroupSize"] = df.groupby(df["PassengerId"].str.split("_").str[0])[
        "PassengerId"
    ].transform("count")

    # Spending across all onboard amenities; cryo-sleepers spend ~nothing
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["SpentAnything"] = (df["TotalSpend"] > 0).astype(int)

    df["IsChild"] = (df["Age"] < 13).astype(int)
    return df


NUMERIC_FEATURES = [
    "Age",
    "CabinNum",
    "TotalSpend",
    "GroupSize",
    "IsChild",
    "SpentAnything",
]
CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


def main() -> None:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X = engineer_features(train)
    y = X.pop("Transported")
    X_test = engineer_features(test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("Per fold:", [f"{s:.4f}" for s in cv_scores])

    model.fit(X, y)
    predictions = model.predict(X_test)

    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Transported": predictions.astype(bool)}
    )
    submission.to_csv("submission.csv", index=False)
    print(f"\nWrote submission.csv ({len(submission)} rows)")


if __name__ == "__main__":
    main()
