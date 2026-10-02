from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def train_model(X, y):
    """
    INPUT:
        X: A pandas table of phone specifications
           Must not include 'id' or 'price_range'
        y: The matching 'price_range' column containing labels 0–3
           Each label must match the phone in the same row of X

    OUTPUT:
        model: A trained model that can predict price categories
        metrics: A dictionary containing accuracy, weighted F1,
               describing model performance
    """

    # Use 80% of the data for training and 20% for validation
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create a random forest containing 100 decision trees
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Learn from the training specifications and  known categories
    model.fit(X_train, y_train)

    # Predict categories for the validation phones
    predictions = model.predict(X_validation)

    # Compare predictions against validation answers
    metrics = {
        # Fraction of predictions that are correct, between 0 and 1
        "accuracy": accuracy_score(y_validation, predictions),

        # Combines precision and recall, weighted by category size
        "weighted_f1": f1_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0
        ),
    }

    # Store the model and displays the evaluation results
    return model, metrics


def predict_prices(trained_model, X):
    """
    INPUT:
        trained_model: The model returned by train_model()
        X: New phone specifications
           Must have the same columns and order as the training data
           Must not include 'id' or 'price_range'

    OUTPUT:
        One category number per phone:
        0 = low cost
        1 = medium cost
        2 = high cost
        3 = very high cost
    """

    # A trained model is required before making predictions.
    if trained_model is None:
        raise ValueError("Train the model before predicting.")

    # Return category numbers
    return trained_model.predict(X)