from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X_train,y_train,):
    """
    Train Logistic Regression model.
    """
    model = LogisticRegression(random_state=42,max_iter=1000,)
    model.fit(X_train,y_train,)
    return model


def predict(model, X,):
    """
    Predict labels.
    """
    return model.predict(X)


def predict_proba(model, X,):
    """
    Predict probabilities.
    """
    return model.predict_proba(X)


def save_model(model,save_path: str,):
    """
    Save trained model.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True,)
    joblib.dump( model, save_path,)


def load_model(path: str,):
    """
    Load saved model.
    """
    return joblib.load(path)