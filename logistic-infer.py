from joblib import load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self):
        self.iris = load_iris()

        self.X = self.iris.data
        self.y = self.iris.target

        print("Feature Names : ",self.iris.feature_names)
        print("Target Names : ",self.iris.target_names)

    def train_test_split(self, test_siz=0.33, random_state=42):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_siz, random_state=random_state)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise e

    def evaluate(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            print("Accuracy Score : ", accuracy_score(y_test, y_pred))
            return True
        except Exception as e:
            raise e

    def load_model(self):
        try:
            model = load(r"artifacts/model.pkl")
            return model
        except Exception as e:
            raise e            

    def run_pipeline(self):
        _, X_test, _, y_test = self.train_test_split()
        model = self.load_model()
        self.evaluate(model, X_test, y_test)

        return "Pipeline Run Completed "


if __name__ == "__main__":
    trainer = Model()
    trainer.run_pipeline()
