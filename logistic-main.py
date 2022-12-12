from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump 



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

    def train_model(self, X_train, y_train):
        try:
            model = LogisticRegression(solver="saga",max_iter=10000)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            raise e

    def evaluate(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            print("Accuracy Score : ", accuracy_score(y_test, y_pred))
            return True
        except Exception as e:
            raise e

    def save_model(self,model):
        try:
            dump(model,r"artifacts/model.pkl")
            return True
        except Exception as e:
            raise e            

    def run_pipeline(self):
        X_train, X_test, y_train, y_test = self.train_test_split()
        model = self.train_model(X_train,y_train)
        self.evaluate(model, X_test, y_test)
        self.save_model(model)
        return "Pipeline Run Completed "


if __name__ == "__main__":
    trainer = Model()
    trainer.run_pipeline()











def train_model(X,y):
    return True

