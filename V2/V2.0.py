import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor, ElasticNet
from sklearn.metrics import mean_absolute_error
from enum import Enum


class RegressionType(Enum):
    LINEAR = "LinearRegression"
    BAYESIAN_RIDGE = "BayesianRidge"
    SGD = "SGDRegressor"
    ELASTIC_NET = "ElasticNet"


class PredictionModel:
    def __init__(self, file_path, predictors, target, regression_type=RegressionType.LINEAR):
        self.file_path = file_path
        self.predictors = predictors
        self.target = target
        self.regression_type = regression_type

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        self.data.dropna(inplace=True)

    def train_model(self):
        train = self.data[self.data["year"] < 2012].copy()
        test = self.data[self.data["year"] >= 2012].copy()

        if self.regression_type == RegressionType.LINEAR:
            reg = LinearRegression()
        elif self.regression_type == RegressionType.BAYESIAN_RIDGE:
            reg = BayesianRidge()
        elif self.regression_type == RegressionType.SGD:
            reg = SGDRegressor()
        elif self.regression_type == RegressionType.ELASTIC_NET:
            reg = ElasticNet()

        reg.fit(train[self.predictors], train[self.target])

        predictions = reg.predict(test[self.predictors])
        test["predictions"] = predictions
        test.loc[test["predictions"] < 0, "predictions"] = 0
        test["predictions"] = test["predictions"].round()

        self.test_data = test

    def evaluate_model(self):
        error = mean_absolute_error(self.test_data[self.target], self.test_data["predictions"])
        print("Overall Mean Error:", error)

        errors = (self.test_data[self.target] - self.test_data["predictions"]).abs()
        print("Individual Errors:")
        print(errors)

        print("Predicted Values:")
        print(self.test_data[["year", "predictions"]])

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()


file_path = "../V1/teams.csv"
predictors = ["athletes", "prev_medals"]
target = "medals"
regression_type = RegressionType.LINEAR

model = PredictionModel(file_path, predictors, target, regression_type)
model.run()
