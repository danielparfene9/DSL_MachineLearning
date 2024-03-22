import numpy as np
from antlr4 import *
from gen.dsl_grammarLexer import dsl_grammarLexer
from gen.dsl_grammarParser import dsl_grammarParser
from gen.dsl_grammarListener import dsl_grammarListener

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet

from main_ml import predictors


class DSLInterpreter(dsl_grammarListener):
    def __init__(self):
        self.datasets = {}
        self.regression_models = {
            'LinearRegression': LinearRegression,
            'BayesianRidge': BayesianRidge,
            'SGDRegressor': SGDRegressor,
            'ElasticNet': ElasticNet
        }

    def enterUse_statement(self, ctx: dsl_grammarParser.Use_statementContext):
        dataset_name = ctx.ID().getText()
        self.datasets[dataset_name] = pd.read_csv(dataset_name + ".csv")

    def enterModel_declaration(self, ctx: dsl_grammarParser.Model_declarationContext):
        model_name = ctx.ID().getText()
        regression_type = ctx.regression_type().getText()
        if regression_type in self.regression_models:
            self.datasets[model_name] = self.regression_models[regression_type]()
        else:
            print(f"Error: Unknown regression type '{regression_type}'")

    def enterTrain_command(self, ctx: dsl_grammarParser.Train_commandContext):
        model_name = ctx.ID(0).getText()
        dataset_name = ctx.ID(1).getText()
        predictors = [ctx.ID(i).getText() for i in range(2, len(ctx.ID()) - 1)]
        target = ctx.ID(-1).getText()
        model = self.datasets[model_name]
        dataset = self.datasets[dataset_name]
        model.fit(dataset[predictors], dataset[target])

    def enterPredict_command(self, ctx: dsl_grammarParser.Predict_commandContext):
        model_name = ctx.ID().getText()
        dataset_name = ctx.ID(1).getText()
        model = self.datasets[model_name]
        dataset = self.datasets[dataset_name]
        predictions = model.predict(dataset)
        print("Predictions:")
        print(predictions)

    def enterError_ratio_command(self, ctx: dsl_grammarParser.Error_ratio_commandContext):
        model_name = ctx.ID().getText()
        dataset_name = ctx.ID(1).getText()
        model = self.datasets[model_name]
        dataset = self.datasets[dataset_name]
        predictions = model.predict(dataset[predictors])
        errors = dataset["medals"] - predictions
        error_by_team = errors.groupby(dataset["team"]).mean()
        medals_by_team = dataset["medals"].groupby(dataset["team"]).mean()
        error_ratio = error_by_team / medals_by_team
        error_ratio = error_ratio[~pd.isnull(error_ratio)]
        error_ratio = error_ratio[np.isfinite(error_ratio)]
        error_ratio.sort_values()
        print("Error Ratio:")
        print(error_ratio)


def main():
    input_stream = FileStream('dsl_script.dsl')
    lexer = dsl_grammarLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = dsl_grammarParser(stream)
    tree = parser.dsl()

    interpreter = DSLInterpreter()
    walker = ParseTreeWalker()
    walker.walk(interpreter, tree)


if __name__ == '__main__':
    main()
