# Generated from C:/Users/danie/PycharmProjects/DSL_MachineLearning/dsl_grammar.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .dsl_grammarParser import dsl_grammarParser
else:
    from dsl_grammarParser import dsl_grammarParser

# This class defines a complete listener for a parse tree produced by dsl_grammarParser.
class dsl_grammarListener(ParseTreeListener):

    # Enter a parse tree produced by dsl_grammarParser#dsl.
    def enterDsl(self, ctx:dsl_grammarParser.DslContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#dsl.
    def exitDsl(self, ctx:dsl_grammarParser.DslContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#statement.
    def enterStatement(self, ctx:dsl_grammarParser.StatementContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#statement.
    def exitStatement(self, ctx:dsl_grammarParser.StatementContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#use_statement.
    def enterUse_statement(self, ctx:dsl_grammarParser.Use_statementContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#use_statement.
    def exitUse_statement(self, ctx:dsl_grammarParser.Use_statementContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#model_declaration.
    def enterModel_declaration(self, ctx:dsl_grammarParser.Model_declarationContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#model_declaration.
    def exitModel_declaration(self, ctx:dsl_grammarParser.Model_declarationContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#regression_type.
    def enterRegression_type(self, ctx:dsl_grammarParser.Regression_typeContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#regression_type.
    def exitRegression_type(self, ctx:dsl_grammarParser.Regression_typeContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#train_command.
    def enterTrain_command(self, ctx:dsl_grammarParser.Train_commandContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#train_command.
    def exitTrain_command(self, ctx:dsl_grammarParser.Train_commandContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#predict_command.
    def enterPredict_command(self, ctx:dsl_grammarParser.Predict_commandContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#predict_command.
    def exitPredict_command(self, ctx:dsl_grammarParser.Predict_commandContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#error_ratio_command.
    def enterError_ratio_command(self, ctx:dsl_grammarParser.Error_ratio_commandContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#error_ratio_command.
    def exitError_ratio_command(self, ctx:dsl_grammarParser.Error_ratio_commandContext):
        pass


    # Enter a parse tree produced by dsl_grammarParser#predictors.
    def enterPredictors(self, ctx:dsl_grammarParser.PredictorsContext):
        pass

    # Exit a parse tree produced by dsl_grammarParser#predictors.
    def exitPredictors(self, ctx:dsl_grammarParser.PredictorsContext):
        pass



del dsl_grammarParser