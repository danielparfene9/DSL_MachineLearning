# Generated from C:/Users/danie/PycharmProjects/DSL_MachineLearning/dsl_grammar.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .dsl_grammarParser import dsl_grammarParser
else:
    from dsl_grammarParser import dsl_grammarParser

# This class defines a complete generic visitor for a parse tree produced by dsl_grammarParser.

class dsl_grammarVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by dsl_grammarParser#dsl.
    def visitDsl(self, ctx:dsl_grammarParser.DslContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#statement.
    def visitStatement(self, ctx:dsl_grammarParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#use_statement.
    def visitUse_statement(self, ctx:dsl_grammarParser.Use_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#model_declaration.
    def visitModel_declaration(self, ctx:dsl_grammarParser.Model_declarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#regression_type.
    def visitRegression_type(self, ctx:dsl_grammarParser.Regression_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#train_command.
    def visitTrain_command(self, ctx:dsl_grammarParser.Train_commandContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#predict_command.
    def visitPredict_command(self, ctx:dsl_grammarParser.Predict_commandContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#error_ratio_command.
    def visitError_ratio_command(self, ctx:dsl_grammarParser.Error_ratio_commandContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dsl_grammarParser#predictors.
    def visitPredictors(self, ctx:dsl_grammarParser.PredictorsContext):
        return self.visitChildren(ctx)



del dsl_grammarParser