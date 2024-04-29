# Generated from C:/Users/danie/PycharmProjects/DSL_MachineLearning/V2/GrammarANTLR.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .GrammarANTLRParser import GrammarANTLRParser
else:
    from GrammarANTLRParser import GrammarANTLRParser

# This class defines a complete generic visitor for a parse tree produced by GrammarANTLRParser.

class GrammarANTLRVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by GrammarANTLRParser#start.
    def visitStart(self, ctx:GrammarANTLRParser.StartContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by GrammarANTLRParser#load.
    def visitLoad(self, ctx:GrammarANTLRParser.LoadContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by GrammarANTLRParser#model.
    def visitModel(self, ctx:GrammarANTLRParser.ModelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by GrammarANTLRParser#train.
    def visitTrain(self, ctx:GrammarANTLRParser.TrainContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by GrammarANTLRParser#predict.
    def visitPredict(self, ctx:GrammarANTLRParser.PredictContext):
        return self.visitChildren(ctx)



del GrammarANTLRParser