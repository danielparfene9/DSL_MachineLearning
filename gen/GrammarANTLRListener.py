# Generated from C:/Users/danie/PycharmProjects/DSL_MachineLearning/V2/GrammarANTLR.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .GrammarANTLRParser import GrammarANTLRParser
else:
    from GrammarANTLRParser import GrammarANTLRParser

# This class defines a complete listener for a parse tree produced by GrammarANTLRParser.
class GrammarANTLRListener(ParseTreeListener):

    # Enter a parse tree produced by GrammarANTLRParser#start.
    def enterStart(self, ctx:GrammarANTLRParser.StartContext):
        pass

    # Exit a parse tree produced by GrammarANTLRParser#start.
    def exitStart(self, ctx:GrammarANTLRParser.StartContext):
        pass


    # Enter a parse tree produced by GrammarANTLRParser#load.
    def enterLoad(self, ctx:GrammarANTLRParser.LoadContext):
        pass

    # Exit a parse tree produced by GrammarANTLRParser#load.
    def exitLoad(self, ctx:GrammarANTLRParser.LoadContext):
        pass


    # Enter a parse tree produced by GrammarANTLRParser#model.
    def enterModel(self, ctx:GrammarANTLRParser.ModelContext):
        pass

    # Exit a parse tree produced by GrammarANTLRParser#model.
    def exitModel(self, ctx:GrammarANTLRParser.ModelContext):
        pass


    # Enter a parse tree produced by GrammarANTLRParser#train.
    def enterTrain(self, ctx:GrammarANTLRParser.TrainContext):
        pass

    # Exit a parse tree produced by GrammarANTLRParser#train.
    def exitTrain(self, ctx:GrammarANTLRParser.TrainContext):
        pass


    # Enter a parse tree produced by GrammarANTLRParser#predict.
    def enterPredict(self, ctx:GrammarANTLRParser.PredictContext):
        pass

    # Exit a parse tree produced by GrammarANTLRParser#predict.
    def exitPredict(self, ctx:GrammarANTLRParser.PredictContext):
        pass



del GrammarANTLRParser