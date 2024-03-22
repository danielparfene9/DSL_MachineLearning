# Generated from C:/Users/danie/PycharmProjects/DSL_MachineLearning/dsl_grammar.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,19,82,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,1,0,4,0,20,8,0,11,0,12,0,21,1,1,1,1,1,1,1,1,1,
        1,3,1,29,8,1,1,2,1,2,1,2,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,4,
        1,4,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,6,
        1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,8,1,8,1,8,1,8,5,8,
        75,8,8,10,8,12,8,78,9,8,1,8,1,8,1,8,0,0,9,0,2,4,6,8,10,12,14,16,
        0,1,1,0,1,4,78,0,19,1,0,0,0,2,28,1,0,0,0,4,30,1,0,0,0,6,33,1,0,0,
        0,8,42,1,0,0,0,10,44,1,0,0,0,12,56,1,0,0,0,14,63,1,0,0,0,16,70,1,
        0,0,0,18,20,3,2,1,0,19,18,1,0,0,0,20,21,1,0,0,0,21,19,1,0,0,0,21,
        22,1,0,0,0,22,1,1,0,0,0,23,29,3,6,3,0,24,29,3,10,5,0,25,29,3,12,
        6,0,26,29,3,14,7,0,27,29,3,4,2,0,28,23,1,0,0,0,28,24,1,0,0,0,28,
        25,1,0,0,0,28,26,1,0,0,0,28,27,1,0,0,0,29,3,1,0,0,0,30,31,5,7,0,
        0,31,32,5,13,0,0,32,5,1,0,0,0,33,34,5,8,0,0,34,35,5,13,0,0,35,36,
        5,19,0,0,36,37,5,9,0,0,37,38,5,8,0,0,38,39,5,14,0,0,39,40,3,8,4,
        0,40,41,5,15,0,0,41,7,1,0,0,0,42,43,7,0,0,0,43,9,1,0,0,0,44,45,5,
        10,0,0,45,46,5,5,0,0,46,47,5,13,0,0,47,48,5,14,0,0,48,49,5,13,0,
        0,49,50,5,18,0,0,50,51,3,16,8,0,51,52,5,17,0,0,52,53,5,18,0,0,53,
        54,5,13,0,0,54,55,5,15,0,0,55,11,1,0,0,0,56,57,5,11,0,0,57,58,5,
        5,0,0,58,59,5,13,0,0,59,60,5,14,0,0,60,61,5,13,0,0,61,62,5,15,0,
        0,62,13,1,0,0,0,63,64,5,12,0,0,64,65,5,5,0,0,65,66,5,13,0,0,66,67,
        5,14,0,0,67,68,5,13,0,0,68,69,5,15,0,0,69,15,1,0,0,0,70,71,5,16,
        0,0,71,76,5,13,0,0,72,73,5,18,0,0,73,75,5,13,0,0,74,72,1,0,0,0,75,
        78,1,0,0,0,76,74,1,0,0,0,76,77,1,0,0,0,77,79,1,0,0,0,78,76,1,0,0,
        0,79,80,5,17,0,0,80,17,1,0,0,0,3,21,28,76
    ]

class dsl_grammarParser ( Parser ):

    grammarFileName = "dsl_grammar.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'LinearRegression'", "'BayesianRidge'", 
                     "'SGDRegressor'", "'ElasticNet'", "'.'", "<INVALID>", 
                     "'use'", "'model'", "'new'", "'train'", "'predict'", 
                     "'error_ratio'", "<INVALID>", "'('", "')'", "'['", 
                     "']'", "','", "'='" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "WS", "USE", "MODEL", "NEW", 
                      "TRAIN", "PREDICT", "ERROR_RATIO", "ID", "LPAREN", 
                      "RPAREN", "LBRACK", "RBRACK", "COMMA", "EQUALS" ]

    RULE_dsl = 0
    RULE_statement = 1
    RULE_use_statement = 2
    RULE_model_declaration = 3
    RULE_regression_type = 4
    RULE_train_command = 5
    RULE_predict_command = 6
    RULE_error_ratio_command = 7
    RULE_predictors = 8

    ruleNames =  [ "dsl", "statement", "use_statement", "model_declaration", 
                   "regression_type", "train_command", "predict_command", 
                   "error_ratio_command", "predictors" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    WS=6
    USE=7
    MODEL=8
    NEW=9
    TRAIN=10
    PREDICT=11
    ERROR_RATIO=12
    ID=13
    LPAREN=14
    RPAREN=15
    LBRACK=16
    RBRACK=17
    COMMA=18
    EQUALS=19

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class DslContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(dsl_grammarParser.StatementContext)
            else:
                return self.getTypedRuleContext(dsl_grammarParser.StatementContext,i)


        def getRuleIndex(self):
            return dsl_grammarParser.RULE_dsl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDsl" ):
                listener.enterDsl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDsl" ):
                listener.exitDsl(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDsl" ):
                return visitor.visitDsl(self)
            else:
                return visitor.visitChildren(self)




    def dsl(self):

        localctx = dsl_grammarParser.DslContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_dsl)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 19 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 18
                self.statement()
                self.state = 21 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & 7552) != 0)):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def model_declaration(self):
            return self.getTypedRuleContext(dsl_grammarParser.Model_declarationContext,0)


        def train_command(self):
            return self.getTypedRuleContext(dsl_grammarParser.Train_commandContext,0)


        def predict_command(self):
            return self.getTypedRuleContext(dsl_grammarParser.Predict_commandContext,0)


        def error_ratio_command(self):
            return self.getTypedRuleContext(dsl_grammarParser.Error_ratio_commandContext,0)


        def use_statement(self):
            return self.getTypedRuleContext(dsl_grammarParser.Use_statementContext,0)


        def getRuleIndex(self):
            return dsl_grammarParser.RULE_statement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatement" ):
                listener.enterStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatement" ):
                listener.exitStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStatement" ):
                return visitor.visitStatement(self)
            else:
                return visitor.visitChildren(self)




    def statement(self):

        localctx = dsl_grammarParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 28
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [8]:
                self.enterOuterAlt(localctx, 1)
                self.state = 23
                self.model_declaration()
                pass
            elif token in [10]:
                self.enterOuterAlt(localctx, 2)
                self.state = 24
                self.train_command()
                pass
            elif token in [11]:
                self.enterOuterAlt(localctx, 3)
                self.state = 25
                self.predict_command()
                pass
            elif token in [12]:
                self.enterOuterAlt(localctx, 4)
                self.state = 26
                self.error_ratio_command()
                pass
            elif token in [7]:
                self.enterOuterAlt(localctx, 5)
                self.state = 27
                self.use_statement()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Use_statementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def USE(self):
            return self.getToken(dsl_grammarParser.USE, 0)

        def ID(self):
            return self.getToken(dsl_grammarParser.ID, 0)

        def getRuleIndex(self):
            return dsl_grammarParser.RULE_use_statement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUse_statement" ):
                listener.enterUse_statement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUse_statement" ):
                listener.exitUse_statement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUse_statement" ):
                return visitor.visitUse_statement(self)
            else:
                return visitor.visitChildren(self)




    def use_statement(self):

        localctx = dsl_grammarParser.Use_statementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_use_statement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 30
            self.match(dsl_grammarParser.USE)
            self.state = 31
            self.match(dsl_grammarParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Model_declarationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MODEL(self, i:int=None):
            if i is None:
                return self.getTokens(dsl_grammarParser.MODEL)
            else:
                return self.getToken(dsl_grammarParser.MODEL, i)

        def ID(self):
            return self.getToken(dsl_grammarParser.ID, 0)

        def EQUALS(self):
            return self.getToken(dsl_grammarParser.EQUALS, 0)

        def NEW(self):
            return self.getToken(dsl_grammarParser.NEW, 0)

        def LPAREN(self):
            return self.getToken(dsl_grammarParser.LPAREN, 0)

        def regression_type(self):
            return self.getTypedRuleContext(dsl_grammarParser.Regression_typeContext,0)


        def RPAREN(self):
            return self.getToken(dsl_grammarParser.RPAREN, 0)

        def getRuleIndex(self):
            return dsl_grammarParser.RULE_model_declaration

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterModel_declaration" ):
                listener.enterModel_declaration(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitModel_declaration" ):
                listener.exitModel_declaration(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitModel_declaration" ):
                return visitor.visitModel_declaration(self)
            else:
                return visitor.visitChildren(self)




    def model_declaration(self):

        localctx = dsl_grammarParser.Model_declarationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_model_declaration)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 33
            self.match(dsl_grammarParser.MODEL)
            self.state = 34
            self.match(dsl_grammarParser.ID)
            self.state = 35
            self.match(dsl_grammarParser.EQUALS)
            self.state = 36
            self.match(dsl_grammarParser.NEW)
            self.state = 37
            self.match(dsl_grammarParser.MODEL)
            self.state = 38
            self.match(dsl_grammarParser.LPAREN)
            self.state = 39
            self.regression_type()
            self.state = 40
            self.match(dsl_grammarParser.RPAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Regression_typeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return dsl_grammarParser.RULE_regression_type

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRegression_type" ):
                listener.enterRegression_type(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRegression_type" ):
                listener.exitRegression_type(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitRegression_type" ):
                return visitor.visitRegression_type(self)
            else:
                return visitor.visitChildren(self)




    def regression_type(self):

        localctx = dsl_grammarParser.Regression_typeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_regression_type)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 42
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 30) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Train_commandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TRAIN(self):
            return self.getToken(dsl_grammarParser.TRAIN, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(dsl_grammarParser.ID)
            else:
                return self.getToken(dsl_grammarParser.ID, i)

        def LPAREN(self):
            return self.getToken(dsl_grammarParser.LPAREN, 0)

        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(dsl_grammarParser.COMMA)
            else:
                return self.getToken(dsl_grammarParser.COMMA, i)

        def predictors(self):
            return self.getTypedRuleContext(dsl_grammarParser.PredictorsContext,0)


        def RBRACK(self):
            return self.getToken(dsl_grammarParser.RBRACK, 0)

        def RPAREN(self):
            return self.getToken(dsl_grammarParser.RPAREN, 0)

        def getRuleIndex(self):
            return dsl_grammarParser.RULE_train_command

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrain_command" ):
                listener.enterTrain_command(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrain_command" ):
                listener.exitTrain_command(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrain_command" ):
                return visitor.visitTrain_command(self)
            else:
                return visitor.visitChildren(self)




    def train_command(self):

        localctx = dsl_grammarParser.Train_commandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_train_command)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 44
            self.match(dsl_grammarParser.TRAIN)
            self.state = 45
            self.match(dsl_grammarParser.T__4)
            self.state = 46
            self.match(dsl_grammarParser.ID)
            self.state = 47
            self.match(dsl_grammarParser.LPAREN)
            self.state = 48
            self.match(dsl_grammarParser.ID)
            self.state = 49
            self.match(dsl_grammarParser.COMMA)
            self.state = 50
            self.predictors()
            self.state = 51
            self.match(dsl_grammarParser.RBRACK)
            self.state = 52
            self.match(dsl_grammarParser.COMMA)
            self.state = 53
            self.match(dsl_grammarParser.ID)
            self.state = 54
            self.match(dsl_grammarParser.RPAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Predict_commandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PREDICT(self):
            return self.getToken(dsl_grammarParser.PREDICT, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(dsl_grammarParser.ID)
            else:
                return self.getToken(dsl_grammarParser.ID, i)

        def LPAREN(self):
            return self.getToken(dsl_grammarParser.LPAREN, 0)

        def RPAREN(self):
            return self.getToken(dsl_grammarParser.RPAREN, 0)

        def getRuleIndex(self):
            return dsl_grammarParser.RULE_predict_command

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPredict_command" ):
                listener.enterPredict_command(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPredict_command" ):
                listener.exitPredict_command(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPredict_command" ):
                return visitor.visitPredict_command(self)
            else:
                return visitor.visitChildren(self)




    def predict_command(self):

        localctx = dsl_grammarParser.Predict_commandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_predict_command)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 56
            self.match(dsl_grammarParser.PREDICT)
            self.state = 57
            self.match(dsl_grammarParser.T__4)
            self.state = 58
            self.match(dsl_grammarParser.ID)
            self.state = 59
            self.match(dsl_grammarParser.LPAREN)
            self.state = 60
            self.match(dsl_grammarParser.ID)
            self.state = 61
            self.match(dsl_grammarParser.RPAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Error_ratio_commandContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ERROR_RATIO(self):
            return self.getToken(dsl_grammarParser.ERROR_RATIO, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(dsl_grammarParser.ID)
            else:
                return self.getToken(dsl_grammarParser.ID, i)

        def LPAREN(self):
            return self.getToken(dsl_grammarParser.LPAREN, 0)

        def RPAREN(self):
            return self.getToken(dsl_grammarParser.RPAREN, 0)

        def getRuleIndex(self):
            return dsl_grammarParser.RULE_error_ratio_command

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterError_ratio_command" ):
                listener.enterError_ratio_command(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitError_ratio_command" ):
                listener.exitError_ratio_command(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitError_ratio_command" ):
                return visitor.visitError_ratio_command(self)
            else:
                return visitor.visitChildren(self)




    def error_ratio_command(self):

        localctx = dsl_grammarParser.Error_ratio_commandContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_error_ratio_command)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 63
            self.match(dsl_grammarParser.ERROR_RATIO)
            self.state = 64
            self.match(dsl_grammarParser.T__4)
            self.state = 65
            self.match(dsl_grammarParser.ID)
            self.state = 66
            self.match(dsl_grammarParser.LPAREN)
            self.state = 67
            self.match(dsl_grammarParser.ID)
            self.state = 68
            self.match(dsl_grammarParser.RPAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PredictorsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LBRACK(self):
            return self.getToken(dsl_grammarParser.LBRACK, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(dsl_grammarParser.ID)
            else:
                return self.getToken(dsl_grammarParser.ID, i)

        def RBRACK(self):
            return self.getToken(dsl_grammarParser.RBRACK, 0)

        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(dsl_grammarParser.COMMA)
            else:
                return self.getToken(dsl_grammarParser.COMMA, i)

        def getRuleIndex(self):
            return dsl_grammarParser.RULE_predictors

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPredictors" ):
                listener.enterPredictors(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPredictors" ):
                listener.exitPredictors(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPredictors" ):
                return visitor.visitPredictors(self)
            else:
                return visitor.visitChildren(self)




    def predictors(self):

        localctx = dsl_grammarParser.PredictorsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_predictors)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 70
            self.match(dsl_grammarParser.LBRACK)
            self.state = 71
            self.match(dsl_grammarParser.ID)
            self.state = 76
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==18:
                self.state = 72
                self.match(dsl_grammarParser.COMMA)
                self.state = 73
                self.match(dsl_grammarParser.ID)
                self.state = 78
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 79
            self.match(dsl_grammarParser.RBRACK)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





