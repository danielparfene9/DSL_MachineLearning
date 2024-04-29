# Generated from C:/Users/danie/PycharmProjects/DSL_MachineLearning/V2/GrammarANTLR.g4 by ANTLR 4.13.1
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
        4,1,11,43,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,1,0,1,0,1,0,1,
        0,3,0,15,8,0,1,0,1,0,1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
        1,2,1,3,1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,0,0,5,0,
        2,4,6,8,0,0,40,0,14,1,0,0,0,2,18,1,0,0,0,4,21,1,0,0,0,6,30,1,0,0,
        0,8,36,1,0,0,0,10,15,3,2,1,0,11,15,3,4,2,0,12,15,3,6,3,0,13,15,3,
        8,4,0,14,10,1,0,0,0,14,11,1,0,0,0,14,12,1,0,0,0,14,13,1,0,0,0,15,
        16,1,0,0,0,16,17,5,0,0,1,17,1,1,0,0,0,18,19,5,1,0,0,19,20,5,10,0,
        0,20,3,1,0,0,0,21,22,5,2,0,0,22,23,5,11,0,0,23,24,5,3,0,0,24,25,
        5,4,0,0,25,26,5,2,0,0,26,27,5,5,0,0,27,28,5,10,0,0,28,29,5,6,0,0,
        29,5,1,0,0,0,30,31,5,11,0,0,31,32,5,7,0,0,32,33,5,8,0,0,33,34,5,
        5,0,0,34,35,5,6,0,0,35,7,1,0,0,0,36,37,5,11,0,0,37,38,5,7,0,0,38,
        39,5,9,0,0,39,40,5,5,0,0,40,41,5,6,0,0,41,9,1,0,0,0,1,14
    ]

class GrammarANTLRParser ( Parser ):

    grammarFileName = "GrammarANTLR.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'load'", "'model'", "'='", "'new'", "'('", 
                     "')'", "'.'", "'train'", "'predict'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "FILENAME", "MODEL_NAME" ]

    RULE_start = 0
    RULE_load = 1
    RULE_model = 2
    RULE_train = 3
    RULE_predict = 4

    ruleNames =  [ "start", "load", "model", "train", "predict" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    FILENAME=10
    MODEL_NAME=11

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class StartContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(GrammarANTLRParser.EOF, 0)

        def load(self):
            return self.getTypedRuleContext(GrammarANTLRParser.LoadContext,0)


        def model(self):
            return self.getTypedRuleContext(GrammarANTLRParser.ModelContext,0)


        def train(self):
            return self.getTypedRuleContext(GrammarANTLRParser.TrainContext,0)


        def predict(self):
            return self.getTypedRuleContext(GrammarANTLRParser.PredictContext,0)


        def getRuleIndex(self):
            return GrammarANTLRParser.RULE_start

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStart" ):
                listener.enterStart(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStart" ):
                listener.exitStart(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStart" ):
                return visitor.visitStart(self)
            else:
                return visitor.visitChildren(self)




    def start(self):

        localctx = GrammarANTLRParser.StartContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_start)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 14
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.state = 10
                self.load()
                pass

            elif la_ == 2:
                self.state = 11
                self.model()
                pass

            elif la_ == 3:
                self.state = 12
                self.train()
                pass

            elif la_ == 4:
                self.state = 13
                self.predict()
                pass


            self.state = 16
            self.match(GrammarANTLRParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LoadContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FILENAME(self):
            return self.getToken(GrammarANTLRParser.FILENAME, 0)

        def getRuleIndex(self):
            return GrammarANTLRParser.RULE_load

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLoad" ):
                listener.enterLoad(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLoad" ):
                listener.exitLoad(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLoad" ):
                return visitor.visitLoad(self)
            else:
                return visitor.visitChildren(self)




    def load(self):

        localctx = GrammarANTLRParser.LoadContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_load)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 18
            self.match(GrammarANTLRParser.T__0)
            self.state = 19
            self.match(GrammarANTLRParser.FILENAME)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ModelContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MODEL_NAME(self):
            return self.getToken(GrammarANTLRParser.MODEL_NAME, 0)

        def FILENAME(self):
            return self.getToken(GrammarANTLRParser.FILENAME, 0)

        def getRuleIndex(self):
            return GrammarANTLRParser.RULE_model

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterModel" ):
                listener.enterModel(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitModel" ):
                listener.exitModel(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitModel" ):
                return visitor.visitModel(self)
            else:
                return visitor.visitChildren(self)




    def model(self):

        localctx = GrammarANTLRParser.ModelContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_model)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 21
            self.match(GrammarANTLRParser.T__1)
            self.state = 22
            self.match(GrammarANTLRParser.MODEL_NAME)
            self.state = 23
            self.match(GrammarANTLRParser.T__2)
            self.state = 24
            self.match(GrammarANTLRParser.T__3)
            self.state = 25
            self.match(GrammarANTLRParser.T__1)
            self.state = 26
            self.match(GrammarANTLRParser.T__4)
            self.state = 27
            self.match(GrammarANTLRParser.FILENAME)
            self.state = 28
            self.match(GrammarANTLRParser.T__5)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TrainContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MODEL_NAME(self):
            return self.getToken(GrammarANTLRParser.MODEL_NAME, 0)

        def getRuleIndex(self):
            return GrammarANTLRParser.RULE_train

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrain" ):
                listener.enterTrain(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrain" ):
                listener.exitTrain(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTrain" ):
                return visitor.visitTrain(self)
            else:
                return visitor.visitChildren(self)




    def train(self):

        localctx = GrammarANTLRParser.TrainContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_train)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 30
            self.match(GrammarANTLRParser.MODEL_NAME)
            self.state = 31
            self.match(GrammarANTLRParser.T__6)
            self.state = 32
            self.match(GrammarANTLRParser.T__7)
            self.state = 33
            self.match(GrammarANTLRParser.T__4)
            self.state = 34
            self.match(GrammarANTLRParser.T__5)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PredictContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MODEL_NAME(self):
            return self.getToken(GrammarANTLRParser.MODEL_NAME, 0)

        def getRuleIndex(self):
            return GrammarANTLRParser.RULE_predict

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPredict" ):
                listener.enterPredict(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPredict" ):
                listener.exitPredict(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPredict" ):
                return visitor.visitPredict(self)
            else:
                return visitor.visitChildren(self)




    def predict(self):

        localctx = GrammarANTLRParser.PredictContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_predict)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 36
            self.match(GrammarANTLRParser.MODEL_NAME)
            self.state = 37
            self.match(GrammarANTLRParser.T__6)
            self.state = 38
            self.match(GrammarANTLRParser.T__8)
            self.state = 39
            self.match(GrammarANTLRParser.T__4)
            self.state = 40
            self.match(GrammarANTLRParser.T__5)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





