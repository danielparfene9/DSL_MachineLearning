grammar dsl_grammar;

// Lexer rules
WS: [ \t\r\n]+ -> skip; // Ignore whitespace

// Keywords
USE: 'use';
MODEL: 'model';
NEW: 'new';
TRAIN: 'train';
PREDICT: 'predict';
ERROR_RATIO: 'error_ratio';

// Identifiers
ID: [a-zA-Z_][a-zA-Z0-9_]*;

// Punctuation
LPAREN: '(';
RPAREN: ')';
LBRACK: '[';
RBRACK: ']';
COMMA: ',';
EQUALS: '=';

// Parser rules
dsl: statement+;

statement: model_declaration | train_command | predict_command | error_ratio_command | use_statement;

use_statement: USE ID;

model_declaration: MODEL ID EQUALS NEW MODEL LPAREN regression_type RPAREN;

regression_type: 'LinearRegression';

train_command: TRAIN '.' ID LPAREN ID COMMA predictors RBRACK COMMA ID RPAREN;

predict_command: PREDICT '.' ID LPAREN ID RPAREN;

error_ratio_command: ERROR_RATIO '.' ID LPAREN ID RPAREN;

predictors: LBRACK ID (COMMA ID)* RBRACK;
