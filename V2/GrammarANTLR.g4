grammar GrammarANTLR;

start: (load | model | train | predict) EOF;

load: 'load' FILENAME;
model: 'model' MODEL_NAME '=' 'new' 'model' '(' FILENAME ')';
train: MODEL_NAME '.' 'train' '(' ')';
predict: MODEL_NAME '.' 'predict' '(' ')';

FILENAME: [a-zA-Z0-9_]+ '.csv';
MODEL_NAME: [a-zA-Z]+;
