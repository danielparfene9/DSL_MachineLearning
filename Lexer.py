class Token:
    def __init__(self, type, value=None, pos_start=None, pos_end=None):
        self.type = type
        self.value = value
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __str__(self):
        return f'Token({self.type}, {self.value})'

    def __repr__(self):
        return self.__str__()


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.keywords = {
            'import': Token('IMPORT', 'import'),
            'model': Token('MODEL', 'model'),
            'new': Token('NEW', 'creation'),
            'csv': Token('EXTENSION', 'csv'),
            'train': Token('TRAIN', 'training'),
            'predict': Token('PREDICT', 'prediction'),
            'error_ratio': Token('ERROR_RATIO', 'error_ratio_handling'),
            'LinearRegression': Token('REGRESSION_TYPE', 'LinearRegression'),
            'BayesianRidge': Token('REGRESSION_TYPE', 'BayesianRidge'),
            'SGDRegressor': Token('REGRESSION_TYPE', 'SGDRegressor'),
            'ElasticNet': Token('REGRESSION_TYPE', 'ElasticNet'),

        }

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        else:
            return None

    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def identifier(self):
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        return result

    def string(self):
        result = ''
        self.advance()  # Consume opening double quote
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()
        if self.current_char is None:
            self.error()  # String literal not terminated
        self.advance()  # Consume closing double quote
        return Token('STRING', result)

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha() or self.current_char == '_':
                identifier = self.identifier()
                return self.keywords.get(identifier, Token('IDENTIFIER', identifier))

            if self.current_char == '"':
                return Token('STRING', self.string())

            if self.current_char == '=':
                self.advance()
                return Token('ASSIGN', '=')

            if self.current_char == '(':
                self.advance()
                return Token('LPAREN', '(')

            if self.current_char == ')':
                self.advance()
                return Token('RPAREN', ')')

            if self.current_char == ',':
                self.advance()
                return Token('COMMA', ',')

            if self.current_char == '.':
                self.advance()
                return Token('DOT', '.')

            if self.current_char == '[':
                self.advance()
                return Token('LSQPAREN', '[')

            if self.current_char == ']':
                self.advance()
                return Token('RSQPAREN', ']')

            self.error()

        return Token('EOF')


class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def parse(self):
        return self.dsl()

    def dsl(self):
        statements = []
        while self.current_token.type != 'EOF':
            statement = self.statement()
            if statement:
                statements.append(statement)
        return statements

    def string(self):
        token_value = self.current_token.value
        self.eat('STRING')
        return token_value

    def assignment_statement(self):
        identifier = self.identifier()
        self.eat('ASSIGN')
        if self.current_token.value == 'import':
            self.eat('IMPORT')
            self.eat('LPAREN')
            file_path = self.string()
            self.eat('RPAREN')
            return {'type': 'ImportStatement', 'identifier': identifier, 'file_path': file_path}
        else:
            self.error()

    def statement(self):
        if self.current_token.type == 'MODEL':
            return self.model_declaration()
        elif self.current_token.type == 'TRAIN':
            return self.train_command()
        elif self.current_token.type == 'PREDICT':
            return self.predict_command()
        elif self.current_token.type == 'ERROR_RATIO':
            return self.error_ratio_command()
        elif self.current_token.type == 'IDENTIFIER':
            return self.assignment_statement()
        else:
            self.error()

    def model_declaration(self):
        self.eat('MODEL')
        identifier = self.identifier()
        self.eat('ASSIGN')
        self.eat('NEW')
        self.eat('MODEL')
        self.eat('LPAREN')
        regression_type = self.regression_type()
        self.eat('RPAREN')
        return {'type': 'ModelDeclaration', 'identifier': identifier, 'regression_type': regression_type}

    def regression_type(self):
        token_value = self.current_token.value
        self.eat('REGRESSION_TYPE')
        return token_value

    def train_command(self):
        self.eat('TRAIN')
        self.eat('DOT')
        identifier = self.identifier()
        self.eat('LPAREN')
        dataframe = self.dataframe()
        self.eat('COMMA')
        predictors = self.predictors()
        self.eat('COMMA')
        target = self.target()
        self.eat('RPAREN')
        return {'type': 'TrainCommand', 'identifier': identifier, 'dataframe': dataframe, 'predictors': predictors, 'target': target}

    def predict_command(self):
        self.eat('PREDICT')
        self.eat('DOT')
        identifier = self.identifier()
        self.eat('LPAREN')
        dataframe = self.dataframe()
        self.eat('RPAREN')
        return {'type': 'PredictCommand', 'identifier': identifier, 'dataframe': dataframe}

    def error_ratio_command(self):
        self.eat('ERROR_RATIO')
        self.eat('DOT')
        identifier = self.identifier()
        self.eat('LPAREN')
        dataframe = self.dataframe()
        self.eat('RPAREN')
        return {'type': 'ErrorRatioCommand', 'identifier': identifier, 'dataframe': dataframe}

    def dataframe(self):
        return self.identifier()

    def predictors(self):
        self.eat('LSQPAREN')
        columns = []
        while self.current_token.type != 'RSQPAREN':
            columns.append(self.identifier())
            if self.current_token.type == 'COMMA':
                self.eat('COMMA')
        self.eat('RSQPAREN')
        return columns

    def target(self):
        return self.identifier()

    def identifier(self):
        token_value = self.current_token.value
        self.eat('IDENTIFIER')
        return token_value


text = """
    df = import(data.csv)
    model my_model = new model(LinearRegression)
    train.my_model(df, [col1, col2], target)
    predict.my_model(df)
    error_ratio.my_model(df)
    """

lexer = Lexer(text)
tokens = []
while True:
    token = lexer.get_next_token()
    print(token)
    tokens.append(token)
    if token.type == 'EOF':
        break

lexer = Lexer(text)
parser = Parser(lexer)
parsed_result = parser.parse()
print(parsed_result)

