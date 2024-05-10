import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.impute import SimpleImputer
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
import joblib


class ModelTrainer:
    def __init__(self, data, target_col):
        self.data = pd.read_csv(data)
        self.target_col = target_col
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

### ДОБАВИЛ ПАРАМЕТР missing_values. Если пользователь выберет допустим что у него они есть, то нужно будет поменять на True
    def populate_x_y(self, missing_values=False):

        target_col = self.target_col

        # ЭТО НАМ СКАЗАЛИ СДЕЛАТЬ НА ВЫБОР ПОЛЬЗОВАТЕЛЯ
        # Вставляет значения основываясь на срендем арифметическом

        if missing_values:
            if self.data[target_col].isnull().any():
                imputer = SimpleImputer(strategy='mean')
                self.data[target_col] = imputer.fit_transform(self.data[target_col].values.reshape(-1, 1)).flatten()

        self.data = self.data.dropna(subset=[target_col])

        self.data = self.handle_dates(self.data)

        self.X = self.data.drop(target_col, axis=1)
        self.y = self.data[target_col]

    def split_data(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    ### ЭТО КОММАНДЫ ДЛЯ ПРЕДОБРАБОТКИ НА ВЫБОР ПОЛЬЗОВАТЕЛЯ
    # def scaling(self):
    #
    #     scaler = StandardScaler()
    #     scaler.fit(self.X_train)
    #     self.X_train = scaler.transform(self.X_train)
    #     self.X_test = scaler.transform(self.X_test)

    # ВОТ ЭТОТ НУЖНО ИСПОЛЬЗОВАТЬ ТОЛЬКО ПОСЛЕ Populate_x_y перед split_data
    # def principal_component_analysis(self):
    #
    #     pca = PCA(n_components=0.7)
    #     self.X = pca.fit_transform(self.X)
    #
    # def z_score_normalization(self):
    #     mean = self.X_train.mean(axis=0)
    #     std_dev = self.X_train.std(axis=0)
    #     self.X_train = (self.X_train - mean) / std_dev
    #     self.X_test = (self.X_test - mean) / std_dev

    # ВОТ ЭТИ ДВЕ СЛЕДУЮЩИЕ ТОЛЬКО ДЛЯ КЛАСИФИКАЦИИ
    # def one_hot_encoding(self):
    #     encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    #     self.X_train = encoder.fit_transform(self.X_train)
    #     self.X_test = encoder.transform(self.X_test)
    #
    # def label_encoding(self):
    #     encoder = LabelEncoder()
    #     self.X_train = encoder.fit_transform(self.X_train)
    #     self.X_test = encoder.transform(self.X_test)
    #
    # def min_max_scaling(self):
    #     scaler = MinMaxScaler()
    #     self.X_train = scaler.fit_transform(self.X_train)
    #     self.X_test = scaler.transform(self.X_test)

# Я НЕ СТАЛ ДЕЛАТЬ ОТДЕЛЬНУЮ ФУНКЦИЮ ГДЕ ЗАКИНУ ВОТ ЭТИ ВСЕ МЕТОДЫ, РЕШИЛ ЧТО БУДЕТ ЛЕГЧЕ ВНУТРИ НИХ УЖЕ ВНЕДРИТЬ
# МЕТОДЫ ПРЕДОБРАБОТКИ СВЕРХУ ДЛЯ КАЖДОГО ОТДЕЛЬНО ТК ЕСТЬ ТЕ КОТОРЫЕ МОЖНО ИСПОЛЬЗОВАТЬ ТОЛЬКО С КЛАССИФИКАЦИЕИ
# ВСЕ ЧТО ЗАКОМЕНТИРОВАНО СДЕЛАНО НА ВЫБОР ПОЛЬЗОВАТЕЛЯ

    def preprocess_data_lin_reg(self):

        self.populate_x_y()
        # self.principal_component_analysis()
        self.split_data()
        # self.scaling()
        # self.min_max_scaling()
        # self.z_score_normalization()

    def preprocess_data_rfr(self):

        self.populate_x_y()
        # self.principal_component_analysis()
        self.split_data()
        # self.scaling()
        # self.min_max_scaling()
        # self.z_score_normalization()

    def preprocess_data_svr(self):

        self.populate_x_y()
        # self.principal_component_analysis()
        self.split_data()
        # self.scaling()
        # self.min_max_scaling()
        # self.z_score_normalization()

    def preprocess_data_rfc(self):

        self.populate_x_y()
        # self.principal_component_analysis()
        self.split_data()
        # self.scaling()
        # self.min_max_scaling()
        # self.z_score_normalization()
        # self.one_hot_encoding()
        # self.label_encoding()

    def preprocess_data_log_reg(self):

        self.populate_x_y()
        # self.principal_component_analysis()
        self.split_data()
        # self.scaling()
        # self.min_max_scaling()
        # self.z_score_normalization()
        # self.one_hot_encoding()
        # self.label_encoding()

    def preprocess_data_svc(self):

        self.populate_x_y()
        # self.principal_component_analysis()
        self.split_data()
        # self.scaling()
        # self.min_max_scaling()
        # self.z_score_normalization()
        # self.one_hot_encoding()
        # self.label_encoding()

    def handle_dates(self, df):
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                print(f"Column '{col}' could not be converted to datetime.")

        for col in date_cols:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = (df[col] - df[col].min()).dt.days

        return df

### ДОБАВИЛ ПАРАМЕТР save_model ( в формате pkl ) через который будем сохранять саму модель после того как затрейнили
    def train_linear_regression(self, save_model='lin_reg.pkl'):
        if self.is_classification_target():
            raise ValueError("Cannot train Linear Regression model with classification target.")
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
### ЕСЛИ БЫЛО НАЗНАЧЕНО НАЗВАНИЕ ФАЙЛА И ПУТЬ КУДА СОХРАНИТЬ ТО МЫ СОХРАНЯЕМ
        if save_model:
            joblib.dump(model, save_model)
        return model

    def load_model(self, model_path):
        model = joblib.load(model_path)
        return model

    def train_random_forest_regression(self):
        if self.is_classification_target():
            raise ValueError("Cannot train Random Forest Regression model with classification target.")
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        return model

    def train_support_vector_regression(self):
        if self.is_classification_target():
            raise ValueError("Cannot train Support Vector Regression model with classification target.")
        model = SVR()
        model.fit(self.X_train, self.y_train)
        return model

    def train_logistic_regression(self):
        if not self.is_classification_target():
            raise ValueError("Cannot train Logistic Regression model with regression target.")
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        return model

    def train_random_forest_classifier(self):
        if not self.is_classification_target():
            raise ValueError("Cannot train Random Forest Classifier model with regression target.")
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        return model

    def train_support_vector_classifier(self):
        if not self.is_classification_target():
            raise ValueError("Cannot train Support Vector Classifier model with regression target.")
        model = SVC()
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate_regression_model(self, model):
        y_pred = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        return y_pred, mae

    def evaluate_classification_model(self, model):
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return y_pred, accuracy

    def is_classification_target(self):
        return self.y_train.dtype not in [int, float]


model_trainer = ModelTrainer('Sales.csv', 'sales')

#### Линейная Регрессия ( Самая Быстрая )
# try:
#     model_trainer.preprocess_data_lin_reg()
#     linear_regression_model = model_trainer.train_linear_regression()
#     y_pred_lr, mae_lr = model_trainer.evaluate_regression_model(linear_regression_model)
#     print("Linear Regression - Mean Absolute Error:", mae_lr)
#     print("Linear Regression - Predicted Values:", y_pred_lr)
# except ValueError as e:
#     print(e)

### ТЕПЕРЬ ЕСЛИ МЫ ЗАГРУЖАЕМ МОДЕЛЬ НУЖНО СДЕЛАТЬ ВОТ ТАК ОТДЕЛЬНО
try:
    model_trainer.preprocess_data_lin_reg()
    linear_regression_model = model_trainer.load_model('lin_reg.pkl')
    y_pred_lr, mae_lr = model_trainer.evaluate_regression_model(linear_regression_model)
    print("Linear Regression - Mean Absolute Error:", mae_lr)
    print("Linear Regression - Predicted Values:", y_pred_lr)
except ValueError as e:
    print(e)

### Строит деверво ( +- 3-4 минукты )
# try:
#     model_trainer.preprocess_data_rfr()
#     random_forest_regression_model = model_trainer.train_random_forest_regression()
#     y_pred_rf, mae_rf = model_trainer.evaluate_regression_model(random_forest_regression_model)
#     print("Random Forest Regression - Mean Absolute Error:", mae_rf)
#     print("Random Forest Regression - Predicted Values:", y_pred_rf)
# except ValueError as e:
#     print(e)

#### Строит вектора в 2д виде ( +- 8 минукт )
# try:
#     model_trainer.preprocess_data_svr()
#     support_vector_regression_model = model_trainer.train_support_vector_regression()
#     y_pred_svr, mae_svr = model_trainer.evaluate_regression_model(support_vector_regression_model)
#     print("Support Vector Regression - Mean Absolute Error:", mae_svr)
#     print("Support Vector Regression - Predicted Values:", y_pred_svr)
# except ValueError as e:
#     print(e)


#### Тоже самое только для класификации
# try:
#     model_trainer.preprocess_data_log_reg()
#     logistic_regression_model = model_trainer.train_logistic_regression()
#     y_pred_logistic, accuracy_logistic = model_trainer.evaluate_classification_model(logistic_regression_model)
#     print("Logistic Regression - Accuracy:", accuracy_logistic)
#     print("Logistic Regression - Predicted Values:", y_pred_logistic)
# except ValueError as e:
#     print(e)
#### Тоже самое только для класификации
# try:
#     model_trainer.preprocess_data_rfc()
#     random_forest_classifier_model = model_trainer.train_random_forest_classifier()
#     y_pred_rf_class, accuracy_rf_class = model_trainer.evaluate_classification_model(random_forest_classifier_model)
#     print("Random Forest Classifier - Accuracy:", accuracy_rf_class)
#     print("Random Forest Classifier - Predicted Values:", y_pred_rf_class)
# except ValueError as e:
#     print(e)


#### ДОЛГО
# try:
#     model_trainer.preprocess_data_svc()
#     support_vector_classifier_model = model_trainer.train_support_vector_classifier()
#     y_pred_svc, accuracy_svc = model_trainer.evaluate_classification_model(support_vector_classifier_model)
#     print("Support Vector Classifier - Accuracy:", accuracy_svc)
#     print("Support Vector Classifier - Predicted Values:", y_pred_svc)
# except ValueError as e:
#     print(e)
