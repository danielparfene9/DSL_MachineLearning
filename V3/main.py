import warnings

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
import os


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

# ДОБАВИЛ ПАРАМЕТР missing_values. Если пользователь выберет допустим что у него они есть,
    # то нужно будет поменять на True
    def populate_x_y(self, missing_values=False):

        target_col = self.target_col

        # ЭТО НАМ СКАЗАЛИ СДЕЛАТЬ НА ВЫБОР ПОЛЬЗОВАТЕЛЯ
        # вставляет значения основываясь на среднем арифметическом

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

    # ЭТО КОММАНДЫ ДЛЯ ПРЕДОБРАБОТКИ НА ВЫБОР ПОЛЬЗОВАТЕЛЯ
    def scaling(self):

        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def principal_component_analysis(self):

        pca = PCA(n_components=0.7)
        self.X = pca.fit_transform(self.X)

    def z_score_normalization(self):
        mean = self.X_train.mean(axis=0)
        std_dev = self.X_train.std(axis=0)
        self.X_train = (self.X_train - mean) / std_dev
        self.X_test = (self.X_test - mean) / std_dev

    def one_hot_encoding(self):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.X_train = encoder.fit_transform(self.X_train)
        self.X_test = encoder.transform(self.X_test)

    def label_encoding(self):
        encoder = LabelEncoder()
        self.X_train = encoder.fit_transform(self.X_train)
        self.X_test = encoder.transform(self.X_test)

    def min_max_scaling(self):
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def preprocess_data_lin_reg(self, pca=False, scaling=False, min_max=False, z_score=False, inputer=False):

        self.populate_x_y(inputer)
        if pca:
            self.principal_component_analysis()
        self.split_data()
        if scaling:
            self.scaling()
        if min_max:
            self.min_max_scaling()
        if z_score:
            self.z_score_normalization()

    def preprocess_data_rfr(self, pca=False, scaling=False, min_max=False, z_score=False, inputer=False):

        self.populate_x_y(inputer)
        if pca:
            self.principal_component_analysis()
        self.split_data()
        if scaling:
            self.scaling()
        if min_max:
            self.min_max_scaling()
        if z_score:
            self.z_score_normalization()

    def preprocess_data_svr(self, pca=False, scaling=False, min_max=False, z_score=False, inputer=False):

        self.populate_x_y(inputer)
        if pca:
            self.principal_component_analysis()
        self.split_data()
        if scaling:
            self.scaling()
        if min_max:
            self.min_max_scaling()
        if z_score:
            self.z_score_normalization()

    def preprocess_data_rfc(self, pca=False, scaling=False, min_max=False, z_score=False, one_hot=False, label=False
                            , inputer=False):

        self.populate_x_y(inputer)
        if pca:
            self.principal_component_analysis()
        self.split_data()
        if scaling:
            self.scaling()
        if min_max:
            self.min_max_scaling()
        if z_score:
            self.z_score_normalization()
        if one_hot:
            self.one_hot_encoding()
        if label:
            self.label_encoding()

    def preprocess_data_log_reg(self, pca=False, scaling=False, min_max=False, z_score=False, one_hot=False, label=False
                                , inputer=False):

        self.populate_x_y(inputer)
        if pca:
            self.principal_component_analysis()
        self.split_data()
        if scaling:
            self.scaling()
        if min_max:
            self.min_max_scaling()
        if z_score:
            self.z_score_normalization()
        if one_hot:
            self.one_hot_encoding()
        if label:
            self.label_encoding()

    def preprocess_data_svc(self, pca=False, scaling=False, min_max=False, z_score=False, one_hot=False, label=False
                            , inputer=False):

        self.populate_x_y(inputer)
        if pca:
            self.principal_component_analysis()
        self.split_data()
        if scaling:
            self.scaling()
        if min_max:
            self.min_max_scaling()
        if z_score:
            self.z_score_normalization()
        if one_hot:
            self.one_hot_encoding()
        if label:
            self.label_encoding()

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

    def load_model(self, model_path):
        model = joblib.load(model_path)
        return model

    def train_linear_regression(self, save_model=False):
        if self.is_classification_target():
            raise ValueError("Cannot train Linear Regression model with classification target.")
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        if save_model:
            index = 0
            while True:
                filename = f"{'lin_reg'}_{index}.pkl"
                if not os.path.exists(filename):
                    break
                index += 1
            joblib.dump(model, filename)
        return model

    def train_random_forest_regression(self, save_model=False):
        if self.is_classification_target():
            raise ValueError("Cannot train Random Forest Regression model with classification target.")
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)

        if save_model:
            index = 0
            while True:
                filename = f"{'random_forest_reg'}_{index}.pkl"
                if not os.path.exists(filename):
                    break
                index += 1
            joblib.dump(model, filename)
        return model

    def train_support_vector_regression(self, save_model=False):
        if self.is_classification_target():
            raise ValueError("Cannot train Support Vector Regression model with classification target.")
        model = SVR()
        model.fit(self.X_train, self.y_train)

        if save_model:
            index = 0
            while True:
                filename = f"{'svr_reg'}_{index}.pkl"
                if not os.path.exists(filename):
                    break
                index += 1
            joblib.dump(model, filename)
        return model

    def train_logistic_regression(self, save_model=False):
        if not self.is_classification_target():
            raise ValueError("Cannot train Logistic Regression model with regression target.")
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)

        if save_model:
            index = 0
            while True:
                filename = f"{'logistic_reg_cl'}_{index}.pkl"
                if not os.path.exists(filename):
                    break
                index += 1
            joblib.dump(model, filename)
        return model

    def train_random_forest_classifier(self, save_model=False):
        if not self.is_classification_target():
            raise ValueError("Cannot train Random Forest Classifier model with regression target.")
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)

        if save_model:
            index = 0
            while True:
                filename = f"{'random_forest_cl'}_{index}.pkl"
                if not os.path.exists(filename):
                    break
                index += 1
            joblib.dump(model, filename)
        return model

    def train_support_vector_classifier(self, save_model=False):
        if not self.is_classification_target():
            raise ValueError("Cannot train Support Vector Classifier model with regression target.")
        model = SVC()
        model.fit(self.X_train, self.y_train)

        if save_model:
            index = 0
            while True:
                filename = f"{'svc_cl'}_{index}.pkl"
                if not os.path.exists(filename):
                    break
                index += 1
            joblib.dump(model, filename)
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

    def rg0(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, inputer=False):

        try:
            model_trainer.preprocess_data_lin_reg(pca, scaling, min_max, z_score, inputer)
            
            if load_model:
                linear_regression_model = model_trainer.load_model(model_name)
            else:
                linear_regression_model = model_trainer.train_linear_regression(save_model)

            y_pred_lr, mae_lr = model_trainer.evaluate_regression_model(linear_regression_model)
            print("Linear Regression - Mean Absolute Error:", mae_lr)
            print("Linear Regression - Predicted Values:", y_pred_lr)
        except ValueError as e:
            print("Warning: ", e)
            
    def rg1(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, inputer=False):

        try:
            model_trainer.preprocess_data_rfr(pca, scaling, min_max, z_score, inputer)
            if load_model:
                random_forest_regression_model = model_trainer.load_model(model_name)
            else:
                random_forest_regression_model = model_trainer.train_random_forest_regression(save_model)
                
            y_pred_rf, mae_rf = model_trainer.evaluate_regression_model(random_forest_regression_model)
            print("Random Forest Regression - Mean Absolute Error:", mae_rf)
            print("Random Forest Regression - Predicted Values:", y_pred_rf)
        except ValueError as e:
            print("Warning: ", e)
            
    def rg2(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, inputer=False):

        try:
            model_trainer.preprocess_data_svr(pca, scaling, min_max, z_score, inputer)
            
            if load_model:
                support_vector_regression_model = model_trainer.load_model(model_name)
            else:
                support_vector_regression_model = model_trainer.train_support_vector_regression(save_model)
                
            y_pred_svr, mae_svr = model_trainer.evaluate_regression_model(support_vector_regression_model)
            print("Support Vector Regression - Mean Absolute Error:", mae_svr)
            print("Support Vector Regression - Predicted Values:", y_pred_svr)
        except ValueError as e:
            print("Warning: ", e)
            
    def cl0(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, one_hot=False, label=False, inputer=False):

        try:
            model_trainer.preprocess_data_log_reg(pca, scaling, min_max, z_score, inputer, one_hot, label)

            if load_model:
                logistic_regression_model = model_trainer.load_model(model_name)
            else:
                logistic_regression_model = model_trainer.train_logistic_regression(save_model)

            y_pred_logistic, accuracy_logistic = model_trainer.evaluate_classification_model(logistic_regression_model)
            print("Logistic Regression - Accuracy:", accuracy_logistic)
            print("Logistic Regression - Predicted Values:", y_pred_logistic)
        except ValueError as e:
            print("Warning: ", e)
            
    def cl1(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, one_hot=False, label=False, inputer=False):

        try:
            model_trainer.preprocess_data_rfc(pca, scaling, min_max, z_score, inputer, one_hot, label)
            
            if load_model:
                random_forest_classifier_model = model_trainer.load_model(model_name)
            else:
                random_forest_classifier_model = model_trainer.train_random_forest_classifier(save_model)
                
            y_pred_rf_class, accuracy_rf_class = model_trainer.evaluate_classification_model(
                random_forest_classifier_model)
            print("Random Forest Classifier - Accuracy:", accuracy_rf_class)
            print("Random Forest Classifier - Predicted Values:", y_pred_rf_class)
        except ValueError as e:
            print("Warning: ", e)
            
    def cl2(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, one_hot=False, label=False, inputer=False):

        try:
            model_trainer.preprocess_data_svc(pca, scaling, min_max, z_score, inputer, one_hot, label)
            
            if load_model:
                support_vector_classifier_model = model_trainer.load_model(model_name)
            else:
                support_vector_classifier_model = model_trainer.train_support_vector_classifier(save_model)
                
            y_pred_svc, accuracy_svc = model_trainer.evaluate_classification_model(support_vector_classifier_model)
            print("Support Vector Classifier - Accuracy:", accuracy_svc)
            print("Support Vector Classifier - Predicted Values:", y_pred_svc)
        except ValueError as e:
            print("Warning: ", e)

    def func_call(self, model_type=None, load_model=False, model_name=None, save_model=False, pca=False, scaling=False,
                  min_max=False, z_score=False, one_hot=False, label=False, inputer=False):

        if model_type == 0:
            model_trainer.rg0(load_model, model_name, save_model, pca, scaling, min_max, z_score, inputer)
        if model_type == 1:
            model_trainer.rg1(load_model, model_name, save_model, pca, scaling, min_max, z_score, inputer)
        if model_type == 2:
            model_trainer.rg2(load_model, model_name, save_model, pca, scaling, min_max, z_score, inputer)
        if model_type == 3:
            model_trainer.cl0(load_model, model_name, save_model, pca, scaling, min_max, z_score, one_hot, label,
                              inputer)
        if model_type == 4:
            model_trainer.cl1(load_model, model_name, save_model, pca, scaling, min_max, z_score, one_hot, label,
                              inputer)
        if model_type == 5:
            model_trainer.cl2(load_model, model_name, save_model, pca, scaling, min_max, z_score, one_hot, label,
                              inputer)


model_trainer = ModelTrainer('Heart_Disease_Class.csv', 'sex')

model_trainer.func_call(3, False, None, False, False, True, True,
                        True)

