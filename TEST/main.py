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
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import asyncio
import time

result = {}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def str_to_bool(value):
    """Converts 'True' and 'False' strings to corresponding boolean values, case-insensitively."""
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
    return value


def read_row(filename, row_num):
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == row_num:
                return int(line.strip())
    return None


def increment_row(filename, row_num):
    with open(filename, 'r') as file:
        lines = file.readlines()

    if 0 <= row_num < len(lines):
        current_value = int(lines[row_num].strip())
        lines[row_num] = f"{current_value + 1}\n"

        with open(filename, 'w') as file:
            file.writelines(lines)
    else:
        print(f"Row number {row_num} is out of bounds.")


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_column_in_csv(file_name, column_name):
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return column_name in headers


@app.route('/upload', methods=['POST'])
def upload_file():
    global globalFileName

    # Check if the file part is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty part without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        globalFileName = filename

        # Path to save the file, which is the current directory of the script
        save_path = os.path.join(os.getcwd(), filename)
        file.save(save_path)

        # Read only column names without loading entire data
        try:
            with open(save_path, 'r') as f:
                columns = pd.read_csv(f, nrows=0).columns.tolist()
        except Exception as e:
            return jsonify({'message': 'Failed to read file columns', 'details': str(e)}), 500

        return jsonify({'message': 'File uploaded successfully', 'columns': columns})
    else:
        return jsonify({'message': 'Invalid file or file type'}), 400


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
            increment_row("saved_models_names.txt", 0)
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
            increment_row("saved_models_names.txt", 1)
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
            increment_row("saved_models_names.txt", 2)
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
            increment_row("saved_models_names.txt", 3)
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
            increment_row("saved_models_names.txt", 4)
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
            increment_row("saved_models_names.txt", 5)
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
            self.preprocess_data_lin_reg(pca, scaling, min_max, z_score, inputer)

            if load_model:
                linear_regression_model = self.load_model(f"{'lin_reg'}_{read_row('saved_models_names.txt', 0)}.pkl")
            else:
                linear_regression_model = self.train_linear_regression(save_model)

            y_pred_lr, mae_lr = self.evaluate_regression_model(linear_regression_model)
            print("Linear Regression - Mean Absolute Error:", mae_lr)
            print("Linear Regression - Predicted Values:", y_pred_lr)
            temp = np.array(y_pred_lr)
            global result
            return {
                "type": "result",
                "predicted": temp.tolist(),
                "init": self.y_test.tolist(),
                "error": mae_lr,
                "error_type": "Mean Absolute Error",
                "algorithm": "regression"
            }

        except ValueError as e:
            return
            {
                "type": "error",
                "result": str(e)
            }

    def rg1(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, inputer=False):

        try:
            self.preprocess_data_rfr(pca, scaling, min_max, z_score, inputer)
            if load_model:
                random_forest_regression_model = self.load_model(
                    f"{'random_forest_reg'}_{read_row('saved_models_names.txt', 1)}.pkl")
            else:
                random_forest_regression_model = self.train_random_forest_regression(save_model)

            y_pred_rf, mae_rf = self.evaluate_regression_model(random_forest_regression_model)
            print("Random Forest Regression - Mean Absolute Error:", mae_rf)
            print("Random Forest Regression - Predicted Values:", y_pred_rf)
            temp = np.array(y_pred_rf)
            global result
            return {
                "type": "result",
                "predicted": temp.tolist(),
                "init": self.y_test.tolist(),
                "error": mae_rf,
                "error_type": "Mean Absolute Error",
                "algorithm": "regression"
            }
        except ValueError as e:
            return
            {
                "type": "error",
                "result": str(e)
            }

    def rg2(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, inputer=False):

        try:
            self.preprocess_data_svr(pca, scaling, min_max, z_score, inputer)

            if load_model:
                support_vector_regression_model = self.load_model(
                    f"{'svr_reg'}_{read_row('saved_models_names.txt', 2)}.pkl")
            else:
                support_vector_regression_model = self.train_support_vector_regression(save_model)

            y_pred_svr, mae_svr = self.evaluate_regression_model(support_vector_regression_model)
            print("Support Vector Regression - Mean Absolute Error:", mae_svr)
            print("Support Vector Regression - Predicted Values:", y_pred_svr)
            temp = np.array(y_pred_svr)
            global result
            return {
                "type": "result",
                "predicted": temp.tolist(),
                "init": self.y_test.tolist(),
                "error": mae_svr,
                "error_type": "Mean Absolute Error",
                "algorithm": "regression"
            }
        except ValueError as e:
            return
            {
                "type": "error",
                "result": str(e)
            }

    def cl0(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, one_hot=False, label=False, inputer=False):

        try:
            self.preprocess_data_log_reg(pca, scaling, min_max, z_score, inputer, one_hot, label)

            if load_model:
                logistic_regression_model = self.load_model(
                    f"{'logistic_reg_cl'}_{read_row('saved_models_names.txt', 3)}.pkl")
            else:
                logistic_regression_model = self.train_logistic_regression(save_model)

            y_pred_logistic, accuracy_logistic = self.evaluate_classification_model(logistic_regression_model)
            print("Logistic Regression - Accuracy:", accuracy_logistic)
            print("Logistic Regression - Predicted Values:", y_pred_logistic)
            temp = np.array(y_pred_logistic)
            global result
            return {
                "type": "result",
                "predicted": temp.tolist(),
                "init": self.y_test.tolist(),
                "error": accuracy_logistic,
                "error_type": "Accuracy",
                "algorithm": "classification"
            }
        except ValueError as e:
            print(e)
            return
            {
                "type": "error",
                "result": str(e)
            }

    def cl1(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, one_hot=False, label=False, inputer=False):

        try:
            self.preprocess_data_rfc(pca, scaling, min_max, z_score, inputer, one_hot, label)

            if load_model:
                random_forest_classifier_model = self.load_model(
                    f"{'random_forest_cl'}_{read_row('saved_models_names.txt', 4)}.pkl")
            else:
                random_forest_classifier_model = self.train_random_forest_classifier(save_model)

            y_pred_rf_class, accuracy_rf_class = self.evaluate_classification_model(
                random_forest_classifier_model)
            print("Random Forest Classifier - Accuracy:", accuracy_rf_class)
            print("Random Forest Classifier - Predicted Values:", y_pred_rf_class)
            temp = np.array(y_pred_rf_class)
            print("getting result")
            global result
            return {
                "type": "result",
                "predicted": temp.tolist(),
                "init": self.y_test.tolist(),
                "error": accuracy_rf_class,
                "error_type": "Accuracy",
                "algorithm": "classification"
            }
        except ValueError as e:
            return
            {
                "type": "error",
                "result": str(e)
            }

    def cl2(self, load_model=False, model_name=None, save_model=False, pca=False, scaling=False, min_max=False,
            z_score=False, one_hot=False, label=False, inputer=False):

        try:
            self.preprocess_data_svc(pca, scaling, min_max, z_score, inputer, one_hot, label)

            if load_model:
                support_vector_classifier_model = self.load_model(
                    f"{'svc_cl'}_{read_row('saved_models_names.txt', 5)}.pkl")
            else:
                support_vector_classifier_model = self.train_support_vector_classifier(save_model)

            y_pred_svc, accuracy_svc = self.evaluate_classification_model(support_vector_classifier_model)
            print("Support Vector Classifier - Accuracy:", accuracy_svc)
            print("Support Vector Classifier - Predicted Values:", y_pred_svc)
            temp = np.array(y_pred_svc)
            global result
            return {
                "type": "result",
                "predicted": temp.tolist(),
                "init": self.y_test.tolist(),
                "error": accuracy_svc,
                "error_type": "Accuracy",
                "algorithm": "classification"
            }
        except ValueError as e:
            return
            {
                "type": "error",
                "result": str(e)
            }

    def func_call(self, model_type=None, load_model=False, model_name=None, save_model=False, pca=False, scaling=False,
                  min_max=False, z_score=False, one_hot=False, label=False, inputer=False):

        if model_type == '0':
            print("here")
            return (self.rg0(str_to_bool(load_model), model_name, str_to_bool(save_model), str_to_bool(pca),
                             str_to_bool(scaling), str_to_bool(min_max), str_to_bool(z_score), str_to_bool(inputer)))
        if model_type == '1':
            return (self.rg1(str_to_bool(load_model), model_name, str_to_bool(save_model), str_to_bool(pca),
                             str_to_bool(scaling), str_to_bool(min_max), str_to_bool(z_score), str_to_bool(inputer)))
        if model_type == '2':
            return (self.rg2(str_to_bool(load_model), model_name, str_to_bool(save_model), str_to_bool(pca),
                             str_to_bool(scaling), str_to_bool(min_max), str_to_bool(z_score), str_to_bool(inputer)))
        if model_type == '3':
            return (self.cl0(str_to_bool(load_model), model_name, str_to_bool(save_model), str_to_bool(pca),
                             str_to_bool(scaling), str_to_bool(min_max), str_to_bool(z_score), str_to_bool(one_hot),
                             str_to_bool(label),
                             str_to_bool(inputer)))
        if model_type == '4':
            return (self.cl1(str_to_bool(load_model), model_name, str_to_bool(save_model), str_to_bool(pca),
                             str_to_bool(scaling), str_to_bool(min_max), str_to_bool(z_score), str_to_bool(one_hot),
                             str_to_bool(label),
                             str_to_bool(inputer)))
        if model_type == '5':
            return (self.cl2(str_to_bool(load_model), model_name, str_to_bool(save_model), str_to_bool(pca),
                             str_to_bool(scaling), str_to_bool(min_max), str_to_bool(z_score), str_to_bool(one_hot),
                             str_to_bool(label),
                             str_to_bool(inputer)))

@app.route('/process', methods=['POST'])
def handle_command():
    data = request.get_json()
    file_name = data.pop('file_name', None)
    target_column = data.pop('target_column', None)
    print(data)
    model_trainer = ModelTrainer(file_name, target_column)
    tempRes = model_trainer.func_call(**data)
    if tempRes is None:
        return jsonify({
            "type": "error",
            "result": "Not possible to train model"
        })
    return jsonify(tempRes)

if __name__ == "__main__":
    app.run(debug=True)

    # model_trainer = ModelTrainer("Heart_Disease_Class.csv", "age")
    # print(model_trainer.func_call(0))
