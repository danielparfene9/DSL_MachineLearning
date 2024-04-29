import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SalesPredictor:
    def __init__(self, filename):
        self.filename = filename

    def preprocess_data(self, sales_data):
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        sales_data['date'] = sales_data['date'].dt.to_period('M')

        monthly_sales = sales_data.groupby('date').sum().reset_index()
        monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
        monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
        monthly_sales = monthly_sales.dropna()

        return monthly_sales

    def create_supervised_dataset(self, monthly_sales_data, lag=12):
        supervised_data = monthly_sales_data.drop(['date', 'sales'], axis=1)

        for i in range(1, lag + 1):
            col_name = 'month_' + str(i)
            supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
        supervised_data = supervised_data.dropna().reset_index(drop=True)

        return supervised_data


    def train_model(self, train_data, scaler):
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        X_train, y_train = scaled_train_data[:, 1:], scaled_train_data[:, 0:1]
        y_train = y_train.ravel()

        linreg_model = LinearRegression()
        linreg_model.fit(X_train, y_train)

        return linreg_model

    def predict_sales(self, model, test_data, scaler, actual_sales):
        scaled_test_data = scaler.transform(test_data)
        X_test, y_test = scaled_test_data[:, 1:], scaled_test_data[:, 0:1]
        y_test = y_test.ravel()

        linreg_pred = model.predict(X_test)

        linreg_pred_test_set = np.concatenate([linreg_pred.reshape(-1, 1), X_test], axis=1)
        linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

        result_list = [linreg_pred_test_set[index][0] + actual_sales[index] for index in
                       range(len(linreg_pred_test_set))]
        return result_list

    def evaluate_model(self, predictions, actual):
        rmse = np.sqrt(mean_squared_error(predictions, actual))
        mae = mean_absolute_error(predictions, actual)
        r2 = r2_score(predictions, actual)
        return rmse, mae, r2

    def run(self):
        sales = pd.read_csv(self.filename)
        sales = sales.drop(['store', 'item'], axis=1)

        monthly_sales = self.preprocess_data(sales)
        supervised_data = self.create_supervised_dataset(monthly_sales)

        train_data = supervised_data[:-12]
        test_data = supervised_data[-12:]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        linreg_model = self.train_model(train_data, scaler)

        sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
        predict_df = pd.DataFrame(sales_dates)
        act_sales = monthly_sales['sales'][-13:].to_list()

        predictions = self.predict_sales(linreg_model, test_data, scaler, act_sales)

        predict_df['linreg_pred'] = predictions

        linreg_rmse, linreg_mae, linreg_r2 = self.evaluate_model(predictions, monthly_sales['sales'][-12:])
        print('Linear Regression RMSE:', linreg_rmse)
        print('Linear Regression MAE:', linreg_mae)
        print('Linear Regression R2 Score:', linreg_r2)

        predict_df_float = predict_df.applymap(lambda x: float("{:.2f}".format(x)) if isinstance(x, float) else x)
        print(predict_df_float.to_string(index=False))


if __name__ == "__main__":
    predictor = SalesPredictor('train.csv')
    predictor.run()
