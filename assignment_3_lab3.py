import numpy as np # type: ignore
from linear_regression import LinearRegression
import pandas as pd # type: ignore
def organize_data(data: pd.DataFrame):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = np.array(X)
    y = np.array(y)

    return (X, y)

def convert_month2int(month: str) -> int:
    month_dict = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12,
    }
    return month_dict[month]

def convert_day2int(day: str) -> int:
    day_dict = {
        'sun': 1,
        'mon': 2,
        'tue': 3,
        'wed': 4,
        'thu': 5,
        'fri': 6,
        'sat': 7,
    }
    return day_dict[day]

def preprocess_data( df: pd.DataFrame, drop_dmc=False):
    df['month'] = df['month'].map(convert_month2int)
    df['day'] = df['day'].map(convert_day2int)

    continues_variables = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'wind','RH', 'rain']
    for variable in continues_variables:
        df[variable] = (df[variable] - df[variable].mean()) / df[variable].std()

    if drop_dmc:
        df = df.drop(columns=['DMC'])
    return df

if __name__ == "__main__":
    data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Tài liệu\forestfires.csv")
    data = data.sample(frac=1, ignore_index=True)

    data_goc = preprocess_data(data.copy())
    train_data = data_goc.iloc[:414]
    test_data = data_goc.iloc[414:]

    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_predicted = lr_model.predict(X_test)
    rmse_goc = lr_model.evaluate(y_test, y_predicted)

    data_moi = preprocess_data(data.copy(), drop_dmc=True)
    train_data = data_moi.iloc[:414]
    test_data = data_moi.iloc[414:]

    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_predicted = lr_model.predict(X_test)
    rmse_moi = lr_model.evaluate(y_test, y_predicted)

    print("Rmse trước khi bỏ DMC: ", rmse_goc)
    print("Rmse sau khi bỏ DMC: ", rmse_moi)