import pandas as pd
import functions as fs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

train_id = train_set['Id']
test_id = test_set['Id']
train_set.drop('Id', axis=1, inplace=True)
test_set.drop('Id', axis=1, inplace=True)
# Dealing with outliers (Partial Sales that likely don’t represent actual market values)
plt.scatter(train_set['GrLivArea'], train_set['SalePrice'], s=15)
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
train_set = train_set.drop(train_set[(train_set['GrLivArea'] > 4000) & (train_set['SalePrice'] < 300000)].index)

train_set['SalePrice'] = fs.np.log1p(train_set['SalePrice'])
rows_num = train_set.shape[0]
train_set_y = train_set.pop('SalePrice')

all_data = pd.concat([train_set, test_set], ignore_index=True)
all_data.info()

# Converting some numerical variables that are really categorical type
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

missing_values = (all_data.isnull().sum() / len(all_data)) * 100
missing_values = missing_values.drop(missing_values[missing_values == 0].index).sort_values(ascending=False)

sns.barplot(x=missing_values.index, y=missing_values)
plt.xticks(rotation='90')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
all_data = all_data.drop([
    'PoolQC',
    'MiscFeature',
    'Alley', 'Fence',
    'MoSold',
    'GarageYrBlt',
    'BsmtFinSF1',
    'BsmtFinSF2'],
    axis=1
)

categorical_col = [
    'PoolQC',
    'MiscFeature',
    'Alley',
    'Fence',
    'FireplaceQu',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'GarageType',
    'BsmtExposure',
    'BsmtQual',
    'BsmtCond',
    'BsmtFinType1',
    'BsmtFinType2',
    'MasVnrType',
    'MSZoning',
    'Functional',
    'Utilities',
    'Exterior1st',
    'Exterior2nd',
    'SaleType',
    'KitchenQual',
    'Electrical'
]

numerical_col = [
    'GarageYrBlt',
    'MasVnrArea',
    'BsmtHalfBath',
    'BsmtFullBath',
    'GarageArea',
    'GarageCars',
    'TotalBsmtSF',
    'BsmtUnfSF',
    'BsmtFinSF1',
    'BsmtFinSF2'
]

for column in categorical_col[:15]:
    all_data[column] = all_data[column].fillna('None')

for column in categorical_col[15:]:
    all_data[column] = all_data[column].fillna(all_data[column].mode()[0])

for column in numerical_col:
    all_data[column] = all_data[column].fillna(0)

all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())
all_data.info()

all_data['YrsSinceBuilt'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['YrsSinceRemod'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']

all_data = all_data.drop([
    'YrSold',
    'MoSold',
    'YearBuilt',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'HalfBath',
    'BsmtFullBath',
    'BsmtHalfBath',
    'OpenPorchSF',
    '3SsnPorch',
    'EnclosedPorch',
    'ScreenPorch',
    'WoodDeckSF'],
    axis=1
)

features_to_drop = []
for feature in all_data.columns:
    max_value = all_data[feature].value_counts().max()
    if max_value / all_data.shape[0] > 0.95:
        features_to_drop.append(feature)

print(f'{features_to_drop}')
all_data = all_data.drop(features_to_drop, axis=1)

num_features = all_data.columns[all_data.dtypes != 'object']
skew_values = all_data[num_features].apply(lambda x: skew(x))
high_skew = skew_values[skew_values > 0.5]
skew_indices = high_skew.index
print(skew_indices)
for index in skew_indices:
    all_data[index] = fs.np.log1p(all_data[index])

# Encoding
all_data = pd.get_dummies(all_data, drop_first=True)
dummies_to_drop = []

for feature in all_data.columns:
    if 0 in all_data[feature].value_counts().index:
        zero_values = all_data[feature].value_counts()[0]
        if zero_values / all_data.shape[0] > 0.95:
            dummies_to_drop.append(feature)

all_data = all_data.drop(dummies_to_drop, axis=1)

train = all_data[:rows_num]
test = all_data[rows_num:]
X_train, X_test, y_train, y_test = train_test_split(train, train_set_y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_LinR = regressor.predict(X_test)
print("------LinearRegression------")
fs.get_model_score(regressor, y_test, y_pred_LinR, X_train, y_train)

regressor = RandomForestRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_pred_RandF = regressor.predict(X_test)
print("------RandomForest------")
fs.get_model_score(regressor, y_test, y_pred_RandF, X_train, y_train)

regressor = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1, seed=27)
regressor = regressor.fit(X_train, y_train)
xg_pred = regressor.predict(X_test)
print("------XGB------")
fs.get_model_score(regressor, y_test, xg_pred, X_train, y_train)

subm_regressor = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1, seed=27)
subm_regressor = subm_regressor.fit(train, train_set_y)
subm_y_pred = subm_regressor.predict(test)
fs.get_model_score(subm_regressor, train_set_y, subm_regressor.predict(train), train, train_set_y)

for i in range(len(subm_y_pred)):
    subm_y_pred[i] = fs.np.expm1(subm_y_pred[i])

for i in range(len(y_test)):
    y_test.iloc[i] = np.expm1(y_test.iloc[i])

submission = pd.DataFrame({"Id": test_id, "SalePrice": subm_y_pred})
submission.to_csv('Submission.csv', index=False)
