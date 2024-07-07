import numpy as np
import pandas as pd
from pyswarm import pso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deap import base, creator, tools, algorithms
import pickle
import xgboost as xgb
import lightgbm as lgb
from plot_img import rf_learning_curve, plot_feature_importance, plot_predictions_vs_actuals, plot_residuals


def train_Random_Forest(data, target_column=None, pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/random_forest_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        model.set_params(**loaded_params)
        model.fit(input_train, output_train)
        return model
    else:
        print("Training Random Forest model Using Bayes...")
        search_spaces = {
                'n_estimators': Integer(50, 100),
                'max_features': Categorical(['log2', 'sqrt']),
                'max_depth': Integer(5, 15),
                'min_samples_split': Integer(2, 5),
                'min_samples_leaf': Integer(1, 5)
        }
        opt = BayesSearchCV(estimator=model,
                            search_spaces=search_spaces,
                            n_iter=10,
                            cv=3,
                            scoring='neg_mean_squared_error',
                            verbose=2,
                            random_state=42,
                            n_jobs=1)
        opt.fit(input_train, output_train)
        best_model = opt.best_estimator_

        predictions = best_model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Optimized Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        model_params = best_model.get_params()
        with open('weight_ckpt/random_forest_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

        return best_model


def train_XGBoost(data, target_column=None, pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/xgboost_params.pkl', 'rb') as f:
            loaded_params = pickle.load(f)
        model = xgb.XGBRegressor(**loaded_params)
        model.fit(input_train, output_train)
        return model
    else:
        print("Training XGBoost model Using PSO...")

        def pso_objective(params):
            max_depth, learning_rate, alpha = int(params[0]), params[1], params[2]
            model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    colsample_bytree=0.4,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    alpha=alpha,
                    n_estimators=100,
                    random_state=42
            )
            model.fit(input_train, output_train)
            predictions = model.predict(input_test)
            mse = mean_squared_error(output_test, predictions)
            return mse

        lb = [1, 0.01, 1]
        ub = [5, 0.2, 50]
        best_params, _ = pso(pso_objective, lb, ub, swarmsize=20, maxiter=50, minstep=1e-8, minfunc=1e-8, debug=True)

        model_params = {
                'objective': 'reg:squarederror',
                'colsample_bytree': 0.4,
                'learning_rate': best_params[1],
                'max_depth': int(best_params[0]),
                'alpha': best_params[2],
                'n_estimators': 100,
                'random_state': 42
        }

        model = xgb.XGBRegressor(**model_params)
        model.fit(input_train, output_train)

        with open('weight_ckpt/xgboost_params.pkl', 'wb') as f:
            pickle.dump(model_params, f)

        predictions = model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Optimized Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        return model


def train_Decision_Tree(data, target_column=None, pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/decision_tree_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
        model = BaggingRegressor(base_estimator=DecisionTreeRegressor(**best_params), n_estimators=10, random_state=42)
        model.fit(input_train, output_train)
        predictions = model.predict(input_test)
        return model, output_test, predictions
    else:
        print("Training Decision Tree model Using GA...")

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attr_max_depth", np.random.randint, 1, 10)
        toolbox.register("attr_min_samples_split", np.random.randint, 2, 10)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_max_depth, toolbox.attr_min_samples_split), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evalModel(individual):
            model = DecisionTreeRegressor(max_depth=individual[0], min_samples_split=individual[1], random_state=42)
            model.fit(input_train, output_train)
            predictions = model.predict(input_test)
            mse = mean_squared_error(output_test, predictions)
            return (mse,)

        toolbox.register("evaluate", evalModel)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[1, 2], up=[10, 10], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=20)
        NGEN = 20
        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))

        best_ind = tools.selBest(population, 1)[0]
        best_params = {'max_depth': best_ind[0], 'min_samples_split': best_ind[1]}
        print("Best individual is %s, with MSE = %f" % (best_ind, best_ind.fitness.values[0]))

        model = BaggingRegressor(base_estimator=DecisionTreeRegressor(**best_params), n_estimators=10, random_state=42)
        model.fit(input_train, output_train)

        with open('weight_ckpt/decision_tree_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)

        predictions = model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Bagging Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        return model, output_test, predictions


def train_LGBM(data, target_column=None, pretrained=False):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(random_state=42)

    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/lightgbm_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        model.set_params(**loaded_params)
        model.fit(input_train, output_train)
        predictions = model.predict(input_test)
        return model, output_test, predictions
    else:
        print("Training LightGBM model Using Bayesian Optimization...")
        search_spaces = {
                'n_estimators': Integer(50, 100),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.1),
                'num_leaves': Integer(20, 50),
                'colsample_bytree': Real(0.1, 0.5)
        }

        opt = BayesSearchCV(
                estimator=model,
                search_spaces=search_spaces,
                n_iter=10,
                cv=3,
                scoring='neg_mean_squared_error',
                verbose=2,
                random_state=42,
                n_jobs=1
        )

        opt.fit(input_train, output_train)
        best_model = opt.best_estimator_

        predictions = best_model.predict(input_test)
        r2 = r2_score(output_test, predictions)
        mse = mean_squared_error(output_test, predictions)
        mae = mean_absolute_error(output_test, predictions)

        print(f"R²: {r2}")
        print(f"Optimized Mean Squared Error: {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        model_params = best_model.get_params()
        with open('weight_ckpt/lightgbm_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

        return best_model, output_test, predictions


def train_SVR(data, target_column=None, pretrained=True):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

    # 特征缩放
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    bagging_model = BaggingRegressor(base_estimator=svr_model, n_estimators=10, random_state=42)
    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/svr_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)

        if 'estimator' in loaded_params:
            loaded_params['base_estimator'] = loaded_params.pop('estimator')

        bagging_model.set_params(**loaded_params)
        bagging_model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = bagging_model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return bagging_model, y_pred, y_test
    else:
        bagging_model.fit(X_train_scaled, y_train_scaled)
        model_params = bagging_model.get_params()
        with open('weight_ckpt/svr_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

    y_pred_scaled = bagging_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R²: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    return svr_model, y_pred, y_test


def train_stacking_model(data, target_column=None, pretrained=True):
    input = data.drop(target_column, axis=1)
    output = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

    # 加载预训练好的模型
    rf_model = train_Random_Forest(data, target_column, pretrained=True)
    xgb_model = train_XGBoost(data, target_column, pretrained=True)
    dt_model = train_Decision_Tree(data, target_column, pretrained=True)[0]
    lgbm_model = train_LGBM(data, target_column, pretrained=True)[0]
    svr_model = train_SVR(data, target_column, pretrained=True)[0]

    estimators = [
            ('random_forest', rf_model),
            ('xgboost', xgb_model),
            ('decision_tree', dt_model),
            ('lightgbm', lgbm_model),
            ('svr', svr_model)
    ]

    # Stacking Regressor
    stacking_regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=3
    )
    if pretrained:
        print('=' * 25 + "Find weight file so return Model directly" + '=' * 25)
        with open('weight_ckpt/stacking_model_params.pkl', 'rb') as file:
            loaded_params = pickle.load(file)
        stacking_regressor.set_params(**loaded_params)
        stacking_regressor.fit(X_train, y_train)
        y_pred = stacking_regressor.predict(X_test)
        return stacking_regressor, y_pred, y_test
    else:
        # 训练Stacking模型（这里仅用于重新调整final_estimator）
        stacking_regressor.fit(X_train, y_train)
        model_params = stacking_regressor.get_params()
        with open('weight_ckpt/stacking_model_params.pkl', 'wb') as file:
            pickle.dump(model_params, file)

    y_pred = stacking_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Stacking Model R²: {r2}")
    print(f"Stacking Model Mean Squared Error: {mse}")
    print(f"Stacking Model Mean Absolute Error (MAE): {mae}")

    return stacking_regressor, y_pred, y_test


def train_model(csv_path, method='Random_Forest', target_column=None, pretrained=False):
    print('=' * 25 + "Training Model Start..." + '=' * 25)
    df = pd.read_csv(csv_path)
    selected_columns = ["地形排水", "基础设施恶化", "季风强度", "淤积", "滑坡", "人口得分", "气候变化", "无效防灾",
                        "农业实践", "流域", "政策因素", "规划不足", "洪水概率"]
    df = df[selected_columns]
    if method == 'Random_Forest':
        model = train_Random_Forest(df, target_column=target_column, pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        rf_learning_curve(df, target_column=target_column, filename='output_img/RF_learning_curve.png')

    elif method == 'Decision_Tree':
        model, output_test, pred = train_Decision_Tree(df, target_column=target_column, pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_predictions_vs_actuals(pred, output_test, filename='output_img/Decision_Tree_predictions_vs_actuals.png')

    elif method == 'XGBoost':
        model = train_XGBoost(df, target_column=target_column, pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_feature_importance(model, filename='output_img/XGBoost_feature_importance.png')

    elif method == 'LGBM':
        model, output_test, pred = train_LGBM(df, target_column=target_column, pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_predictions_vs_actuals(pred, output_test, filename='output_img/LGBM_predictions_vs_actuals.png')

    elif method == 'SVR':
        model, y_pred, y_test = train_SVR(df, target_column=target_column, pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_residuals(y_pred, y_test, filename='output_img/SVR_residuals.png')

    elif method == 'stacking_model':
        model, y_pred, y_test = train_stacking_model(df, target_column=target_column, pretrained=pretrained)
        print('=' * 25 + "Plotting Start..." + '=' * 25)
        plot_predictions_vs_actuals(y_pred, y_test, filename='output_img/Stacking_predictions_vs_actuals.png')

    return model


def predict(pred_csv_path=None, model=None):
    print('=' * 25 + "Energy Consumption Predict Start..." + '=' * 25)
    X = pd.read_csv(pred_csv_path,encoding='gbk')
    selected_columns = ["地形排水", "基础设施恶化", "季风强度", "淤积", "滑坡", "人口得分", "气候变化", "无效防灾",
                        "农业实践", "流域", "政策因素", "规划不足"]
    X = X[selected_columns]
    pred_output = model.predict(X)
    return pred_output


if __name__ == '__main__':
    train_dataset_path = '../DATA_PROCESS/processed_train_data.csv'
    model = train_model(train_dataset_path, method='SVR', target_column="洪水概率", pretrained=False)

    pred_path = '../DATA_PROCESS/processed_test_data.csv'
    pred_output = predict(pred_csv_path=pred_path, model=model)
    print(pred_output)

# Random_Forest
# R²: 0.41219147102043086
# Optimized Mean Squared Error: 0.0014037171463660435
# Mean Absolute Error (MAE): 0.030518168593419076


# Decision_Tree
# Best individual is [10, 8], with MSE = 0.001785
