import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from deap import algorithms, base, creator, tools
import random
from multiprocessing import Pool
from scipy import stats
import seaborn as sns
import statsmodels.api as sm



# 定义文件路径
TEST_DATA_PATH = 'your/path/to/the/file'
TRAIN_DATA_PATH = 'your/path/to/the/file'

# 加载数据
test_data = pd.read_excel(TEST_DATA_PATH)
train_data = pd.read_excel(TRAIN_DATA_PATH)

X_train = train_data.iloc[:, 1:-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, 1:-1].values
y_test = test_data.iloc[:, -1].values

# 设置遗传算法参数
POP_SIZE = 200
NGEN = 5000
RANDOM_SEED = 12345
MUTPB = 0.2
CXPB = 0.5
NUM_FEATURES = 6

# 设置相关性阈值
CORRELATION_THRESHOLD = 0.7

def get_correlation(selected_features):
    X_train_selected = X_train[:, selected_features]
    corr_matrix = np.corrcoef(X_train_selected, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)
    abs_corr_matrix = np.abs(corr_matrix)
    return abs_corr_matrix



# 设置随机种子
random.seed(RANDOM_SEED)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)



# 注册遗传算法操作
toolbox = base.Toolbox()

def generate_individual():
    individual = [0] * X_train.shape[1]
    selected_indices = random.sample(range(X_train.shape[1]), NUM_FEATURES)
    for index in selected_indices:
        individual[index] = 1
    return creator.Individual(individual)

toolbox.register("individual", generate_individual)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selBest)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

def evaluate(individual):
    selected_features = [index for index, value in enumerate(individual) if value > 0.5]
    if len(selected_features) != NUM_FEATURES:
        return -1e9,

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    model = LinearRegression()
    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)
    r2 = r2_score(y_test, y_pred)




toolbox.register("evaluate", evaluate)



def evaluate_individual(individual):
    return evaluate(individual)

def eval_population(individuals):
    return list(pool.map(evaluate, individuals))


deap_stats = tools.Statistics(lambda ind: ind.fitness.values)
deap_stats.register("avg", np.mean)
deap_stats.register("min", np.min)
deap_stats.register("max", np.max)



pool = None

def main():
    global TEST_DATA_PATH, TRAIN_DATA_PATH, pool


    pool = Pool()
    toolbox.register("evaluate", evaluate_individual)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=deap_stats, halloffame=hof, verbose=True)
    best_individual = hof[0]
    best_features = [index for index, value in enumerate(best_individual) if value > 0.5]
    print("Best features:", best_features)

    X_train_selected = X_train[:, best_features]
    X_test_selected = X_test[:, best_features]

    model = LinearRegression()
    model.fit(X_train_selected, y_train)

    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Train R^2:", train_r2)
    print("Test R^2:", test_r2)

    # R2-adj
    t_stat = model.coef_ / np.sqrt(np.diag(np.linalg.inv(np.dot(X_train_selected.T, X_train_selected))))
    r2_adj = 1 - (1 - train_r2) * (y_train.shape[0] - 1) / (y_train.shape[0] - X_train_selected.shape[1] - 1)
    print("R2-adj:", r2_adj)

    # 输出方程
    equation = "y = "
    for coef, feature_index in zip(model.coef_, best_features):
        equation += f"{coef:6f} * {train_data.columns[feature_index + 1]} + "
    equation += f"{model.intercept_:.6f}"

    print("Equation:", equation)

    # 使用留一法计算Q2
    loo = LeaveOneOut()
    q2_numerator = 0
    q2_denominator = 0
    for train_index, test_index in loo.split(X_train_selected):
        X_train_loo, X_test_loo = X_train_selected[train_index], X_train_selected[test_index]
        y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]

        model_loo = LinearRegression()
        model_loo.fit(X_train_loo, y_train_loo)
        y_test_loo_pred = model_loo.predict(X_test_loo)

        q2_numerator += (y_test_loo - y_test_loo_pred) ** 2
        q2_denominator += (y_test_loo - np.mean(y_train_loo)) ** 2

    q2 = 1 - q2_numerator / q2_denominator
    print("Q^2:", q2)

    # A k-fold cross-validation
    k = 7
    kf = KFold(n_splits=k, random_state=RANDOM_SEED, shuffle=True)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='r2')
    print(f"{k}-fold cross-validation scores:", cv_scores)
    print("Average k-fold R^2:", cv_scores.mean())

    # 计算R²(pred)、RMS residual error、Friedman和L.O.F.S.O.R.F值
    PRESS = sum((y_train - y_train_pred) ** 2)
    SSY = sum((y_train - y_train.mean()) ** 2)
    r2_pred = 1 - PRESS / SSY
    rms_residual_error = np.sqrt(mean_squared_error(y_train, y_train_pred))

    f_stat = (train_r2 * (y_train.shape[0] - X_train_selected.shape[1] - 1)) / (1 - train_r2)
    p_value = 1 - stats.f.cdf(f_stat, X_train_selected.shape[1], y_train.shape[0] - X_train_selected.shape[1] - 1)
    friedman = (SSY - PRESS) / PRESS

    lof_numerator = 0
    lof_denominator = 0
    for train_index, test_index in loo.split(X_train_selected):
        X_train_loo, X_test_loo = X_train_selected[train_index], X_train_selected[test_index]
        y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]

        model_loo = LinearRegression()
        model_loo.fit(X_train_loo, y_train_loo)
        y_test_loo_pred = model_loo.predict(X_test_loo)

        lof_numerator += (y_test_loo - y_test_loo_pred) ** 2
        lof_denominator += (y_test_loo - np.mean(y_train_loo)) ** 2

    lof_sorf_value = 1 - lof_numerator / lof_denominator

    print("R²(pred):", r2_pred)
    print("RMS residual error:", rms_residual_error)
    print("Friedman:", friedman)
    print("L.O.F.S.O.R.F value:", lof_sorf_value)
    # 输出最优个体的P值
    best_individual = hof[0]
    selected_features = [index for index, value in enumerate(best_individual) if value > 0.5]
    X_train_selected = X_train[:, selected_features]
    p_values_best = []
    for i in range(X_train_selected.shape[1]):
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_train_selected[:, i], y_train)
        p_values_best.append(p_value)
    print("Best individual P-values:", p_values_best)


    # 创建一个新的DataFrame，包含分子名称、原始的PIC50值和通过方程计算的PIC50值
    results = pd.DataFrame({'Molecule': test_data.iloc[:, 0],
                            'Original_PIC50': y_test.flatten(),
                            'Predicted_PIC50': y_test_pred.flatten()})

    # 构建一个与输入文件相同的路径来保存结果文件
    results_dir = os.path.dirname(TEST_DATA_PATH)
    results_file = os.path.join(results_dir, 'test_results.xlsx')

    # 将结果保存到一个xlsx文件中
    results.to_excel(results_file, index=False)
    # 计算训练集的预测值
    y_train_pred = model.predict(X_train_selected)

    # 创建一个新的DataFrame，包含训练集分子名称、原始的PIC50值和通过方程计算的PIC50值
    train_results = pd.DataFrame({'Molecule': train_data.iloc[:, 0],
                                  'Original_PIC50': y_train.flatten(),
                                  'Predicted_PIC50': y_train_pred.flatten()})

    # 构建一个与输入文件相同的路径来保存训练集结果文件
    train_results_dir = os.path.dirname(TRAIN_DATA_PATH)
    train_results_file = os.path.join(train_results_dir, 'train_results.xlsx')

    # 将训练集结果保存到一个xlsx文件中
    train_results.to_excel(train_results_file, index=False)
    # 提取最优个体
    best_individual = hof[0]

    # 提取最优个体对应的特征索引
    best_features_indices = [index for index, value in enumerate(best_individual) if value > 0.5]

    # 根据索引提取特征名称
    feature_names = train_data.columns[1:-1]  # 提取所有特征名称
    selected_feature_names = feature_names[best_features_indices]  # 提取最优个体的特征名称

    # 计算相关性矩阵
    correlation_matrix = get_correlation(best_features_indices)

    # 将相关性矩阵转换为DataFrame
    correlation_df = pd.DataFrame(correlation_matrix, columns=selected_feature_names, index=selected_feature_names)

    # 保存相关性矩阵到Excel文件
    correlation_df.to_excel(os.path.join(os.path.dirname(TRAIN_DATA_PATH),"correlation_matrix.xlsx"),index=False)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()