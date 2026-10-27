import json
from pathlib import Path

path = Path('hw01.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

nb['cells'][12]['source'] = '''## 4. 模型实现与参数调优

### 4.1 特征组合

两种特征选择方式得到的三个特征相同，因此后续建模固定使用这一组三特征：`RM`、`LSTAT`、`PTRATIO`。

- 相关性法得到：`RM`、`LSTAT`、`PTRATIO`；
- 训练集交叉验证法得到：`RM`、`PTRATIO`、`LSTAT`；
- 两者只是顺序不同，实际特征集合完全一致。

因此，后续不再比较不同特征组合，只比较不同线性模型及其正则化参数。
'''.splitlines(keepends=True)

# Insert selected_features code by replacing previous candidate dict cell if it exists; current cell 13 is markdown, so add to start of cell 14 code.
nb['cells'][13]['source'] = '''### 4.2 参数调优方法

固定三特征后，使用训练集 5 折交叉验证比较以下模型：普通线性回归、Ridge、Lasso 和 ElasticNet。调参只使用前 400 个训练样本，最后 100 个测试样本不参与模型选择，只用于最终评价。
'''.splitlines(keepends=True)

nb['cells'][14]['source'] = '''selected_features = ['RM', 'LSTAT', 'PTRATIO']

print(f'最终固定使用的三个特征: {selected_features}')
print(f'相关性法特征: {corr_top_features}')
print(f'交叉验证法特征: {cv_top_features}')
print(f'两种方法得到的特征集合是否相同: {set(corr_top_features) == set(cv_top_features) == set(selected_features)}')

param_grid = [
    {'model': [LinearRegression()]},
    {'model': [Ridge()], 'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'model': [Lasso(max_iter=20000, random_state=RANDOM_STATE)], 'model__alpha': [0.001, 0.01, 0.1, 1]},
    {'model': [ElasticNet(max_iter=20000, random_state=RANDOM_STATE)],
     'model__alpha': [0.001, 0.01, 0.1, 1],
     'model__l1_ratio': [0.2, 0.5, 0.8]},
]

def model_label(best_estimator):
    model = best_estimator.named_steps['model']
    if isinstance(model, LinearRegression):
        return 'LinearRegression'
    if isinstance(model, Ridge):
        return f'Ridge(alpha={model.alpha})'
    if isinstance(model, Lasso):
        return f'Lasso(alpha={model.alpha})'
    if isinstance(model, ElasticNet):
        return f'ElasticNet(alpha={model.alpha}, l1_ratio={model.l1_ratio})'
    return type(model).__name__

base_pipe = make_pipeline(Ridge())
grid = GridSearchCV(
    estimator=base_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    n_jobs=None,
)
grid.fit(X_train_all[selected_features], y_train)

best_model = grid.best_estimator_
cv_results_df = pd.DataFrame(grid.cv_results_).sort_values('rank_test_score').reset_index(drop=True)

model_comparison_df = cv_results_df[[
    'rank_test_score', 'mean_test_score', 'std_test_score',
    'param_model', 'param_model__alpha', 'param_model__l1_ratio'
]].head(12).copy()
model_comparison_df['cv_RMSE_mean'] = -model_comparison_df['mean_test_score']
model_comparison_df = model_comparison_df.drop(columns=['mean_test_score'])

display(model_comparison_df)
print(f'训练集交叉验证选择的最优模型: {model_label(best_model)}')
print(f'最优交叉验证 RMSE: {-grid.best_score_:.4f}')
'''.splitlines(keepends=True)

nb['cells'][15]['source'] = '''# 使用最优模型分别在训练集和测试集上预测，并汇总评价指标。
y_train_pred = best_model.predict(X_train_all[selected_features])
y_test_pred = best_model.predict(X_test_all[selected_features])

metrics_df = pd.DataFrame([
    {
        'dataset': 'train_first_400',
        'RMSE': rmse_score(y_train, y_train_pred),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'R2': r2_score(y_train, y_train_pred),
    },
    {
        'dataset': 'test_last_100',
        'RMSE': rmse_score(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred),
    },
])

display(metrics_df)
'''.splitlines(keepends=True)

nb['cells'][17]['source'] = '''### 5.2 模型性能比较

由于两种特征选择方式得到的三特征相同，本实验后续固定使用 `RM`、`LSTAT`、`PTRATIO`。模型性能比较的重点转为：在同一组三特征下，普通线性回归、Ridge、Lasso 和 ElasticNet 哪个模型及参数组合更优。

从交叉验证结果看，正则化模型能够在普通线性回归基础上提升稳定性。最终模型由训练集 5 折交叉验证确定，测试集只用于最终泛化性能评价。
'''.splitlines(keepends=True)

nb['cells'][18]['source'] = '''# 固定使用训练集交叉验证得到的最优模型。
final_model = best_model
final_features = selected_features
final_pred = y_test_pred

final_result_df = pd.DataFrame({
    'sample_index': X_test_all.index,
    'actual_MEDV': y_test.values,
    'predicted_MEDV': final_pred,
    'residual': y_test.values - final_pred,
})

test_metrics = metrics_df[metrics_df['dataset'] == 'test_last_100'].iloc[0]

print('最终展示模型')
print(f'三个特征: {final_features}')
print(f'模型: {model_label(final_model)}')
print(f"训练集交叉验证 RMSE: {-grid.best_score_:.4f}")
print(f"测试集 RMSE: {test_metrics['RMSE']:.4f}")
print(f"测试集 MAE : {test_metrics['MAE']:.4f}")
print(f"测试集 R²  : {test_metrics['R2']:.4f}")

display(final_result_df.head(10))
'''.splitlines(keepends=True)

nb['cells'][21]['source'] = '''plt.figure(figsize=(10, 5))
plot_df = cv_results_df.head(10).copy()
plot_df['model_setting'] = plot_df.apply(
    lambda row: f"{type(row['param_model']).__name__}, alpha={row.get('param_model__alpha', '')}, l1={row.get('param_model__l1_ratio', '')}",
    axis=1
)
plot_df['cv_RMSE'] = -plot_df['mean_test_score']
sns.barplot(data=plot_df, x='cv_RMSE', y='model_setting', color='#6baed6')
plt.title('Top 10 Model Settings by 5-fold CV RMSE')
plt.xlabel('CV RMSE (lower is better)')
plt.ylabel('Model Setting')
plt.tight_layout()
plt.show()
'''.splitlines(keepends=True)

nb['cells'][25]['source'] = '''## 6. 结论与展望

### 6.1 实验结论

1. 相关性法和训练集交叉验证法最终得到相同的三个特征：`RM`、`LSTAT`、`PTRATIO`。
2. 因此，本实验将特征选择固定为这三个变量，后续重点比较不同线性回归模型和正则化参数。
3. 通过训练集 5 折交叉验证选择最终模型，测试集只用于最后评价，避免用测试集反向调参。
4. Ridge、Lasso、ElasticNet 都属于正则化线性模型，可以在普通线性回归基础上提高模型稳定性。
5. 最后 100 个样本的测试指标反映了模型在题目指定测试集上的泛化效果。

### 6.2 实验局限性与改进方向

1. 数据集按行切分而非随机切分，训练集和测试集分布可能存在差异，测试结果会受到样本排序影响。
2. 本实验只允许使用 3 个特征，模型表达能力有限，可能无法充分刻画房价与变量之间的复杂关系。
3. 当前只比较线性模型，后续可以尝试加入多项式特征、交叉项或非线性模型，但需要注意题目对特征数量和模型形式的限制。
4. 如果用于真实预测任务，应增加独立验证集或重复交叉验证，以获得更稳定的模型选择结论。
'''.splitlines(keepends=True)

for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('patched fixed selected_features flow')
