import json
from pathlib import Path

path = Path('hw01.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

nb['cells'][10]['source'] = '''### 3.3 三特征组合尝试

为了回答“哪三个特征效果比较好”，这里保留两种训练阶段可以合理使用的尝试：

1. **相关性法**：选择训练集中与 `MEDV` 绝对相关性最高的 3 个特征；
2. **训练集交叉验证法**：在前 400 个训练样本内部，对所有三特征组合做 5 折交叉验证，选择平均 RMSE 最低的组合。

这两种方法都只使用训练集信息，不使用最后 100 个测试样本参与特征选择，因此更符合规范的建模流程。
'''.splitlines(keepends=True)

nb['cells'][12]['source'] = '''## 4. 模型实现与参数调优

### 4.1 候选特征组合

综合上面的尝试，本实验只保留两种候选特征组合进行模型比较：

- `corr_top`：训练集中与 `MEDV` 相关性最高的三特征，用作直观基线方案；
- `cv_top`：训练集 5 折交叉验证 RMSE 最低的三特征，作为最终推荐方案。

这样做的原则是：特征选择和参数调优只使用训练集，测试集只用于最终评价。
'''.splitlines(keepends=True)

nb['cells'][14]['source'] = '''### 4.2 参数调优方法

对每组候选特征，使用训练集 5 折交叉验证调优模型和正则化参数。调参只使用前 400 个训练样本，最后 100 个测试样本不参与模型选择，只用于最终评价。
'''.splitlines(keepends=True)

nb['cells'][18]['source'] = '''### 5.2 模型性能比较

从模型比较表可以看到：

- `corr_top` 是根据单个特征与房价的相关性筛选出来的，优点是简单直观；
- `cv_top` 是在训练集内部枚举所有三特征组合后，根据 5 折交叉验证 RMSE 选出的，更符合模型选择流程；
- 测试集评价结果用于分析泛化能力，但不反过来决定特征组合或参数。

因此，后续将 `cv_top` 作为最终模型方案进行系数解释和可视化分析。
'''.splitlines(keepends=True)

nb['cells'][19]['source'] = '''# 按规范的训练流程，固定采用训练集交叉验证得到的 cv_top 方案作为最终模型。
final_set_name = 'cv_top'
final_features = candidate_feature_sets[final_set_name]
final_model = best_estimators[final_set_name]

final_row = comparison_df[comparison_df['feature_set'] == final_set_name].iloc[0]
final_pred = final_model.predict(X_test_all[final_features])
final_result_df = pd.DataFrame({
    'sample_index': X_test_all.index,
    'actual_MEDV': y_test.values,
    'predicted_MEDV': final_pred,
    'residual': y_test.values - final_pred,
})

print('最终展示模型')
print(f'特征组名称: {final_set_name}')
print(f'三个特征: {final_features}')
print(f'模型: {model_label(final_model)}')
print(f"训练集交叉验证 RMSE: {final_row['cv_RMSE']:.4f}")
print(f"测试集 RMSE: {final_row['test_RMSE']:.4f}")
print(f"测试集 MAE : {final_row['test_MAE']:.4f}")
print(f"测试集 R²  : {final_row['test_R2']:.4f}")

display(final_result_df.head(10))
'''.splitlines(keepends=True)

nb['cells'][26]['source'] = '''## 6. 结论与展望

### 6.1 实验结论

1. 在“只能使用三个特征”的限制下，特征选择对模型性能影响明显。
2. 相关性法简单直观，适合作为基线特征选择方法。
3. 训练集交叉验证法更系统，能够在不使用测试集信息的前提下比较所有三特征组合，因此本实验将 `cv_top` 作为最终方案。
4. Ridge、Lasso、ElasticNet 都属于正则化线性模型，能够在普通线性回归基础上提高模型稳定性；本实验通过训练集交叉验证选择具体模型和参数。
5. 最后 100 个样本只用于最终测试评价，能够更客观地反映模型泛化能力。

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
print('fixed utf8 cells')
