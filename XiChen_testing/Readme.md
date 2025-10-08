# 医院30天再入院预测 - Transformer模型

## 项目概况

**课程**: 2025秋季 CS526 小组项目  
**数据来源**: MIMIC临床数据库  
**任务**: 二分类（预测患者是否在30天内再入院）  
**模型**: Transformer Encoder

---

## 一、数据处理流程

### 1.1 原始数据概况

```
数据文件: readmission_features_30d_v1.csv
总记录数: 546,028 条
原始特征: 51 个
再入院率: 20.33%
```

### 1.2 数据清洗

**步骤1: 处理缺失值**
- 删除缺失率 > 30% 的列
- 保留列数: 46 个
- 删除包含任何缺失值的行
- 清洗后记录: 327,118 条（保留率 59.9%）

**缺失值处理说明**:
```python
missing_threshold = 0.30
columns_to_keep = df.columns[df.isnull().mean() < missing_threshold].tolist()
df_clean = df[columns_to_keep].copy()
df_clean = df_clean.dropna()  # 删除有缺失的行
```

### 1.3 特征工程

**删除的字段** (非预测特征):
- `subject_id`, `hadm_id` - 患者/入院ID
- `admittime`, `dischtime` - 时间戳
- `readmit_label` - 目标变量
- `index` - 索引列

**特征分类**:

| 特征类型 | 数量 | 说明 |
|---------|------|------|
| 数值特征 | 33个 | 年龄、住院时长、诊断数量等 |
| 分类特征 | 8个 | 性别、婚姻状况、保险类型等 |

**分类特征清单**:
1. `last_service` - 最后服务类型
2. `gender` - 性别
3. `language` - 语言
4. `marital_status` - 婚姻状况
5. `insurance` - 保险类型
6. `admission_type` - 入院类型
7. `admission_location` - 入院来源
8. `discharge_location` - 出院去向

**编码方式**:
- 所有分类特征使用 LabelEncoder 进行编码
- 编码后的特征命名为 `原特征名_encoded`
- 最终特征数: 41个 (33数值 + 8分类)

### 1.4 数据集划分

```python
训练集: 261,694 条 (80%)  正类比例: 22.61%
测试集: 65,424 条 (20%)   正类比例: 22.61%
```

**类别分布**:
- 负类（无再入院）: 77.39%
- 正类（再入院）: 22.61%
- **类别不平衡比**: 约 1:3.4

**数据标准化**:
```python
StandardScaler 标准化
训练集范围: [-12.79, 510.56]
```

---

## 二、模型架构

### 2.1 Transformer配置

```python
输入维度: 41
模型维度 (d_model): 256
注意力头数: 8
编码器层数: 3
Dropout: 0.2
总参数量: 2,422,978
```

### 2.2 网络结构

```
输入层 (41) 
    ↓
输入投影层 (Linear + LayerNorm + ReLU + Dropout)
    ↓
添加位置编码 (Positional Encoding)
    ↓
Transformer Encoder × 3
    ↓
分类头 (256 → 128 → 64 → 2)
    ↓
输出 (2分类)
```

### 2.3 训练配置

**优化器**: AdamW
```python
learning_rate: 0.0001
weight_decay: 0.01
betas: (0.9, 0.999)
```

**学习率调度**: CosineAnnealingWarmRestarts
```python
T_0: 10
T_mult: 2
eta_min: 1e-6
```

**损失函数**: 加权交叉熵
```python
类别权重: [1.0, 3.423]  # 负类:正类
# 权重计算: class_counts[0] / class_counts[1]
```

**训练策略**:
- Batch Size: 256
- 最大Epochs: 100
- 早停耐心值: 15
- 梯度裁剪: max_norm=1.0
- GPU: NVIDIA A100-SXM4-40GB

---

## 三、实验结果

### 3.1 训练过程

```
总训练轮次: 82 epochs (早停)
最佳模型出现在: Epoch 65
训练批次数: 1,023 批/epoch
测试批次数: 256 批
```

**训练曲线关键节点**:

| Epoch | Train Loss | Train Acc | Test F1 | Test AUC | 备注 |
|-------|-----------|-----------|---------|----------|------|
| 5 | 0.6376 | 0.6246 | 0.4381 | 0.6863 | - |
| 10 | 0.6319 | 0.6292 | 0.4395 | 0.6900 | - |
| 20 | 0.6272 | 0.6355 | 0.4437 | 0.6942 | - |
| 40 | 0.6212 | 0.6431 | 0.4452 | 0.6978 | - |
| 65 | 0.6153 | 0.6516 | 0.4470 | **0.6997** | ✓ 最佳 |
| 82 | - | - | - | - | 早停 |

### 3.2 最终性能指标

**整体指标**:
```
准确率 (Accuracy):   64.35%
精确率 (Precision):  34.41%
召回率 (Recall):     63.68%
F1分数 (F1-Score):   44.68%
AUC-ROC:            70.00% ⭐
```

### 3.3 详细分类报告

| 类别 | 精确率 | 召回率 | F1分数 | 样本量 |
|-----|--------|--------|--------|--------|
| 无再入院 | 0.86 | 0.65 | 0.74 | 50,632 |
| 再入院 | 0.34 | 0.64 | 0.45 | 14,792 |
| **加权平均** | **0.74** | **0.64** | **0.67** | **65,424** |

### 3.4 混淆矩阵

```
                 预测结果
              无再入院   再入院
实际  无再入院   32,681   17,951
      再入院      5,373    9,419

真负例 (TN): 32,681   假正例 (FP): 17,951
假负例 (FN):  5,373   真正例 (TP):  9,419
```

**性能分析**:
- 真负例率 (特异性): 64.54%
- 真正例率 (灵敏度): 63.68%
- 假正例率: 35.46%
- 假负例率: 36.32%

### 3.5 阈值优化（可选）

通过 Precision-Recall 曲线优化决策阈值:

```
默认阈值: 0.500
优化阈值: 约 0.3-0.4 (根据PR曲线)
```

当前模型使用默认阈值 0.5，可以根据实际业务需求调整:
- 提高阈值 → 提升精确率，降低召回率（减少误报）
- 降低阈值 → 提升召回率，降低精确率（减少漏报）

---

## 四、结果解读

### 4.1 模型优势

✓ **AUC达到0.70** - 说明模型有较好的区分能力  
✓ **召回率63.68%** - 能找出约2/3的再入院患者  
✓ **稳定训练** - 82个epoch稳步提升，无过拟合

### 4.2 存在问题

⚠️ **精确率较低(34.41%)** - 预测为再入院的患者中，约2/3是误报  
⚠️ **类别不平衡** - 正类样本仅占22.61%，影响模型学习  
⚠️ **数据损失** - 清洗过程损失40.1%的数据

### 4.3 业务价值

**适用场景**:
- 高风险患者初筛（利用高召回率）
- 医疗资源预分配
- 患者出院随访优先级排序

**不适用场景**:
- 直接临床决策（精确率不足）
- 需要高精度的自动化系统

---

## 五、文件说明

```
项目文件结构:
├── 2025Fall_CS526_GroupProject.ipynb    # 主notebook
├── readmission_features_30d_v1.csv     # 原始数据(未上传)
├── best_transformer_model.pth          # 最佳模型权重
└── transformer_results.png             # ROC & PR曲线图
```

---

## 六、后续优化方向

### 6.1 数据层面
- [ ] 尝试插补法处理缺失值（目前直接删除）
- [ ] 增加特征交互项
- [ ] 时间序列特征工程（如就诊间隔、季节性）

### 6.2 模型层面
- [ ] 尝试SMOTE等过采样方法平衡数据
- [ ] 调整类别权重（当前3.42可能不是最优）
- [ ] 尝试Focal Loss处理类别不平衡
- [ ] 集成学习（Transformer + XGBoost）

### 6.3 评估层面
- [ ] 增加成本敏感评估（FN vs FP的实际成本）
- [ ] 按患者子群体分析（年龄、性别、疾病类型）
- [ ] 特征重要性分析（Attention权重可视化）

---

## 七、代码快速使用

### 加载训练好的模型
```python
model = TransformerClassifier(input_dim=41, d_model=256, nhead=8, num_layers=3)
model.load_state_dict(torch.load('best_transformer_model.pth'))
model.eval()
```

### 预测新数据
```python
# 假设 new_data 已经过相同的预处理
new_data_tensor = torch.FloatTensor(new_data).to(device)
with torch.no_grad():
    outputs = model(new_data_tensor)
    probs = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probs, dim=1)
    readmit_probs = probs[:, 1].cpu().numpy()  # 再入院概率
```

---

**最后更新**: 2025年10月  
**运行环境**: Google Colab (A100 GPU)  
**PyTorch版本**: 2.8.0+cu126