# 荔枝预测模型

## 概述

这是一个整合的荔枝预测模型，包含数据加载、清洗、训练、预测、在线学习、结果导出等完整功能。模型使用SARIMA（季节性ARIMA）算法进行时间序列预测。

## 主要功能

### 1. 数据输入适配
- 支持从`MonitorLitchiSummaryEntity`实体类加载数据
- 支持从`PageData`对象列表加载数据
- 自动处理数据类型转换和格式验证

### 2. 模型训练
- 支持多种SARIMA参数组合的自动训练
- 自动选择最佳模型（基于AIC指标）
- 支持模型序列化保存和加载

### 3. 预测功能
- 支持单步和多步预测
- 避免数据泄漏，确保预测的准确性
- 包含趋势和季节性模式分析

### 4. 数据清洗
- 异常值检测和处理（IQR方法）
- 季节性调整
- 数据平滑处理
- 缺失值处理
- 数据标准化

### 5. 结果导出
- JSON格式输出
- 包含预测值和模拟环境数据
- 支持按荔枝园分组的数据结构

## 使用方法

### 基本使用

```java
// 创建模型实例
IntegratedLitchiPredictionModel model = new IntegratedLitchiPredictionModel(1, 1, 1, 0, 0, 0, 7);

// 训练模型
model.saveDefaultModel("data_path", "model_path");

// 加载模型
IntegratedLitchiPredictionModel loadedModel = IntegratedLitchiPredictionModel.loadModel("model_path");

// 进行预测
List<Double> predictions = loadedModel.predictForCombination(20.0, 1, 30);
```

### 使用PageData进行预测

```java
List<PageData> pageDataList = // 获取PageData列表
model.predictWithPageData(pageDataList);
```

### 使用MonitorLitchiSummaryEntity进行训练

```java
List<MonitorLitchiSummaryEntity> entityList = // 获取实体列表
Map<String, ModelResult> results = model.trainAllModels(entityList);
```

## 数据格式

### MonitorLitchiSummaryEntity字段
- `litchiId`: 果园ID (Long)
- `thresholdType`: 阈值类型 (String)
- `averageDay`: 日平均值 (BigDecimal)
- `testingTime`: 检测时间 (Date)
- `summaryTime`: 汇总时间 (Date)
- `projectId`: 项目ID (Long)
- `monitorMax`: 最高值 (BigDecimal)
- `monitorMin`: 最低值 (BigDecimal)
- `equipmentType`: 设备类型 (String)

### PageData字段
- `litchiId`: 果园ID (String)
- `thresholdType`: 阈值类型 (String)
- `averageDay`: 日平均值 (String)
- `testingTime`: 检测时间 (String)
- `summaryTime`: 汇总时间 (String)
- `projectId`: 项目ID (String)
- `monitorMax`: 最高值 (String)
- `monitorMin`: 最低值 (String)
- `equipmentType`: 设备类型 (String)

## 模型参数

### SARIMA参数
- `p`: 非季节性AR阶数
- `d`: 非季节性差分阶数
- `q`: 非季节性MA阶数
- `P`: 季节性AR阶数
- `D`: 季节性差分阶数
- `Q`: 季节性MA阶数
- `s`: 季节性周期

### 默认参数组合
模型会自动尝试多种参数组合：
1. SARIMA(1,1,1)(1,0,1,7) - 周季节性
2. SARIMA(1,1,1)(0,0,0,7) - 无季节性
3. SARIMA(2,1,2)(1,0,1,7) - 周季节性
4. 等等...

## 输出格式

### JSON输出结构
```json
{
  "data": [
    {
      "litchiName": "果地20",
      "dataList": [
        {
          "time": "2024-01-01",
          "humidity": 65.2,
          "temperature": 26.8,
          "PH": 6.7,
          "salinity": 0.32
        }
      ]
    }
  ]
}
```

## 注意事项

1. 数据质量：模型对数据质量要求较高，建议在训练前进行充分的数据清洗
2. 数据量：每个组合至少需要20个数据点才能进行有效训练
3. 预测精度：模型会自动选择最佳参数组合，但预测精度仍取决于数据质量和数量
4. 内存使用：大量数据训练时注意内存使用情况

## 扩展功能

- 支持在线学习，可以定期更新模型参数
- 支持模型性能评估，包括MSE、MAE、R²等指标
- 支持批量预测和结果导出
- 支持多种数据源和格式的输入
