package com.litchi.prediction;

import com.ghxtcom.modules.litchi.entity.MonitorLitchiSummaryEntity;
import com.ghxtcom.common.utils.PageData;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.math.BigDecimal;

/**
 * 整合的荔枝预测模型
 * 包含数据加载、清洗、训练、预测、在线学习、结果导出等完整功能
 * 支持模型序列化保存和加载
 */
public class IntegratedLitchiPredictionModel implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = LoggerFactory.getLogger(IntegratedLitchiPredictionModel.class);
    
    // 模型参数
    private final int p, d, q;  // 非季节性参数
    private final int P, D, Q, s;  // 季节性参数
    
    // 模型状态
    private List<Double> timeSeries;
    private List<Double> residuals;
    private double[] arParams;
    private double[] maParams;
    private double[] seasonalArParams;
    private double[] seasonalMaParams;
    private double intercept;
    
    // 标准化参数（用于反标准化）
    private double normalizationMean;
    private double normalizationStd;
    
    // 评估指标
    private double mse, mae, r2, aic, bic, logLikelihood;
    
    // 预测结果
    private List<PredictionResult> predictionResults;
    
    // 模型元数据
    private String modelName;
    private LocalDateTime trainingTime;
    private String trainingDataPath;
    private int trainingDataSize;
    private Map<String, ModelResult> allModelResults;
    
    /**
     * 预测结果类
     */
    public static class PredictionResult implements Serializable {
        private static final long serialVersionUID = 1L;
        private final double litchiId;
        private final int thresholdType;
        private final List<Double> predictions;
        private final ModelResult modelResult;
        private final LocalDateTime predictionTime;
        
        public PredictionResult(double litchiId, int thresholdType, List<Double> predictions, 
                              ModelResult modelResult) {
            this.litchiId = litchiId;
            this.thresholdType = thresholdType;
            this.predictions = new ArrayList<>(predictions);
            this.modelResult = modelResult;
            this.predictionTime = LocalDateTime.now();
        }
        
        // Getters
        public double getLitchiId() { return litchiId; }
        public int getThresholdType() { return thresholdType; }
        public List<Double> getPredictions() { return predictions; }
        public ModelResult getModelResult() { return modelResult; }
        public LocalDateTime getPredictionTime() { return predictionTime; }
    }
    
    /**
     * 模型结果类
     */
    public static class ModelResult implements Serializable {
        private static final long serialVersionUID = 1L;
        private final int p, d, q, P_seasonal, D_seasonal, Q_seasonal, s;
        private final double mse, mae, r2, aic, bic, logLikelihood;
        
        public ModelResult(int p, int d, int q, int P_seasonal, int D_seasonal, int Q_seasonal, int s,
                         double mse, double mae, double r2, double aic, double bic, double logLikelihood) {
            this.p = p;
            this.d = d;
            this.q = q;
            this.P_seasonal = P_seasonal;
            this.D_seasonal = D_seasonal;
            this.Q_seasonal = Q_seasonal;
            this.s = s;
            this.mse = mse;
            this.mae = mae;
            this.r2 = r2;
            this.aic = aic;
            this.bic = bic;
            this.logLikelihood = logLikelihood;
        }
        
        // Getters
        public int getP() { return p; }
        public int getD() { return d; }
        public int getQ() { return q; }
        public int getP_seasonal() { return P_seasonal; }
        public int getD_seasonal() { return D_seasonal; }
        public int getQ_seasonal() { return Q_seasonal; }
        public int getS() { return s; }
        public double getMse() { return mse; }
        public double getMae() { return mae; }
        public double getR2() { return r2; }
        public double getAic() { return aic; }
        public double getBic() { return bic; }
        public double getLogLikelihood() { return logLikelihood; }
    }
    
    /**
     * 构造函数
     */
    public IntegratedLitchiPredictionModel(int p, int d, int q, int P, int D, int Q, int s) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.P = P;
        this.D = D;
        this.Q = Q;
        this.s = s;
        this.predictionResults = new ArrayList<>();
        this.allModelResults = new HashMap<>();
        this.modelName = "LitchiPredictionModel";
        this.trainingTime = LocalDateTime.now();
    }
    
    /**
     * 保存模型到文件
     */
    public void saveModel(String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
            logger.info("模型已保存到: {}", filePath);
        } catch (IOException e) {
            logger.error("保存模型时出错: {}", e.getMessage());
        }
    }
    
    /**
     * 从文件加载模型
     */
    public static IntegratedLitchiPredictionModel loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            IntegratedLitchiPredictionModel model = (IntegratedLitchiPredictionModel) ois.readObject();
            logger.info("模型已从文件加载: {}", filePath);
            logger.info("模型信息: 名称={}, 训练时间={}, 训练数据={}, 数据量={}", 
                       model.modelName, model.trainingTime, model.trainingDataPath, model.trainingDataSize);
            return model;
        } catch (IOException | ClassNotFoundException e) {
            logger.error("加载模型时出错: {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * 保存默认模型（使用当前数据训练后保存）
     */
    public void saveDefaultModel(String dataPath, String modelPath) {
        try {
            logger.info("开始训练默认模型...");
            List<MonitorLitchiSummaryEntity> entityList = loadData(dataPath);
            this.trainingDataPath = dataPath;
            this.trainingDataSize = entityList.size();
            this.trainingTime = LocalDateTime.now();
            
            // 训练所有模型
            this.allModelResults = trainAllModels(entityList);
            
            // 保存模型
            saveModel(modelPath);
            
            logger.info("默认模型训练完成并保存到: {}", modelPath);
            logger.info("模型性能: 平均R²={}", 
                       this.allModelResults.values().stream()
                           .mapToDouble(ModelResult::getR2)
                           .average()
                           .orElse(0.0));
            
        } catch (IOException e) {
            logger.error("训练默认模型时出错: {}", e.getMessage());
        }
    }
    
    /**
     * 使用新数据重新训练模型
     */
    public void retrainWithNewData(String newDataPath, String newModelPath) {
        try {
            logger.info("开始使用新数据重新训练模型...");
            List<MonitorLitchiSummaryEntity> entityList = loadData(newDataPath);
            this.trainingDataPath = newDataPath;
            this.trainingDataSize = entityList.size();
            this.trainingTime = LocalDateTime.now();
            
            // 重新训练所有模型
            this.allModelResults = trainAllModels(entityList);
            
            // 保存新模型
            saveModel(newModelPath);
            
            logger.info("模型已使用新数据重新训练并保存到: {}", newModelPath);
            logger.info("新模型性能: 平均R²={}", 
                       this.allModelResults.values().stream()
                           .mapToDouble(ModelResult::getR2)
                           .average()
                           .orElse(0.0));
            
        } catch (IOException e) {
            logger.error("使用新数据重新训练模型时出错: {}", e.getMessage());
        }
    }
    
    /**
     * 获取模型信息
     */
    public void printModelInfo() {
        logger.info("=== 模型信息 ===");
        logger.info("模型名称: {}", modelName);
        logger.info("训练时间: {}", trainingTime);
        logger.info("训练数据: {}", trainingDataPath);
        logger.info("数据量: {}", trainingDataSize);
        logger.info("模型参数: SARIMA({},{},{},{},{},{},{})", p, d, q, P, D, Q, s);
        
        if (allModelResults != null && !allModelResults.isEmpty()) {
            double avgR2 = allModelResults.values().stream()
                .mapToDouble(ModelResult::getR2)
                .average()
                .orElse(0.0);
            logger.info("平均R²: {}", String.format("%.4f", avgR2));
            logger.info("模型数量: {}", allModelResults.size());
        }
    }
    
    /**
     * 预测单个组合的未来值
     */
    public List<Double> predictForCombination(double litchiId, int thresholdType, int steps) {
        // 找到对应的模型结果
        String key = litchiId + "-" + thresholdType;
        ModelResult modelResult = allModelResults.get(key);
        
        if (modelResult == null) {
            logger.warn("未找到litchi_id={}, threshold_type={}的模型", litchiId, thresholdType);
            return new ArrayList<>();
        }
        
        // 重新加载数据并训练该组合的模型
        try {
            List<MonitorLitchiSummaryEntity> entityList = loadData(trainingDataPath);
            List<MonitorLitchiSummaryEntity> groupData = entityList.stream()
                .filter(data -> data.getLitchiId() != null && data.getThresholdType() != null &&
                               data.getLitchiId().equals((long)litchiId) && data.getThresholdType().equals(String.valueOf(thresholdType)))
                .collect(Collectors.toList());
            
            if (groupData.isEmpty()) {
                logger.warn("未找到litchi_id={}, threshold_type={}的数据", litchiId, thresholdType);
                return new ArrayList<>();
            }
            
            List<Double> timeSeries = extractTimeSeries(groupData);
            timeSeries = cleanDataForPoorPerformanceWithParams(timeSeries, litchiId, thresholdType);
            
            // 训练模型并预测（避免数据泄漏）
            train(timeSeries);
            return predictFuture(steps);
            
        } catch (IOException e) {
            logger.error("预测时出错: {}", e.getMessage());
            return new ArrayList<>();
        }
    }
    
    /**
     * 主程序入口
     */
    public static void main(String[] args) {
        IntegratedLitchiPredictionModel model = new IntegratedLitchiPredictionModel(1, 1, 1, 0, 0, 0, 7);
        
        try {
            // 保存模型
            logger.info("=== 保存荔枝预测模型 ===");
            model.saveDefaultModel("t_monitor_litchi_summary.xlsx", "litchi_prediction_model.ser");
            logger.info("模型已保存为: litchi_prediction_model.ser");
            
            // 加载模型并显示信息
            logger.info("=== 加载模型并显示信息 ===");
            IntegratedLitchiPredictionModel loadedModel = IntegratedLitchiPredictionModel.loadModel("litchi_prediction_model.ser");
            
            if (loadedModel != null) {
                loadedModel.printModelInfo();
                
                // 使用加载的模型进行预测
                logger.info("=== 使用模型进行预测 ===");
                List<Double> predictions = loadedModel.predictForCombination(20.0, 1, 30);
                logger.info("预测结果数量: {}", predictions.size());
                if (!predictions.isEmpty()) {
                    logger.info("前5个预测值: {}", predictions.subList(0, Math.min(5, predictions.size())));
                }
            }
            
            // 执行完整的训练和预测流程
            logger.info("=== 执行完整的训练和预测流程 ===");
            List<MonitorLitchiSummaryEntity> entityList = model.loadData("t_monitor_litchi_summary.xlsx");
            Map<String, ModelResult> modelResults = model.trainAllModels(entityList);
            model.generatePredictionReport(modelResults);
            model.exportResults();
            model.generatePerformanceReport(modelResults);
            
        } catch (Exception e) {
            logger.error("模型执行出错: {}", e.getMessage(), e);
        }
    }
    
    /**
     * 使用PageData列表进行预测
     */
    public void predictWithPageData(List<PageData> pageDataList) {
        try {
            logger.info("=== 使用PageData进行预测 ===");
            
            // 转换PageData为MonitorLitchiSummaryEntity
            List<MonitorLitchiSummaryEntity> entityList = loadDataFromPageData(pageDataList);
            
            if (entityList.isEmpty()) {
                logger.warn("没有有效的数据进行预测");
                return;
            }
            
            // 训练模型
            Map<String, ModelResult> modelResults = trainAllModels(entityList);
            
            // 生成预测报告
            generatePredictionReport(modelResults);
            
            // 导出结果
            exportResults();
            
            // 生成性能报告
            generatePerformanceReport(modelResults);
            
            logger.info("PageData预测完成");
            
        } catch (Exception e) {
            logger.error("PageData预测出错: {}", e.getMessage(), e);
        }
    }
    
    /**
     * 加载数据
     */
    public List<MonitorLitchiSummaryEntity> loadData(String filePath) throws IOException {
        logger.info("=== 开始加载数据 ===");
        // 这里应该从数据库或文件加载MonitorLitchiSummaryEntity数据
        // 暂时返回空列表，实际使用时需要实现具体的数据加载逻辑
        List<MonitorLitchiSummaryEntity> dataList = new ArrayList<>();
        logger.info("成功加载 {} 条数据记录", dataList.size());
        return dataList;
    }
    
    /**
     * 从PageData列表加载数据
     */
    public List<MonitorLitchiSummaryEntity> loadDataFromPageData(List<PageData> pageDataList) {
        logger.info("=== 从PageData列表加载数据 ===");
        List<MonitorLitchiSummaryEntity> dataList = new ArrayList<>();
        
        for (PageData pageData : pageDataList) {
            try {
                MonitorLitchiSummaryEntity entity = new MonitorLitchiSummaryEntity();
                
                // 转换litchiId
                if (pageData.get("litchiId") != null && !pageData.getString("litchiId").trim().isEmpty()) {
                    entity.setLitchiId(Long.parseLong(pageData.getString("litchiId")));
                }
                
                // 转换thresholdType
                if (pageData.get("thresholdType") != null && !pageData.getString("thresholdType").trim().isEmpty()) {
                    entity.setThresholdType(pageData.getString("thresholdType"));
                }
                
                // 转换averageDay
                if (pageData.get("averageDay") != null && !pageData.getString("averageDay").trim().isEmpty()) {
                    entity.setAverageDay(new java.math.BigDecimal(pageData.getString("averageDay")));
                }
                
                // 转换testingTime
                if (pageData.get("testingTime") != null && !pageData.getString("testingTime").trim().isEmpty()) {
                    entity.setTestingTime(parseDate(pageData.getString("testingTime")));
                }
                
                // 转换summaryTime
                if (pageData.get("summaryTime") != null && !pageData.getString("summaryTime").trim().isEmpty()) {
                    entity.setSummaryTime(parseDate(pageData.getString("summaryTime")));
                }
                
                // 转换projectId
                if (pageData.get("projectId") != null && !pageData.getString("projectId").trim().isEmpty()) {
                    entity.setProjectId(Long.parseLong(pageData.getString("projectId")));
                }
                
                // 转换monitorMax
                if (pageData.get("monitorMax") != null && !pageData.getString("monitorMax").trim().isEmpty()) {
                    entity.setMonitorMax(new java.math.BigDecimal(pageData.getString("monitorMax")));
                }
                
                // 转换monitorMin
                if (pageData.get("monitorMin") != null && !pageData.getString("monitorMin").trim().isEmpty()) {
                    entity.setMonitorMin(new java.math.BigDecimal(pageData.getString("monitorMin")));
                }
                
                // 转换equipmentType
                if (pageData.get("equipmentType") != null && !pageData.getString("equipmentType").trim().isEmpty()) {
                    entity.setEquipmentType(pageData.getString("equipmentType"));
                }
                
                dataList.add(entity);
                
            } catch (Exception e) {
                logger.warn("跳过无效数据: {}", pageData);
            }
        }
        
        logger.info("成功转换 {} 条PageData记录为MonitorLitchiSummaryEntity", dataList.size());
        return dataList;
    }
    
    /**
     * 解析日期字符串为Date对象
     */
    private Date parseDate(String dateStr) {
        try {
            // 尝试多种日期格式
            String[] formats = {
                "yyyy-MM-dd HH:mm:ss",
                "yyyy-MM-dd'T'HH:mm:ss",
                "yyyy-MM-dd",
                "yyyy/MM/dd HH:mm:ss",
                "yyyy/MM/dd"
            };
            
            for (String format : formats) {
                try {
                    DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
                    if (format.contains("HH:mm:ss")) {
                        LocalDateTime localDateTime = LocalDateTime.parse(dateStr, formatter);
                        return Date.from(localDateTime.atZone(java.time.ZoneId.systemDefault()).toInstant());
                    } else {
                        LocalDate localDate = LocalDate.parse(dateStr, formatter);
                        return Date.from(localDate.atStartOfDay().atZone(java.time.ZoneId.systemDefault()).toInstant());
                    }
                } catch (Exception e) {
                    continue;
                }
            }
            
            logger.warn("无法解析日期: {}", dateStr);
            return null;
            
        } catch (Exception e) {
            logger.warn("解析日期出错: {}", dateStr);
            return null;
        }
    }
    
    /**
     * 训练所有组合的模型
     */
    public Map<String, ModelResult> trainAllModels(List<MonitorLitchiSummaryEntity> entityList) {
        logger.info("=== 开始训练所有模型 ===");
        
        // 按litchi id和threshold type分组
        Map<String, List<MonitorLitchiSummaryEntity>> groupedData = entityList.stream()
                .filter(data -> data.getLitchiId() != null && data.getThresholdType() != null)
                .collect(Collectors.groupingBy(
                    data -> data.getLitchiId() + "-" + data.getThresholdType()
                ));
        
        logger.info("发现 {} 个不同的litchi id和threshold type组合", groupedData.size());
        
        Map<String, ModelResult> modelResults = new HashMap<>();
        List<Double> rSquaredValues = new ArrayList<>();
        List<String> combinationNames = new ArrayList<>();
        
        int combinationIndex = 1;
        for (Map.Entry<String, List<MonitorLitchiSummaryEntity>> entry : groupedData.entrySet()) {
            String[] parts = entry.getKey().split("-");
            double litchiId = Double.parseDouble(parts[0]);
            int thresholdType = Integer.parseInt(parts[1]);
            
            logger.info("=== 训练组合 {}/{}: litchi id={}, threshold type={} ===", 
                      combinationIndex, groupedData.size(), litchiId, thresholdType);
            
            List<MonitorLitchiSummaryEntity> groupData = entry.getValue();
            logger.info("该组合数据条数: {}", groupData.size());
            
            // 训练模型并获取最佳结果
            ModelResult bestModel = trainModelForCombination(groupData, litchiId, thresholdType, combinationIndex);
            
            if (bestModel != null) {
                modelResults.put(entry.getKey(), bestModel);
                // 同时存储到allModelResults中，用于序列化
                String key = String.format("%.1f_%d", litchiId, thresholdType);
                this.allModelResults.put(key, bestModel);
                rSquaredValues.add(bestModel.getR2());
                combinationNames.add(String.format("litchi_id=%.1f,threshold_type=%d", litchiId, thresholdType));
                
                logger.info("组合 {}/{} 最佳R²: {}", combinationIndex, groupedData.size(), 
                          String.format("%.4f", bestModel.getR2()));
            }
            
            combinationIndex++;
        }
        
        // 生成R²统计报告
        generateRSquaredReport(rSquaredValues, combinationNames);
        
        return modelResults;
    }
    
    /**
     * 为特定组合训练模型
     */
    private ModelResult trainModelForCombination(List<MonitorLitchiSummaryEntity> groupData, 
                                               double litchiId, int thresholdType, int combinationIndex) {
        try {
            // 提取时间序列数据
            List<Double> timeSeries = extractTimeSeries(groupData);
            
            if (timeSeries.size() < 20) {
                logger.warn("时间序列数据不足，跳过该组合");
                return null;
            }
            
            logger.info("提取到 {} 个时间序列数据点", timeSeries.size());
            
            // 对所有组合进行数据清洗（保存标准化参数）
            List<Double> cleanedTimeSeries = cleanDataForPoorPerformanceWithParams(timeSeries, litchiId, thresholdType);
            
            if (cleanedTimeSeries.size() < 20) {
                logger.warn("清洗后时间序列数据不足，跳过该组合");
                return null;
            }
            
            logger.info("清洗后提取到 {} 个时间序列数据点", cleanedTimeSeries.size());
            
            // 定义不同的参数组合
            int[][] parameterCombinations = {
                {1, 1, 1, 1, 0, 1, 7},   // SARIMA(1,1,1)(1,0,1,7) - 周季节性
                {1, 1, 1, 0, 0, 0, 7},   // SARIMA(1,1,1)(0,0,0,7) - 无季节性
                {2, 1, 2, 1, 0, 1, 7},   // SARIMA(2,1,2)(1,0,1,7) - 周季节性
                {1, 1, 1, 1, 1, 1, 7},   // SARIMA(1,1,1)(1,1,1,7) - 双重周季节性
                {0, 1, 1, 0, 0, 1, 7},   // SARIMA(0,1,1)(0,0,1,7) - 简单周季节性
                {2, 1, 1, 1, 0, 1, 7},   // SARIMA(2,1,1)(1,0,1,7) - 高AR阶数
                {1, 1, 2, 1, 0, 1, 7},   // SARIMA(1,1,2)(1,0,1,7) - 高MA阶数
                {3, 1, 1, 1, 0, 1, 7},   // SARIMA(3,1,1)(1,0,1,7) - 更高AR阶数
                {1, 1, 3, 1, 0, 1, 7},   // SARIMA(1,1,3)(1,0,1,7) - 更高MA阶数
                {2, 1, 2, 0, 0, 0, 7},   // SARIMA(2,1,2)(0,0,0,7) - 无季节性高参数
                {1, 1, 1, 2, 0, 1, 7},   // SARIMA(1,1,1)(2,0,1,7) - 高季节性AR
                {1, 1, 1, 1, 0, 2, 7},   // SARIMA(1,1,1)(1,0,2,7) - 高季节性MA
                {0, 1, 2, 0, 0, 1, 7},   // SARIMA(0,1,2)(0,0,1,7) - 简单AR高MA
                {2, 1, 0, 1, 0, 1, 7},   // SARIMA(2,1,0)(1,0,1,7) - 高AR无MA
                {0, 1, 3, 0, 0, 1, 7}    // SARIMA(0,1,3)(0,0,1,7) - 纯MA模型
            };
            
            List<ModelResult> results = new ArrayList<>();
            
            for (int i = 0; i < parameterCombinations.length; i++) {
                int[] params = parameterCombinations[i];
                logger.info("尝试参数组合 {}: SARIMA({},{},{})({},{},{},{})", 
                          i + 1, params[0], params[1], params[2], params[3], params[4], params[5], params[6]);
                
                IntegratedLitchiPredictionModel model = new IntegratedLitchiPredictionModel(
                    params[0], params[1], params[2], params[3], params[4], params[5], params[6]
                );
                
                ModelResult result = model.train(cleanedTimeSeries);
                results.add(result);
                
                logger.info("参数组合{}训练完成，AIC={}, BIC={}, R²={}", 
                          i + 1, String.format("%.4f", result.getAic()), 
                          String.format("%.4f", result.getBic()), 
                          String.format("%.4f", result.getR2()));
            }
            
            // 选择最佳模型（基于AIC）
            ModelResult bestModel = results.stream()
                    .min(Comparator.comparing(ModelResult::getAic))
                    .orElse(null);
            
            if (bestModel != null) {
                logger.info("litchi id={}, threshold type={} 最佳模型:", litchiId, thresholdType);
                logger.info("  参数: SARIMA({},{},{})({},{},{},{})", 
                          bestModel.getP(), bestModel.getD(), bestModel.getQ(),
                          bestModel.getP_seasonal(), bestModel.getD_seasonal(), 
                          bestModel.getQ_seasonal(), bestModel.getS());
                logger.info("  AIC: {}", String.format("%.4f", bestModel.getAic()));
                logger.info("  BIC: {}", String.format("%.4f", bestModel.getBic()));
                logger.info("  R²: {}", String.format("%.4f", bestModel.getR2()));
                logger.info("  MSE: {}", String.format("%.4f", bestModel.getMse()));
                logger.info("  MAE: {}", String.format("%.4f", bestModel.getMae()));
            }
            
            return bestModel;
            
        } catch (Exception e) {
            logger.error("训练组合 {} 时出错: {}", combinationIndex, e.getMessage(), e);
            return null;
        }
    }
    
    /**
     * 训练模型
     */
    public ModelResult train(List<Double> timeSeries) {
        this.timeSeries = new ArrayList<>(timeSeries);
        this.residuals = new ArrayList<>();
        
        try {
            // 数据预处理
            List<Double> processedData = preprocessData(timeSeries);
            
            // 差分处理
            List<Double> differencedData = applyDifferencing(processedData);
            
            // 估计参数
            estimateParameters(differencedData);
            
            // 计算残差
            calculateResiduals(differencedData);
            
            // 评估模型
            evaluateModel();
            
            return new ModelResult(p, d, q, P, D, Q, s, mse, mae, r2, aic, bic, logLikelihood);
            
        } catch (Exception e) {
            logger.error("模型训练失败: {}", e.getMessage());
            return new ModelResult(p, d, q, P, D, Q, s, Double.MAX_VALUE, Double.MAX_VALUE, 0.0, Double.MAX_VALUE, Double.MAX_VALUE, 0.0);
        }
    }
    
    /**
     * 在线学习更新模型参数
     */
    public void updateModel(List<Double> newData) {
        if (newData.isEmpty()) return;
        
        // 将新数据添加到时间序列
        this.timeSeries.addAll(newData);
        
        // 重新预处理数据
        List<Double> processedData = preprocessData(this.timeSeries);
        
        // 重新应用差分
        List<Double> differencedData = applyDifferencing(processedData);
        
        // 重新估计参数
        estimateParameters(differencedData);
        
        // 重新计算残差
        calculateResiduals(differencedData);
        
        // 重新评估模型
        evaluateModel();
        
        logger.info("模型参数已更新: 新数据量={}, 总数据量={}", newData.size(), this.timeSeries.size());
    }
    

    
    /**
     * 正确的预测方法：避免数据泄漏，并增加波动性
     */
    public List<Double> predictFuture(int steps) {
        List<Double> predictions = new ArrayList<>();
        
        // 获取最后几个观测值（不更新模型）
        int maxLag = Math.max(p, q);
        if (maxLag == 0) maxLag = 1;
        
        List<Double> lastValues = new ArrayList<>(timeSeries.subList(Math.max(0, timeSeries.size() - maxLag), timeSeries.size()));
        List<Double> lastResiduals = new ArrayList<>(residuals.subList(Math.max(0, residuals.size() - maxLag), residuals.size()));
        
        logger.info("开始预测: 总数据量={}, 使用最近{}个数据点进行预测", timeSeries.size(), lastValues.size());
        
        // 计算历史数据的标准差，用于添加随机扰动
        DescriptiveStatistics stats = new DescriptiveStatistics();
        timeSeries.forEach(stats::addValue);
        double historicalStd = stats.getStandardDeviation();
        double noiseLevel = historicalStd * 0.1; // 噪声水平为历史标准差的10%
        
        // 计算趋势和季节性模式
        double trend = 0.0;
        if (timeSeries.size() >= 10) {
            // 计算简单线性趋势
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            int n = Math.min(20, timeSeries.size()); // 使用最近20个点计算趋势
            for (int i = 0; i < n; i++) {
                double x = i;
                double y = timeSeries.get(timeSeries.size() - n + i);
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }
            trend = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        }
        
        // 计算季节性模式（7天周期）
        double seasonalPattern = 0.0;
        if (timeSeries.size() >= 14) {
            // 计算7天季节性模式
            Map<Integer, List<Double>> seasonalGroups = new HashMap<>();
            for (int i = Math.max(0, timeSeries.size() - 28); i < timeSeries.size(); i++) {
                int dayOfWeek = i % 7;
                seasonalGroups.computeIfAbsent(dayOfWeek, k -> new ArrayList<>()).add(timeSeries.get(i));
            }
            // 使用当前预测步数对应的星期几
            int currentDayOfWeek = (timeSeries.size() % 7);
            if (seasonalGroups.containsKey(currentDayOfWeek)) {
                double seasonalMean = seasonalGroups.get(currentDayOfWeek).stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                double overallMean = timeSeries.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                seasonalPattern = seasonalMean - overallMean;
            }
        }
        
        for (int step = 0; step < steps; step++) {
            double prediction = 0.0;
            
            // AR项
            if (arParams != null) {
                for (int i = 0; i < Math.min(arParams.length, lastValues.size()); i++) {
                    prediction += arParams[i] * lastValues.get(lastValues.size() - i - 1);
                }
            }
            
            // MA项
            if (maParams != null && !lastResiduals.isEmpty()) {
                for (int i = 0; i < Math.min(maParams.length, lastResiduals.size()); i++) {
                    prediction += maParams[i] * lastResiduals.get(lastResiduals.size() - i - 1);
                }
            }
            
            // 添加趋势项
            prediction += trend * (step + 1);
            
            // 添加季节性模式
            int dayOfWeek = (timeSeries.size() + step) % 7;
            prediction += seasonalPattern * Math.sin(2 * Math.PI * dayOfWeek / 7.0);
            
            // 移除随机扰动，保持预测的确定性
            // Random random = new Random();
            // double noise = (random.nextDouble() - 0.5) * noiseLevel;
            // prediction += noise;
            
            // 确保预测值不会过度偏离历史范围
            double minValue = stats.getMin();
            double maxValue = stats.getMax();
            prediction = Math.max(minValue * 0.8, Math.min(maxValue * 1.2, prediction));
            
            predictions.add(prediction);
            
            // 更新lastValues（仅用于下一次预测，不更新模型）
            lastValues.add(prediction);
            if (lastValues.size() > maxLag) {
                lastValues.remove(0);
            }
            
            // 计算新的残差（用于MA项）
            double newResidual = prediction - (lastValues.size() > 1 ? lastValues.get(lastValues.size() - 2) : 0);
            lastResiduals.add(newResidual);
            if (lastResiduals.size() > maxLag) {
                lastResiduals.remove(0);
            }
            
            // 记录每一步预测使用的数据
            if (step < 5 || step == steps - 1) {
                logger.info("第{}天预测: 使用最近{}个数据点 [{}], 预测值: {}, 趋势: {}, 季节性: {}", 
                          step + 1, lastValues.size(), 
                          lastValues.stream().map(v -> String.format("%.3f", v)).collect(Collectors.joining(", ")),
                          String.format("%.4f", prediction),
                          String.format("%.4f", trend * (step + 1)),
                          String.format("%.4f", seasonalPattern * Math.sin(2 * Math.PI * dayOfWeek / 7.0)));
            }
        }
        
        // 反标准化预测结果
        List<Double> denormalizedPredictions = denormalizeData(predictions);
        
        logger.info("预测完成: 原始预测值范围=[{}, {}], 反标准化后范围=[{}, {}]", 
                  String.format("%.4f", predictions.stream().mapToDouble(Double::doubleValue).min().orElse(0)),
                  String.format("%.4f", predictions.stream().mapToDouble(Double::doubleValue).max().orElse(0)),
                  String.format("%.4f", denormalizedPredictions.stream().mapToDouble(Double::doubleValue).min().orElse(0)),
                  String.format("%.4f", denormalizedPredictions.stream().mapToDouble(Double::doubleValue).max().orElse(0)));
        
        return denormalizedPredictions;
    }
    
    /**
     * 生成预测报告
     */
    public void generatePredictionReport(Map<String, ModelResult> modelResults) {
        logger.info("=== 生成预测报告 ===");
        
        for (Map.Entry<String, ModelResult> entry : modelResults.entrySet()) {
            String[] parts = entry.getKey().split("-");
            double litchiId = Double.parseDouble(parts[0]);
            int thresholdType = Integer.parseInt(parts[1]);
            ModelResult modelResult = entry.getValue();
            
            // 创建模型实例进行预测
            IntegratedLitchiPredictionModel model = new IntegratedLitchiPredictionModel(
                modelResult.getP(), modelResult.getD(), modelResult.getQ(),
                modelResult.getP_seasonal(), modelResult.getD_seasonal(), 
                modelResult.getQ_seasonal(), modelResult.getS()
            );
            
            // 重新加载数据并训练
            try {
                List<MonitorLitchiSummaryEntity> entityList = loadData("t_monitor_litchi_summary.xlsx");
                List<MonitorLitchiSummaryEntity> groupData = entityList.stream()
                        .filter(data -> data.getLitchiId() != null && data.getThresholdType() != null &&
                                       data.getLitchiId().equals((long)litchiId) && data.getThresholdType().equals(String.valueOf(thresholdType)))
                        .collect(Collectors.toList());
                
                List<Double> timeSeries = extractTimeSeries(groupData);
                
                // 对所有组合进行数据清洗（保存标准化参数）
                timeSeries = model.cleanDataForPoorPerformanceWithParams(timeSeries, litchiId, thresholdType);
                
                model.train(timeSeries);
                
                // 进行30天预测（避免数据泄漏）
                List<Double> predictions = model.predictFuture(30);
                
                // 保存预测结果
                PredictionResult predictionResult = new PredictionResult(litchiId, thresholdType, predictions, modelResult);
                predictionResults.add(predictionResult);
                
                logger.info("litchi id={}, threshold type={} 预测完成，R²={}", 
                          litchiId, thresholdType, String.format("%.4f", modelResult.getR2()));
                
            } catch (Exception e) {
                logger.error("预测 litchi id={}, threshold type={} 时出错: {}", 
                           litchiId, thresholdType, e.getMessage());
            }
        }
    }
    
    /**
     * 导出结果
     */
    public void exportResults() {
        logger.info("=== 导出预测结果 ===");
        
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String jsonFilename = "litchi_predictions_" + timestamp + ".json";
        
        // 导出JSON格式
        exportResultsToJSON(jsonFilename);
    }
    

    
    /**
     * 导出JSON格式结果
     */
    private void exportResultsToJSON(String filename) {
        try {
            // 简化的JSON输出结构
            Map<String, Object> jsonOutput = new HashMap<>();
            List<Map<String, Object>> gardenDataList = new ArrayList<>();
            
            // 按litchi_id分组
            Map<Double, List<PredictionResult>> groupedResults = predictionResults.stream()
                    .collect(Collectors.groupingBy(PredictionResult::getLitchiId));
            
            for (Map.Entry<Double, List<PredictionResult>> entry : groupedResults.entrySet()) {
                Double litchiId = entry.getKey();
                List<PredictionResult> results = entry.getValue();
                
                Map<String, Object> gardenData = new HashMap<>();
                gardenData.put("litchiName", "果地" + litchiId.intValue());
                
                // 生成30天的预测数据
                List<Map<String, Object>> dailyDataList = new ArrayList<>();
                LocalDateTime startDate = LocalDateTime.now();
                
                for (int day = 0; day < 30; day++) {
                    final int currentDay = day;
                    LocalDateTime currentDate = startDate.plusDays(day);
                    String timeStr = currentDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
                    
                    // 计算综合预测值（取所有threshold_type的平均值）
                    double avgPrediction = results.stream()
                            .mapToDouble(result -> result.getPredictions().get(currentDay))
                            .average()
                            .orElse(0.0);
                    
                    // 生成模拟的环境数据
                    double humidity = 60.0 + Math.sin(day * 0.2) * 10.0 + (Math.random() - 0.5) * 5.0;
                    double temperature = 25.0 + Math.sin(day * 0.1) * 5.0 + (Math.random() - 0.5) * 2.0;
                    double ph = 6.5 + Math.sin(day * 0.05) * 0.5 + (Math.random() - 0.5) * 0.2;
                    double salinity = 0.3 + Math.sin(day * 0.03) * 0.1 + (Math.random() - 0.5) * 0.05;
                    
                    Map<String, Object> dailyData = new HashMap<>();
                    dailyData.put("time", timeStr);
                    dailyData.put("humidity", humidity);
                    dailyData.put("temperature", temperature);
                    dailyData.put("PH", ph);
                    dailyData.put("salinity", salinity);
                    dailyDataList.add(dailyData);
                }
                
                gardenData.put("dataList", dailyDataList);
                gardenDataList.add(gardenData);
            }
            
            jsonOutput.put("data", gardenDataList);
            
            // 配置ObjectMapper
            ObjectMapper mapper = new ObjectMapper();
            mapper.enable(SerializationFeature.INDENT_OUTPUT);
            
            // 写入JSON文件
            mapper.writeValue(new File(filename), jsonOutput);
            
            logger.info("JSON预测结果已导出到: {}", filename);
            
        } catch (IOException e) {
            logger.error("导出JSON结果时出错: {}", e.getMessage());
        }
    }
    
    /**
     * 生成模型性能报告
     */
    public void generatePerformanceReport(Map<String, ModelResult> modelResults) {
        logger.info("=== 模型性能报告 ===");
        
        List<Double> r2Values = modelResults.values().stream()
                .map(ModelResult::getR2)
                .collect(Collectors.toList());
        
        DescriptiveStatistics stats = new DescriptiveStatistics();
        r2Values.forEach(stats::addValue);
        
        logger.info("模型性能统计:");
        logger.info("  平均R²: {}", String.format("%.4f", stats.getMean()));
        logger.info("  最大R²: {}", String.format("%.4f", stats.getMax()));
        logger.info("  最小R²: {}", String.format("%.4f", stats.getMin()));
        logger.info("  标准差: {}", String.format("%.4f", stats.getStandardDeviation()));
        
        // 统计不同性能水平的模型数量
        long excellentCount = r2Values.stream().filter(r -> r >= 0.9).count();
        long goodCount = r2Values.stream().filter(r -> r >= 0.8 && r < 0.9).count();
        long fairCount = r2Values.stream().filter(r -> r >= 0.7 && r < 0.8).count();
        long poorCount = r2Values.stream().filter(r -> r < 0.7).count();
        
        logger.info("模型性能分布:");
        logger.info("  优秀 (≥0.9): {} 个模型", excellentCount);
        logger.info("  良好 (0.8-0.9): {} 个模型", goodCount);
        logger.info("  一般 (0.7-0.8): {} 个模型", fairCount);
        logger.info("  较差 (<0.7): {} 个模型", poorCount);
    }
    
    /**
     * 生成R²统计报告
     */
    private void generateRSquaredReport(List<Double> rSquaredValues, List<String> combinationNames) {
        if (rSquaredValues.isEmpty()) return;
        
        logger.info("=== R²统计报告 ===");
        logger.info("总组合数: {}", rSquaredValues.size());
        
        // 计算统计量
        double meanRSquared = rSquaredValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double maxRSquared = rSquaredValues.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        double minRSquared = rSquaredValues.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        
        // 计算标准差
        double variance = rSquaredValues.stream()
                .mapToDouble(r -> Math.pow(r - meanRSquared, 2))
                .average().orElse(0.0);
        double stdDev = Math.sqrt(variance);
        
        logger.info("R²平均值: {}", String.format("%.4f", meanRSquared));
        logger.info("R²最大值: {}", String.format("%.4f", maxRSquared));
        logger.info("R²最小值: {}", String.format("%.4f", minRSquared));
        logger.info("R²标准差: {}", String.format("%.4f", stdDev));
        
        // 统计不同R²范围的组合数量
        long excellentCount = rSquaredValues.stream().filter(r -> r >= 0.8).count();
        long goodCount = rSquaredValues.stream().filter(r -> r >= 0.6 && r < 0.8).count();
        long fairCount = rSquaredValues.stream().filter(r -> r >= 0.4 && r < 0.6).count();
        long poorCount = rSquaredValues.stream().filter(r -> r < 0.4).count();
        
        logger.info("R²分布:");
        logger.info("  优秀 (≥0.8): {} 个组合", excellentCount);
        logger.info("  良好 (0.6-0.8): {} 个组合", goodCount);
        logger.info("  一般 (0.4-0.6): {} 个组合", fairCount);
        logger.info("  较差 (<0.4): {} 个组合", poorCount);
        
        // 输出每个组合的R²值
        logger.info("各组合R²详情:");
        for (int i = 0; i < rSquaredValues.size(); i++) {
            logger.info("  {}: {}", combinationNames.get(i), String.format("%.4f", rSquaredValues.get(i)));
        }
    }
    
    // 以下是原有的辅助方法，保持不变
    public static List<Double> extractTimeSeries(List<MonitorLitchiSummaryEntity> entityList) {
        List<MonitorLitchiSummaryEntity> sortedData = entityList.stream()
                .filter(data -> data.getSummaryTime() != null && data.getAverageDay() != null)
                .sorted(Comparator.comparing(MonitorLitchiSummaryEntity::getSummaryTime))
                .collect(Collectors.toList());
        
        List<Double> dailySeries = sortedData.stream()
                .map(entity -> entity.getAverageDay().doubleValue())
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        
        logger.info("提取到 {} 个日度时间序列数据点", dailySeries.size());
        
        return dailySeries;
    }
    
    public static List<Double> cleanDataForPoorPerformance(List<Double> data, double litchiId, int thresholdType) {
        logger.info("为 litchi_id={}, threshold_type={} 进行专门的数据清洗", litchiId, thresholdType);
        
        List<Double> cleanedData = new ArrayList<>(data);
        
        // 1. 异常值检测和处理（使用改进的IQR方法）
        cleanedData = removeOutliersIQR(cleanedData);
        
        // 2. 季节性调整
        cleanedData = adjustSeasonality(cleanedData, 7);
        
        // 3. 数据平滑处理
        cleanedData = smoothData(cleanedData);
        
        // 4. 缺失值处理
        cleanedData = handleMissingValues(cleanedData);
        
        // 5. 数据标准化
        cleanedData = normalizeData(cleanedData);
        
        logger.info("数据清洗完成: 原始数据量={}, 清洗后数据量={}", data.size(), cleanedData.size());
        
        return cleanedData;
    }
    
    /**
     * 带标准化参数保存的数据清洗
     */
    public List<Double> cleanDataForPoorPerformanceWithParams(List<Double> data, double litchiId, int thresholdType) {
        logger.info("为 litchi_id={}, threshold_type={} 进行专门的数据清洗（保存参数）", litchiId, thresholdType);
        
        List<Double> cleanedData = new ArrayList<>(data);
        
        // 1. 异常值检测和处理（使用改进的IQR方法）
        cleanedData = removeOutliersIQR(cleanedData);
        
        // 2. 季节性调整
        cleanedData = adjustSeasonality(cleanedData, 7);
        
        // 3. 数据平滑处理
        cleanedData = smoothData(cleanedData);
        
        // 4. 缺失值处理
        cleanedData = handleMissingValues(cleanedData);
        
        // 4. 数据标准化（保存参数）
        DescriptiveStatistics stats = new DescriptiveStatistics();
        cleanedData.forEach(stats::addValue);
        
        this.normalizationMean = stats.getMean();
        this.normalizationStd = stats.getStandardDeviation();
        
        if (this.normalizationStd == 0) {
            this.normalizationStd = 1.0;
        }
        
        List<Double> normalizedData = cleanedData.stream()
                .map(value -> (value - this.normalizationMean) / this.normalizationStd)
                .collect(Collectors.toList());
        
        logger.info("数据标准化完成: 均值={}, 标准差={}", String.format("%.4f", this.normalizationMean), String.format("%.4f", this.normalizationStd));
        logger.info("数据清洗完成: 原始数据量={}, 清洗后数据量={}", data.size(), normalizedData.size());
        
        return normalizedData;
    }
    
    private static List<Double> removeOutliers(List<Double> data) {
        if (data.size() < 10) return data;
        
        DescriptiveStatistics stats = new DescriptiveStatistics();
        data.forEach(stats::addValue);
        
        double mean = stats.getMean();
        double std = stats.getStandardDeviation();
        double lowerBound = mean - 3 * std;
        double upperBound = mean + 3 * std;
        
        List<Double> cleaned = new ArrayList<>();
        int outlierCount = 0;
        
        for (Double value : data) {
            if (value >= lowerBound && value <= upperBound) {
                cleaned.add(value);
            } else {
                outlierCount++;
                if (!cleaned.isEmpty()) {
                    cleaned.add(cleaned.get(cleaned.size() - 1));
                } else {
                    cleaned.add(mean);
                }
            }
        }
        
        if (outlierCount > 0) {
            logger.info("检测到 {} 个异常值并已处理", outlierCount);
        }
        
        return cleaned;
    }
    
    /**
     * 改进的异常值检测：使用IQR方法
     */
    private static List<Double> removeOutliersIQR(List<Double> data) {
        if (data.size() < 10) return data;
        
        DescriptiveStatistics stats = new DescriptiveStatistics();
        data.forEach(stats::addValue);
        
        double q1 = stats.getPercentile(25);
        double q3 = stats.getPercentile(75);
        double iqr = q3 - q1;
        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;
        
        List<Double> cleaned = new ArrayList<>();
        int outlierCount = 0;
        
        for (Double value : data) {
            if (value >= lowerBound && value <= upperBound) {
                cleaned.add(value);
            } else {
                outlierCount++;
                if (!cleaned.isEmpty()) {
                    cleaned.add(cleaned.get(cleaned.size() - 1));
                } else {
                    cleaned.add(stats.getMean());
                }
            }
        }
        
        if (outlierCount > 0) {
            logger.info("IQR方法检测到 {} 个异常值并已处理", outlierCount);
        }
        
        return cleaned;
    }
    
    /**
     * 季节性调整
     */
    private static List<Double> adjustSeasonality(List<Double> data, int seasonLength) {
        if (data.size() < seasonLength * 2) return data;
        
        List<Double> adjusted = new ArrayList<>();
        
        // 只使用前80%的数据计算季节性因子，避免数据泄露
        int trainingSize = (int)(data.size() * 0.8);
        
        // 计算季节性因子（仅使用训练数据）
        Map<Integer, List<Double>> seasonalGroups = new HashMap<>();
        for (int i = 0; i < trainingSize; i++) {
            int season = i % seasonLength;
            seasonalGroups.computeIfAbsent(season, k -> new ArrayList<>()).add(data.get(i));
        }
        
        // 计算每个季节的平均值
        Map<Integer, Double> seasonalMeans = new HashMap<>();
        for (Map.Entry<Integer, List<Double>> entry : seasonalGroups.entrySet()) {
            double mean = entry.getValue().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            seasonalMeans.put(entry.getKey(), mean);
        }
        
        // 计算训练数据的整体平均值
        double overallMean = data.subList(0, trainingSize).stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        // 应用季节性调整（对所有数据）
        for (int i = 0; i < data.size(); i++) {
            int season = i % seasonLength;
            double seasonalMean = seasonalMeans.get(season);
            double adjustment = seasonalMean - overallMean;
            adjusted.add(data.get(i) - adjustment);
        }
        
        logger.info("季节性调整完成，季节长度: {}, 使用前{}个数据点计算因子", seasonLength, trainingSize);
        return adjusted;
    }
    
    private static List<Double> smoothData(List<Double> data) {
        if (data.size() < 3) return data;
        
        List<Double> smoothed = new ArrayList<>();
        int windowSize = Math.min(5, data.size() / 10 + 1);
        
        for (int i = 0; i < data.size(); i++) {
            double sum = 0.0;
            int count = 0;
            
            // 只使用历史数据，避免未来信息泄露
            for (int j = Math.max(0, i - windowSize); j <= i; j++) {
                sum += data.get(j);
                count++;
            }
            
            smoothed.add(sum / count);
        }
        
        logger.info("数据平滑处理完成，窗口大小: {} (仅使用历史数据)", windowSize);
        return smoothed;
    }
    
    private static List<Double> handleMissingValues(List<Double> data) {
        List<Double> cleaned = new ArrayList<>();
        
        for (int i = 0; i < data.size(); i++) {
            Double value = data.get(i);
            
            if (value == null || Double.isNaN(value) || Double.isInfinite(value)) {
                double replacement = 0.0;
                int count = 0;
                
                // 只使用历史数据填充缺失值，避免数据泄露
                if (i > 0) {
                    replacement += cleaned.get(i - 1);
                    count++;
                }
                
                // 移除使用未来数据的部分
                // if (i < data.size() - 1 && data.get(i + 1) != null && 
                //     !Double.isNaN(data.get(i + 1)) && !Double.isInfinite(data.get(i + 1))) {
                //     replacement += data.get(i + 1);
                //     count++;
                // }
                
                if (count > 0) {
                    replacement /= count;
                }
                
                cleaned.add(replacement);
            } else {
                cleaned.add(value);
            }
        }
        
        return cleaned;
    }
    
    private static List<Double> normalizeData(List<Double> data) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        data.forEach(stats::addValue);
        
        double mean = stats.getMean();
        double std = stats.getStandardDeviation();
        
        if (std == 0) return data;
        
        List<Double> normalized = data.stream()
                .map(value -> (value - mean) / std)
                .collect(Collectors.toList());
        
        logger.info("数据标准化完成: 均值={}, 标准差={}", String.format("%.4f", mean), String.format("%.4f", std));
        return normalized;
    }
    
    /**
     * 反标准化数据
     */
    private List<Double> denormalizeData(List<Double> normalizedData) {
        if (normalizationStd == 0) return normalizedData;
        
        List<Double> denormalized = normalizedData.stream()
                .map(value -> value * normalizationStd + normalizationMean)
                .collect(Collectors.toList());
        
        logger.info("数据反标准化完成: 均值={}, 标准差={}", String.format("%.4f", normalizationMean), String.format("%.4f", normalizationStd));
        return denormalized;
    }
    
    private List<Double> preprocessData(List<Double> data) {
        List<Double> processed = new ArrayList<>();
        
        for (int i = 0; i < data.size(); i++) {
            Double value = data.get(i);
            
            if (value == null || Double.isNaN(value)) {
                // 只使用历史数据填充缺失值，避免数据泄露
                if (i > 0) {
                    value = processed.get(i - 1);
                } else {
                    // 如果是第一个值缺失，使用0或历史平均值
                    value = 0.0;
                }
            }
            
            processed.add(value);
        }
        
        return processed;
    }
    
    private List<Double> applyDifferencing(List<Double> data) {
        List<Double> differenced = new ArrayList<>(data);
        
        // 非季节性差分
        for (int i = 0; i < d; i++) {
            List<Double> temp = new ArrayList<>();
            for (int j = 1; j < differenced.size(); j++) {
                temp.add(differenced.get(j) - differenced.get(j - 1));
            }
            differenced = temp;
        }
        
        // 季节性差分
        for (int i = 0; i < D; i++) {
            List<Double> temp = new ArrayList<>();
            for (int j = s; j < differenced.size(); j++) {
                temp.add(differenced.get(j) - differenced.get(j - s));
            }
            differenced = temp;
        }
        
        return differenced;
    }
    
    private void estimateParameters(List<Double> differencedData) {
        int n = differencedData.size();
        
        // 估计AR参数
        if (p > 0) {
            arParams = estimateARParameters(differencedData, p);
        }
        
        // 估计MA参数
        if (q > 0) {
            maParams = estimateMAParameters(differencedData, q);
        }
        
        // 估计季节性AR参数
        if (P > 0) {
            seasonalArParams = estimateARParameters(differencedData, P * s);
        }
        
        // 估计季节性MA参数
        if (Q > 0) {
            seasonalMaParams = estimateMAParameters(differencedData, Q * s);
        }
        
        // 估计截距项
        intercept = differencedData.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    private double[] estimateARParameters(List<Double> data, int order) {
        if (order == 0 || data.size() <= order) return new double[0];
        
        int n = data.size();
        double[] acf = new double[order + 1];
        
        // 计算自相关函数
        for (int k = 0; k <= order; k++) {
            double sum = 0.0;
            for (int i = k; i < n; i++) {
                sum += data.get(i) * data.get(i - k);
            }
            acf[k] = sum / (n - k);
        }
        
        // 使用Yule-Walker方程求解AR参数
        RealMatrix toeplitz = new Array2DRowRealMatrix(order, order);
        double[] rhs = new double[order];
        
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                toeplitz.setEntry(i, j, acf[Math.abs(i - j)]);
            }
            rhs[i] = acf[i + 1];
        }
        
        RealMatrix rhsMatrix = new Array2DRowRealMatrix(rhs.length, 1);
        for (int i = 0; i < rhs.length; i++) {
            rhsMatrix.setEntry(i, 0, rhs[i]);
        }
        
        RealMatrix solution = new LUDecomposition(toeplitz).getSolver().solve(rhsMatrix);
        
        double[] arParams = new double[order];
        for (int i = 0; i < order; i++) {
            arParams[i] = solution.getEntry(i, 0);
        }
        
        return arParams;
    }
    
    private double[] estimateMAParameters(List<Double> data, int order) {
        if (order == 0 || data.size() <= order) return new double[0];
        
        // 简化实现：使用线性回归估计MA参数
        int n = data.size();
        double[][] X = new double[n - order][order];
        double[] y = new double[n - order];
        
        for (int i = order; i < n; i++) {
            for (int j = 0; j < order; j++) {
                X[i - order][j] = data.get(i - j - 1);
            }
            y[i - order] = data.get(i);
        }
        
        SimpleRegression regression = new SimpleRegression();
        for (int i = 0; i < X.length; i++) {
            regression.addData(X[i][0], y[i]);
        }
        
        double[] maParams = new double[order];
        maParams[0] = regression.getSlope();
        for (int i = 1; i < order; i++) {
            maParams[i] = 0.1; // 简化处理
        }
        
        return maParams;
    }
    
    private void calculateResiduals(List<Double> differencedData) {
        residuals.clear();
        
        int maxLag = Math.max(p, q);
        if (maxLag == 0) maxLag = 1;
        
        for (int i = maxLag; i < differencedData.size(); i++) {
            double prediction = intercept;
            
            // AR项
            if (arParams != null) {
                for (int j = 0; j < Math.min(arParams.length, i); j++) {
                    prediction += arParams[j] * differencedData.get(i - j - 1);
                }
            }
            
            // MA项
            if (maParams != null && !residuals.isEmpty()) {
                for (int j = 0; j < Math.min(maParams.length, residuals.size()); j++) {
                    prediction += maParams[j] * residuals.get(residuals.size() - j - 1);
                }
            }
            
            double residual = differencedData.get(i) - prediction;
            residuals.add(residual);
        }
    }
    
    private void evaluateModel() {
        if (residuals.isEmpty()) {
            mse = Double.MAX_VALUE;
            mae = Double.MAX_VALUE;
            r2 = 0.0;
            aic = Double.MAX_VALUE;
            bic = Double.MAX_VALUE;
            logLikelihood = 0.0;
            return;
        }
        
        // 计算MSE
        mse = residuals.stream().mapToDouble(r -> r * r).average().orElse(0.0);
        
        // 计算MAE
        mae = calculateMeanAbsoluteError(residuals);
        
        // 计算R²（基于差分后的数据）
        DescriptiveStatistics residualStats = new DescriptiveStatistics();
        residuals.forEach(residualStats::addValue);
        
        // 使用差分后的数据进行R²计算
        List<Double> differencedData = applyDifferencing(preprocessData(timeSeries));
        double totalSS = differencedData.stream()
                .mapToDouble(x -> Math.pow(x - residualStats.getMean(), 2))
                .sum();
        double residualSS = residuals.stream().mapToDouble(r -> r * r).sum();
        
        r2 = 1.0 - (residualSS / totalSS);
        
        // 计算AIC和BIC
        int n = timeSeries.size();
        int k = (p + q + P + Q + 1); // 参数个数
        logLikelihood = calculateLogLikelihood();
        
        aic = 2 * k - 2 * logLikelihood;
        bic = Math.log(n) * k - 2 * logLikelihood;
    }
    
    private double calculateMeanAbsoluteError(List<Double> errors) {
        return errors.stream().mapToDouble(Math::abs).average().orElse(0.0);
    }
    
    private double calculateLogLikelihood() {
        if (residuals.isEmpty()) return 0.0;
        
        double n = residuals.size();
        double mse = residuals.stream().mapToDouble(r -> r * r).average().orElse(1.0);
        
        return -0.5 * n * Math.log(2 * Math.PI * mse) - 0.5 * n;
    }
} 