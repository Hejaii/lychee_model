@echo off
chcp 65001 >nul
echo === 荔枝预测模型构建脚本 ===

REM 检查Java版本
echo 检查Java版本...
java -version
if %errorlevel% neq 0 (
    echo 错误: 未找到Java环境，请先安装Java 8或更高版本
    pause
    exit /b 1
)

REM 检查Maven版本
echo 检查Maven版本...
mvn -version
if %errorlevel% neq 0 (
    echo 错误: 未找到Maven环境，请先安装Maven
    pause
    exit /b 1
)

REM 清理之前的构建
echo 清理之前的构建...
mvn clean

REM 编译项目
echo 编译项目...
mvn compile
if %errorlevel% neq 0 (
    echo 错误: 项目编译失败
    pause
    exit /b 1
)

REM 运行测试
echo 运行测试...
mvn test
if %errorlevel% neq 0 (
    echo 警告: 测试失败，但继续构建
)

REM 打包项目
echo 打包项目...
mvn package
if %errorlevel% neq 0 (
    echo 错误: 项目打包失败
    pause
    exit /b 1
)

echo === 构建完成 ===
echo 可执行JAR文件位置: target\lychee-prediction-1.0.0-jar-with-dependencies.jar
echo.
echo 运行项目:
echo java -jar target\lychee-prediction-1.0.0-jar-with-dependencies.jar
echo.
echo 或者直接运行主类:
echo mvn exec:java -Dexec.mainClass="com.litchi.prediction.IntegratedLitchiPredictionModel"
pause 