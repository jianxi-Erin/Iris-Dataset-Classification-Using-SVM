
# **Iris Dataset Classification Using SVM**  

## **项目简介**  
本项目基于经典的 Iris 数据集，利用支持向量机（SVM）算法构建分类模型，旨在对鸢尾花的三种不同类别进行分类。通过本项目，展示了机器学习模型从数据预处理、特征分析、模型训练到模型评估的完整流程，并为未来的模型优化和预测提供了基础。  

---

## **数据集说明**  
- **来源**：Iris 数据集是一个经典的机器学习数据集，包含 150 条记录，分为 3 类，每类 50 条记录。  
- **特征**：  
  - `sepal_length` (萼片长度)  
  - `sepal_width` (萼片宽度)  
  - `petal_length` (花瓣长度)  
  - `petal_width` (花瓣宽度)  
- **目标**：预测鸢尾花属于以下三类中的哪一类：  
  - Iris-setosa  
  - Iris-versicolor  
  - Iris-virginica  

---

## **项目结构**  
```
├── data_sets/                 # 数据文件夹
│   └── IRIS.csv               # Iris 数据集
├── models/                    # 模型文件夹
│   └── svm_model_0.9666.pkl   # 已保存的 SVM 模型
├── notebooks/                 # Jupyter 笔记本
│   └── main.ipynb # 数据分析与训练代码
├── doc/
│   └── 使用SVM进行鸢尾花的分类.docx
├── README.md                  # 项目说明文件
└── requirements.txt           # 依赖库

```  

---

## **安装和运行**  

### **环境要求**  
请确保系统已安装以下工具和环境：  
- Python 3.7 或更高版本  
- Jupyter Notebook 或支持 `.ipynb` 文件的 IDE  

### **安装依赖**  
运行以下命令安装所需依赖：  
```bash
pip install -r requirements.txt
```  

### **运行项目**  
1. **加载数据**：将 `iris.csv` 文件放入 `data/` 文件夹中。  
2. **训练模型**：打开 `notebooks/iris_classification.ipynb` 文件，运行所有单元格以训练 SVM 模型。  
3. **保存模型**：训练完成后，模型会保存到 `models/svm_model.pkl` 文件中。  
4. **预测**：使用保存的模型对新数据进行分类预测。  

---

## **主要功能**  

### **1. 数据分析**  
- 数据的加载与探索性数据分析（EDA）  
- 数据分布的可视化，包括箱线图和散点图  

### **2. 模型训练**  
- 使用支持向量机（SVM）进行分类任务  
- 使用 `train_test_split` 划分数据集（训练集 80%，测试集 20%）  
- 对模型进行训练，并使用准确率评估  

### **3. 模型预测**  
- 使用保存的 SVM 模型对测试集和新数据进行分类预测  

### **4. 结果评估**  
- 通过混淆矩阵、分类报告等工具评估模型性能  
- 本项目的 SVM 模型在测试集上的准确率达到了 **96.67%**  

---

## **文件说明**  

| 文件名                  | 描述                                              |  
|-------------------------|---------------------------------------------------|  
| `data/iris.csv`         | Iris 数据集                                       |  
| `notebooks/iris_classification.ipynb` | 项目的 Jupyter Notebook 文件，包含完整代码和分析 |  
| `models/svm_model.pkl`  | 训练后的 SVM 模型文件                            |  
| `requirements.txt`      | Python 依赖库列表                                |  

---

## **模型使用说明**  

### **加载模型**  
使用保存的 SVM 模型直接进行预测：  
```python
import joblib
import pandas as pd

# 加载模型
model = joblib.load('models/svm_model.pkl')

# 加载新数据
new_data = pd.DataFrame({
    'sepal_length': [5.1, 6.7],
    'sepal_width': [3.5, 3.0],
    'petal_length': [1.4, 5.2],
    'petal_width': [0.2, 2.3]
})

# 进行预测
predictions = model.predict(new_data)
print("预测结果：", predictions)
```

---

## **未来工作展望**  
1. **模型优化**：  
   - 尝试其他分类算法（如随机森林、XGBoost）并比较性能。  
   - 进行超参数调优（如调整 SVM 的核函数、C 值、gamma 值）。  

2. **数据扩展**：  
   - 引入更多样本或使用增强数据来提高模型的泛化能力。  

3. **部署模型**：  
   - 将模型部署到 Web 应用程序（如 Flask 或 Django）中，提供在线分类服务。  

4. **自动化管道**：  
   - 使用工具（如 `MLflow` 或 `Airflow`）创建完整的机器学习工作流。  

---

## **联系信息**  
如有任何问题或建议，请联系：  
- **开发者**：简希 (Jian Xi)  
- **邮箱**：jianxi@example.com  
- **GitHub**：[DigitRec 项目主页](https://github.com/jianxi-Erin/DigitRec)  

--- 

希望本项目对您有所帮助！  