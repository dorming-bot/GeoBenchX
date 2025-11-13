# 依赖安装指南

## 问题诊断

如果看到 `ModuleNotFoundError: No module named 'geopandas'` 或其他依赖错误，说明当前 Python 环境缺少必要的依赖包。

## 快速解决方案

### 方法1：在 Jupyter Notebook 中自动安装（推荐）

我已经更新了 `notebooks/Benchmarking.ipynb`，第一个单元格包含了自动安装代码。只需：

1. **重启 Jupyter 内核**（非常重要！）
   - 点击 Kernel → Restart
   
2. **运行第一个单元格**
   ```python
   # 这个单元格会自动安装所有依赖
   !pip install -q geopandas rasterio folium shapely --quiet
   ```

3. **运行第二个单元格**
   这会添加项目路径到 Python

4. **运行后续单元格**

### 方法2：使用 conda 安装（最推荐，因为 conda 能更好地处理地理空间包依赖）

如果您使用的是 Anaconda 或 Miniconda：

```bash
# 激活您的 conda 环境
conda activate geo_env  # 或者您的环境名

# 安装地理空间包（使用 conda-forge）
conda install -c conda-forge geopandas rasterio folium shapely -y

# 安装其他依赖
pip install -r requirements.txt
```

### 方法3：手动逐个安装

在终端中运行：

```bash
# 1. 激活您的 Python 环境
conda activate geo_env  # 或您的环境名

# 2. 安装地理空间包
conda install -c conda-forge geopandas rasterio folium shapely -y

# 3. 安装所有其他依赖
pip install openai langchain_openai langchain_google_genai langchain_anthropic langgraph
pip install pandas numpy matplotlib scipy
pip install statsmodels scikit-learn
pip install wbgapi plotly networkx pydantic
pip install google-generativeai anthropic python-dotenv requests tiktoken
pip install nbformat streamlit kaleido contextily overpy
```

### 方法4：检查 Jupyter 使用的 Python 环境

在 Jupyter Notebook 中运行：

```python
import sys
print(f"Python 路径: {sys.executable}")
print(f"Python 版本: {sys.version}")
```

确保这个 Python 环境已安装所有依赖。

## 常见问题

### Q1: 为什么 requirements.txt 包含这些包，但还是安装失败？

A: **Jupyter 内核可能使用的是不同的 Python 环境**。即使您在终端中安装了所有依赖，Jupyter 也可能使用另一个环境。

**解决方案**：
1. 检查 Jupyter 使用的 Python 环境：
   ```python
   import sys
   print(sys.executable)
   ```
2. 在该环境中安装依赖：
   ```bash
   # 使用该环境的 pip
   "C:\full\path\to\python.exe" -m pip install geopandas
   ```

### Q2: 在 Windows 上安装 geopandas 失败

A: geopandas 依赖 GDAL，在 Windows 上比较麻烦。

**解决方案**：
1. 使用 conda（最简单）：
   ```bash
   conda install -c conda-forge geopandas -y
   ```

2. 或使用预编译的 wheel：
   ```bash
   pip install geopandas --no-deps
   pip install pyogrio  # 新的后端
   ```

### Q3: 依赖已安装但 Jupyter 仍找不到

A: 可能是 Jupyter 内核缓存问题。

**解决方案**：
1. 重启 Jupyter 内核
2. 重启 Jupyter Notebook
3. 如果还不行，重新安装 ipykernel：
   ```bash
   pip install --force-reinstall ipykernel
   ```

## 验证安装

安装完成后，在 Jupyter Notebook 中运行：

```python
import geopandas as gpd
import rasterio
import folium
import shapely
print("所有地理空间包导入成功！")

import openai
import langchain_openai
import langchain_google_genai
import langchain_anthropic
print("所有 AI 相关包导入成功！")
```

## 推荐流程

对于新用户，推荐以下步骤：

1. **使用 conda 环境**：
   ```bash
   conda create -n geobenchx python=3.10
   conda activate geobenchx
   ```

2. **安装地理空间包**：
   ```bash
   conda install -c conda-forge geopandas rasterio folium shapely -y
   ```

3. **安装其他依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **启动 Jupyter**：
   ```bash
   jupyter notebook
   ```

这样通常可以避免大部分依赖问题。

