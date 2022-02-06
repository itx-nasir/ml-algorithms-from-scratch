# DBSCAN Clustering Implementation

A Python implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. This implementation provides a flexible and efficient way to perform density-based clustering on various datasets.

## Features

- Custom DBSCAN implementation with NumPy optimization
- Interactive visualization of clustering results
- Support for different parameter configurations
- Outlier detection capabilities
- Standardized data preprocessing

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/dbscan-implementation.git
cd dbscan-implementation
```

### 2. Create Virtual Environment
For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
Place your dataset in CSV format in the project root directory. Example format:
```csv
x,y
1.2,2.3
2.1,3.4
...
```

### 5. Run DBSCAN
```bash
python dbscan.py
```
