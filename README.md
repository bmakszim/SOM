# Self Organizing Map
## Makszim Balázs Imre - ENSGO3

## Introduction

A **Self-Organizing Map (SOM)**, also known as a Kohonen network, is a type of artificial neural network designed for unsupervised learning. SOMs are primarily used for clustering and visualizing high-dimensional data in a low-dimensional (typically 2D) map. They are widely applied in fields like data mining, pattern recognition, and exploratory data analysis.

## Key Characteristics of SOMs:
1. **Unsupervised Learning**: They do not require labeled data; instead, they organize and cluster input data based on inherent patterns or similarities.

2. **Topology Preservation**: SOMs maintain the spatial relationship of the data. Input data points that are similar are mapped to nearby nodes in the SOM, preserving their topological structure.

3. **Dimensionality Reduction**: They reduce high-dimensional data into a two-dimensional map, making complex data easier to interpret and visualize.

## Structure of a SOM:

1. **Input Layer**: Consists of feature vectors from the dataset.

2. **Output Layer (Map)**: A grid of neurons (nodes), typically arranged in a 2D lattice. Each node has an associated weight vector of the same dimension as the input vectors.

## How SOMs Work:

1. **Initialization**: Each node in the output layer is initialized with a random weight vector.

2. **Training**:
    - **Input Data Presentation**: A sample input vector is presented to the network.
    - **Best Matching Unit (BMU)**: The node with the weight vector closest to the input vector (in terms of Euclidean distance) is identified.
    - **Weight Update**: The BMU and its neighbors update their weights to become more like the input vector. The degree of adjustment is controlled by a *learning rate* and *neighborhood function*, which decrease over time.

3. **Convergence**: After multiple iterations, the nodes organize themselves so that similar input vectors are mapped to adjacent nodes.

---

_*Learning Rate Decay_: Ensures the SOM converges over time.

_*Neighborhood Function_: Controls how far the influence of a BMU extends to its neighbors.

---

## Applications of SOMs:

1. **Clustering**: Grouping similar data points, such as customer segmentation.

2. **Data Visualization**: Representing high-dimensional data on a 2D map (e.g., in bioinformatics).

3. **Anomaly Detection**: Identifying data points that don’t fit into existing clusters.

4. **Feature Reduction**: Simplifying datasets for further processing.

## Implementation

### Overview

This implementation segments a set of synthetic customer data into clusters using a Self-Organizing Map (SOM).

### Step 1: Generate Synthetic Data

```python
np.random.seed(42)
num_customers = 500
data = np.column_stack((
    np.random.poisson(lam=5, size=num_customers),         # Purchase frequency
    np.random.normal(loc=100, scale=20, size=num_customers),  # Average spending
    np.random.randint(1, 5, size=num_customers),          # Category focus (1-4)
    np.random.randint(18, 65, size=num_customers)         # Age
))
```
**Explanation**:
This code generates a set of random input data.

**Inputs**: Customer behavior: `purchase_frequency`, `avg_spending`, `category_focus`, `age`.

**Outputs**: A matrix (`data`) of shape `(500, 4`) representing 500 customers with 4 features.

### Step 2: Normalize the Data

```python
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

```
**Explanation**: Normalizes all features to the range [0,1], ensuring that all dimensions contribute equally to the Euclidean distance calculation.

**Formula**:
```math
X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
```

### Step 3: Initialize SOM Parameters

```python
som_grid_size = (5, 5)  # SOM grid size
weights = np.random.rand(som_grid_size[0], som_grid_size[1], data_normalized.shape[1])  # Randomly initialized weights
sigma = 2.0  # Neighborhood radius
learning_rate = 0.5
num_iterations = 1000
```

**Explanation**:

- **Grid size:** 5x5 SOM grid.
- **Weights:** Randomly initialized weight vectors for each SOM node.
- **Sigma:** Initial neighborhood radius.
- **Learning rate:** Controls weight updates.

### Step 4: Train the SOM
```python
for iteration in range(num_iterations):
    # Randomly select a sample
    sample = data_normalized[np.random.randint(0, data_normalized.shape[0])]

    # Find Best Matching Unit (BMU)
    distances = np.array([[euclidean_distance(sample, weights[i, j])
                           for j in range(som_grid_size[1])] for i in range(som_grid_size[0])])
    bmu_index = np.unravel_index(np.argmin(distances), distances.shape)

    # Update weights of BMU and its neighbors
    for i in range(som_grid_size[0]):
        for j in range(som_grid_size[1]):
            distance_to_bmu = euclidean_distance(np.array([i, j]), np.array(bmu_index))
            if distance_to_bmu < sigma:
                influence = np.exp(-distance_to_bmu**2 / (2 * sigma**2))
                weights[i, j] += learning_rate * influence * (sample - weights[i, j])

    # Decay learning rate and neighborhood size
    learning_rate *= 0.99
    sigma *= 0.99

```

**Explanation**:
1. **Random Sample**: A data point is randomly selected.
2. **Best Matching Unit (BMU)**:
The SOM node (grid point) whose weight vector is closest to the sample is selected.
3. **Formula (Euclidean distance)**:

```math
d_{ij} = \| x - w_{ij} \|_2 = \sqrt{\sum_k (x_k - w_{ij,k})^2}
```