import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Generate synthetic customer data
np.random.seed(42)

# Customer features: [purchase_frequency, avg_spending, category_focus, age]
num_customers = 500
data = np.column_stack((
    np.random.poisson(lam=5, size=num_customers),         # Purchase frequency
    np.random.normal(loc=100, scale=20, size=num_customers),  # Average spending
    np.random.randint(1, 5, size=num_customers),          # Category focus (1-4)
    np.random.randint(18, 65, size=num_customers)         # Age
))

# Step 2: Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Step 3: Implement SOM parameters
som_grid_size = (5, 5)  # SOM grid size
weights = np.random.rand(som_grid_size[0], som_grid_size[1], data_normalized.shape[1])  # Randomly initialized weights
sigma = 2.0  # Neighborhood radius
learning_rate = 0.5
num_iterations = 1000

# Helper function: Calculate Euclidean distance
def euclidean_distance(x, y):
    return np.linalg.norm(x - y, axis=-1)

# Step 4: Train the SOM
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

# Step 5: Assign each customer to a cluster
customer_clusters = np.array([
    np.unravel_index(np.argmin([[euclidean_distance(customer, weights[i, j])
                                 for j in range(som_grid_size[1])] for i in range(som_grid_size[0])]), weights.shape[:2])
    for customer in data_normalized
])

# Step 6: Visualize the clusters
unique_clusters = np.unique(customer_clusters, axis=0)


# Step 7: Map clusters to unique colors for visualization
# Assign a unique color to each cluster
unique_cluster_indices = {tuple(cluster): idx for idx, cluster in enumerate(unique_clusters)}
cluster_colors = np.array([unique_cluster_indices[tuple(c)] for c in customer_clusters])

# Normalize cluster indices for color mapping
color_map = plt.cm.get_cmap("tab10", len(unique_clusters))

# Visualize the SOM with cluster colors
plt.figure(figsize=(10, 10))
for i, (x, y) in enumerate(customer_clusters):
    plt.scatter(
        x + np.random.uniform(-0.4, 0.4),
        y + np.random.uniform(-0.4, 0.4),
        color=color_map(cluster_colors[i]),
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5
    )

for idx, cluster in enumerate(unique_clusters):
    plt.text(cluster[0], cluster[1], str(idx), ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.title("Customer Segments with Cluster Colors in SOM Grid")
plt.xlabel("SOM Grid X-axis")
plt.ylabel("SOM Grid Y-axis")
plt.grid()
plt.show()