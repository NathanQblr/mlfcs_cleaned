import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
import joblib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def plot_colored_quaternary(data_beads):
    """Plot data in a 3D quaternary plot with colored regions for dominant probabilities.

    Args:
        data_beads: pd.DataFrame with predicted probabilities for 4 models.

    Returns:
        plt.figure
    """
    # Load predicted probabilities for each model
    prob_data = data_beads[['Proba MB', 'Proba FBM', 'Proba CTRW', 'Proba RWF']]

    # Normalize the probabilities to ensure they sum to 1 for each point
    prob_data = prob_data.div(prob_data.sum(axis=1), axis=0)

    # Define vertices of the tetrahedron in 3D space
    vertices = np.array([
        [1, 0, 0],               # Vertex 1 (Proba MB)
        [0, 1, 0],               # Vertex 2 (Proba FBM)
        [0, 0, 1],               # Vertex 3 (Proba CTRW)
        [np.sqrt(2), np.sqrt(2), np.sqrt(2)]  # Vertex 4 (Proba RWF)
    ])

    # Convert probabilities to 3D points using a linear combination of the vertices
    coords = prob_data.dot(vertices)

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define color for each region where one probability is dominant
    palette4 = ['#bc8977', '#6a73a4', '#719b78', '#ffb347']  # Colors for 4 classes

    # Create transparent polygons to represent the dominant regions
    region_faces = [
        [vertices[0], vertices[1], vertices[3]],  # Region dominated by Proba MB
        [vertices[1], vertices[2], vertices[3]],  # Region dominated by Proba FBM
        [vertices[2], vertices[0], vertices[3]],  # Region dominated by Proba CTRW
        [vertices[0], vertices[1], vertices[2]]   # Base of the tetrahedron (Proba RWF)
    ]

    for i, face in enumerate(region_faces):
        poly = Poly3DCollection([face], color=palette4[i], alpha=0.2)
        ax.add_collection3d(poly)

    # Plot the data points and connect them by 'file name' and 'window number'
    glycerol = np.unique(data_beads['Glycerol'])
    for gly in glycerol:
        subset = data_beads[data_beads['Glycerol'] == gly].sort_values(by='window number')

        # Normalize 'file name' for color coding
        name_encoder = LabelEncoder()
        subset['file name'] = name_encoder.fit_transform(subset['file name'])

        # Plot each file's trajectory
        for name in subset['file name'].unique():
            file_subset = subset[subset['file name'] == name]
            ax.plot(
                coords.loc[file_subset.index, 0], coords.loc[file_subset.index, 1], coords.loc[file_subset.index, 2],
                marker='o', label=f'File {name}' if gly == glycerol[0] else "", alpha=0.7
            )

    # Plot tetrahedron edges for reference
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]],
                color='black', linewidth=1
            )

    # Set labels and title
    ax.set_xlabel('X (Proba MB)')
    ax.set_ylabel('Y (Proba FBM)')
    ax.set_zlabel('Z (Proba CTRW)')
    ax.set_title('3D Quaternary Plot with Colored Regions')

    plt.tight_layout()
    return fig




def plot_quaternary(data_beads):
    """Plot data from classification in a 3D quaternary plot for 4 candidate models.

    Args:
        data_beads: pd.DataFrame with prediction of models.

    Returns:
        plt.figure
    """
    # Load predicted probabilities for each model
    prob_data = data_beads[['Proba MB', 'Proba FBM', 'Proba CTRW', 'Proba RWF']]

    # Normalize the probabilities to ensure they sum to 1 for each point
    prob_data = prob_data.div(prob_data.sum(axis=1), axis=0)

    # Define vertices of the tetrahedron in 3D space
    vertices = np.array([
        [1, 0, 0],               # Vertex 1 (Proba MB)
        [0, 1, 0],               # Vertex 2 (Proba FBM)
        [0, 0, 1],               # Vertex 3 (Proba CTRW)
        [0, 0, 0]#[1/3, 1/3, 1/3 * np.sqrt(2)]  # Vertex 4 (Proba RWF) to ensure it's at the top of the tetrahedron
    ])

    # Convert probabilities to 3D points using a linear combination of the vertices
    coords = prob_data.dot(vertices)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    scatter = ax.scatter(
        coords.iloc[:, 0], coords.iloc[:, 1], coords.iloc[:, 2],
        c=data_beads['file name'], cmap='viridis', s=15
    )

    # Set the labels
    ax.set_xlabel('X (Proba MB)')
    ax.set_ylabel('Y (Proba FBM)')
    ax.set_zlabel('Z (Proba CTRW)')

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('File Name (encoded)')

    # Set equal aspect ratio for the plot
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Plot tetrahedron edges for reference
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]],
                color='black', linewidth=1
            )

    ax.set_title('3D Quaternary Plot of Model Probabilities')
    plt.tight_layout()

    return fig




data_beads = pd.read_pickle('../dat/beads_4models.pk')
fig = plot_colored_quaternary(data_beads)
fig.savefig('../fig_4models/beads_model_pred_traj_quaternary.png',dpi=600)
plt.show()