import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # To store activations and gradients for visualization
        self.a1 = None  # Activation from hidden layer
        self.z1 = None  # Input to activation function at hidden layer
        self.a2 = None  # Output predictions
        self.z2 = None  # Input to activation function at output layer
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def activation(self, z):
        if self.activation_fn == 'tanh':
            return np.tanh(z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_fn == 'relu':
            return np.where(z > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # Forward pass, apply layers to input X
        self.z1 = np.dot(X, self.W1) + self.b1  # Input to activation function
        self.a1 = self.activation(self.z1)      # Activation from hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Input to output activation
        self.a2 = 1 / (1 + np.exp(-self.z2))    # Sigmoid activation for binary classification
        return self.a2

    def backward(self, X, y):
        m = y.shape[0]

        # Compute gradients using chain rule
        dz2 = self.a2 - y  # Derivative of loss w.r.t z2
        self.dW2 = np.dot(self.a1.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)  # Derivative w.r.t z1
        self.dW1 = np.dot(X.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights with gradient descent
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input data
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, hidden_dim):
    try:
        ax_hidden.cla()
        ax_input.cla()
        ax_gradient.cla()

        # Perform training steps by calling forward and backward function
        for _ in range(10):
            # Perform a training step
            mlp.forward(X)
            mlp.backward(X, y)

        # Plot hidden features
        hidden_features = mlp.a1  # Activations from the hidden layer

        # Adjust plotting based on the dimensionality of the hidden layer
        if hidden_dim >= 3:
            # 3D plot
            ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                              c=y.ravel(), cmap='bwr', alpha=0.7)
            ax_hidden.set_xlabel('Neuron 1 Activation')
            ax_hidden.set_ylabel('Neuron 2 Activation')
            ax_hidden.set_zlabel('Neuron 3 Activation')
        elif hidden_dim == 2:
            # 2D plot
            ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1],
                              c=y.ravel(), cmap='bwr', alpha=0.7)
            ax_hidden.set_xlabel('Neuron 1 Activation')
            ax_hidden.set_ylabel('Neuron 2 Activation')
        elif hidden_dim == 1:
            # 1D plot
            ax_hidden.scatter(hidden_features[:, 0], np.zeros_like(hidden_features[:, 0]),
                              c=y.ravel(), cmap='bwr', alpha=0.7)
            ax_hidden.set_xlabel('Neuron 1 Activation')
        else:
            raise ValueError("Hidden layer must have at least one neuron")

        ax_hidden.set_title('Hidden Layer Feature Space')

        # Hyperplane visualization in the hidden space
        w2 = mlp.W2.flatten()
        b2 = mlp.b2.flatten()
        if hidden_dim >= 2:
            xx, yy = np.meshgrid(
                np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 10),
                np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 10)
            )
            if hidden_dim >= 3 and w2[2] != 0:
                z = (-w2[0]*xx - w2[1]*yy - b2) / w2[2]
                ax_hidden.plot_surface(xx, yy, z, alpha=0.3)
            elif hidden_dim == 2 and w2[1] != 0:
                z = (-w2[0]*xx - b2) / w2[1]
                ax_hidden.contour(xx, yy, z, levels=[0], colors='k')
        else:
            # For hidden_dim == 1, we can't plot a hyperplane
            pass

        # Distorted input space transformed by the hidden layer
        transformed_X = mlp.a1
        if hidden_dim >= 2:
            ax_input.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
            ax_input.set_xlabel('Transformed Feature 1')
            ax_input.set_ylabel('Transformed Feature 2')
        elif hidden_dim == 1:
            ax_input.scatter(transformed_X[:, 0], np.zeros_like(transformed_X[:, 0]), c=y.ravel(), cmap='bwr', alpha=0.7)
            ax_input.set_xlabel('Transformed Feature 1')
        ax_input.set_title('Distorted Input Space')

        # Plot input layer decision boundary
        h = 0.05  # Mesh step size
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = mlp.forward(grid_points)
        Z = Z.reshape(xx.shape)
        ax_input.contourf(xx, yy, Z > 0.5, alpha=0.2, cmap='bwr')
        ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap='bwr')
        ax_input.set_xlabel('Feature 1')
        ax_input.set_ylabel('Feature 2')
        ax_input.set_title('Decision Boundary in Input Space')

        # Visualize features and gradients as circles and edges
        ax_gradient.axis('off')
        node_positions = {
            'input': [(0, i) for i in range(X.shape[1])],
            'hidden': [(1, i) for i in range(hidden_dim)],
            'output': [(2, 0)]
        }

        # Plot nodes
        for layer, positions in node_positions.items():
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            ax_gradient.scatter(x_coords, y_coords, s=500, label=layer, zorder=5)

        # Plot edges with gradient magnitudes
        # Input to Hidden
        for i in range(X.shape[1]):  # Input neurons
            for j in range(hidden_dim):  # Hidden neurons
                x_coords = [node_positions['input'][i][0], node_positions['hidden'][j][0]]
                y_coords = [node_positions['input'][i][1], node_positions['hidden'][j][1]]
                weight_grad = abs(mlp.dW1[i, j])
                ax_gradient.plot(x_coords, y_coords, 'k-', lw=weight_grad * 1000, alpha=0.5)

        # Hidden to Output
        for i in range(hidden_dim):  # Hidden neurons
            x_coords = [node_positions['hidden'][i][0], node_positions['output'][0][0]]
            y_coords = [node_positions['hidden'][i][1], node_positions['output'][0][1]]
            weight_grad = abs(mlp.dW2[i, 0])
            ax_gradient.plot(x_coords, y_coords, 'k-', lw=weight_grad * 1000, alpha=0.5)

        ax_gradient.set_title('Gradient Visualization')
        ax_gradient.legend()

    except Exception as e:
        print(f"Exception during update at frame {frame}: {e}")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    hidden_dim = 3  # Adjust this value as needed
    mlp = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))

    # Prepare axes based on hidden_dim
    if hidden_dim >= 3:
        ax_hidden = fig.add_subplot(131, projection='3d')
    else:
        ax_hidden = fig.add_subplot(131)

    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    frames = max(step_num // 10, 1)  # Ensure at least one frame
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input,
                                     ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y, hidden_dim=hidden_dim),
                        frames=frames, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
