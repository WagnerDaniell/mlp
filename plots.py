import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_curves(history, config_name, best_epoch, save_dir="plots"):
    """
    Plota as curvas de perda de treino e validação ao longo das épocas.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    plt.plot(history['val_loss'], label='Val Loss', alpha=0.8)
    
    # Marca o ponto de early stopping
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch (ES): {best_epoch}')
    
    plt.title(f'Learning Curves - {config_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f'loss_curve_{config_name}.png'))
    plt.close()

def plot_decision_boundary(model, X, y, title, save_path):
    """
    Plota a fronteira de decisão aprendida pela MLP.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predições para o meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap=plt.cm.RdYlBu)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.savefig(save_path)
    plt.close()

def plot_dataset(X, y, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
