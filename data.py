from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(n_samples=1000, noise=0.2, random_state=42, test_size=0.2, val_size=0.25):
    """
    Gera o dataset Two Moons e o divide em conjuntos de treino, validação e teste.
    
    :param n_samples: Total de amostras.
    :param noise: Ruído para o dataset.
    :param random_state: Semente para reprodutibilidade.
    :param test_size: Proporção do conjunto de teste (em relação ao total).
    :param val_size: Proporção do conjunto de validação (em relação ao treino resultante).
                     Com test_size=0.2 e val_size=0.25, teremos 60% treino, 20% validação e 20% teste.
    :return: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Gerando os dados
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Redimensionando y para (n_samples, 1) para facilitar os cálculos matriciais da MLP
    y = y.reshape(-1, 1)
    
    # Dividindo em treino (80%) e teste (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Dividindo o treino_val em treino (75% de 80% = 60%) e validação (25% de 80% = 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test
