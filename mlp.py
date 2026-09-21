import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, random_state=42):
        """
        Inicializa a Multilayer Perceptron (MLP).
        Utiliza 1 camada oculta com ativação ReLU e uma camada de saída com ativação Sigmoid.
        """
        np.random.seed(random_state)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Inicialização He (adequada para ReLU)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Inicialização Xavier/Glorot (adequada para Sigmoid)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def relu(self, Z):
        return np.maximum(0, Z)
        
    def relu_deriv(self, Z):
        return (Z > 0).astype(float)
        
    def sigmoid(self, Z):
        # Evita overflow
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))
        
    def forward(self, X):
        """
        Passo forward da rede.
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
        
    def backward(self, X, y):
        """
        Passo backward (retropropagação) da rede para atualização dos pesos.
        """
        m = X.shape[0]
        
        # Gradiente na camada de saída (Cross-Entropy com Sigmoid tem gradiente A2 - y)
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Gradiente na camada oculta
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Atualização dos pesos e viéses
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def compute_loss(self, y_true, y_pred):
        """
        Calcula a perda (Binary Cross Entropy).
        """
        m = y_true.shape[0]
        epsilon = 1e-15 # Para evitar log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def predict(self, X):
        """
        Realiza a predição retornando 0 ou 1.
        """
        predictions = self.forward(X)
        return (predictions >= 0.5).astype(int)
        
    def accuracy(self, y_true, y_pred):
        """
        Calcula a acurácia.
        """
        return np.mean(y_true == y_pred)
