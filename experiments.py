import numpy as np
import pandas as pd
from copy import deepcopy
from mlp import MLP
from data import load_data

def run_experiment(X_train, y_train, X_val, y_val, hidden_size, learning_rate, max_epochs=2000, patience=200):
    """
    Treina a MLP com early stopping e registra as métricas a cada época.
    """
    input_size = X_train.shape[1]
    output_size = 1
    
    model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=learning_rate)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Salvar o melhor modelo
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Forward pass (Treino)
        train_preds_prob = model.forward(X_train)
        train_loss = model.compute_loss(y_train, train_preds_prob)
        train_preds_class = (train_preds_prob >= 0.5).astype(int)
        train_acc = model.accuracy(y_train, train_preds_class)
        
        # Backward pass (usa as ativações do treino)
        model.backward(X_train, y_train)
        
        # Forward pass (Validação)
        val_preds_prob = model.forward(X_val)
        val_loss = model.compute_loss(y_val, val_preds_prob)
        val_preds_class = (val_preds_prob >= 0.5).astype(int)
        val_acc = model.accuracy(y_val, val_preds_class)
        
        # Registrando histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {
                'W1': np.copy(model.W1), 'b1': np.copy(model.b1),
                'W2': np.copy(model.W2), 'b2': np.copy(model.b2)
            }
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            # Restaurando os melhores pesos
            model.W1, model.b1 = best_model_state['W1'], best_model_state['b1']
            model.W2, model.b2 = best_model_state['W2'], best_model_state['b2']
            break
            
    # Se não parou antecipadamente, restaura o melhor estado de qualquer forma
    if patience_counter < patience and best_model_state is not None:
        model.W1, model.b1 = best_model_state['W1'], best_model_state['b1']
        model.W2, model.b2 = best_model_state['W2'], best_model_state['b2']
        
    return model, history, best_epoch

def run_all_experiments():
    """
    Executa a grade de experimentos exigida no exercício.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_samples=1000)
    
    hidden_neurons_list = [2, 5, 10, 20]
    learning_rates_list = [0.01, 0.1, 0.5]
    
    results = []
    models = {}
    histories = {}
    
    for hn in hidden_neurons_list:
        for lr in learning_rates_list:
            print(f"Treinando modelo - Neurônios: {hn}, LR: {lr}")
            
            model, history, best_epoch = run_experiment(
                X_train, y_train, X_val, y_val, 
                hidden_size=hn, learning_rate=lr, 
                max_epochs=2000, patience=200
            )
            
            # Recalculando métricas finais com os melhores pesos (para garantir)
            train_preds = model.predict(X_train)
            train_acc = model.accuracy(y_train, train_preds)
            train_loss = model.compute_loss(y_train, model.forward(X_train))
            
            val_preds = model.predict(X_val)
            val_acc = model.accuracy(y_val, val_preds)
            val_loss = model.compute_loss(y_val, model.forward(X_val))
            
            epochs_run = len(history['train_loss'])
            
            results.append({
                'Neurons': hn,
                'LR': lr,
                'Total Epochs Executed': epochs_run,
                'Best Val Epoch': best_epoch,
                'Train Loss': train_loss,
                'Val Loss': val_loss,
                'Train Acc': train_acc,
                'Val Acc': val_acc
            })
            
            # Salvando para acesso posterior nos gráficos
            config_name = f"{hn}_neurons_{lr}_lr"
            models[config_name] = model
            histories[config_name] = history
            
    results_df = pd.DataFrame(results)
    return results_df, models, histories, (X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    df, _, _, _ = run_all_experiments()
    print("\nResultados finais:")
    print(df)
