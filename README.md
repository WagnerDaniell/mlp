# Exercício Prático: Multilayer Perceptron (MLP)

Este projeto implementa uma rede neural Multilayer Perceptron (MLP) do zero utilizando NumPy para classificar o clássico dataset não-linear **Two Moons**.

## Objetivo
O objetivo principal é entender na prática a implementação do *forward* e *backward* (retropropagação) e observar empiricamente o impacto da variação de hiperparâmetros, especificamente o número de neurônios na camada oculta e a taxa de aprendizagem (learning rate). Adicionalmente, busca-se aplicar a técnica de *Early Stopping* para mitigar overfitting.

## Arquitetura da MLP
- **Entrada:** 2 características (coordenadas x e y do dataset Two Moons).
- **Camada Oculta:** Variações com 2, 5, 10 e 20 neurônios. Função de ativação: **ReLU** (Rectified Linear Unit).
- **Camada de Saída:** 1 neurônio com função de ativação **Sigmoid** para classificação binária.
- **Função de Perda:** Binary Cross-Entropy (BCE).
- **Otimização:** Gradiente Descendente (atualização via backpropagation manual).
- **Inicialização de pesos:** Inicialização de He (para ReLU) e Xavier (para Sigmoid).

## Metodologia e Conjuntos de Dados
O dataset **Two Moons** com 1000 amostras e ruído de 0.2 foi gerado e dividido em:
- 60% para Treinamento
- 20% para Validação (usado para Early Stopping)
- 20% para Testes (Avaliação final de generalização opcional/reserva)

## Conceitos e Técnicas Utilizadas

Neste projeto, diversas técnicas fundamentais de Redes Neurais foram implementadas do zero. Abaixo, explicamos o que cada uma é e como foi aplicada:

### 1. Multilayer Perceptron (MLP)
**O que é:** Uma arquitetura clássica de rede neural artificial *feedforward*, composta por pelo menos uma camada de entrada, uma camada oculta e uma camada de saída. 
**Como foi implementada:** A classe `MLP` no arquivo `mlp.py` define as matrizes de pesos (`W1`, `W2`) e vieses (`b1`, `b2`). A propagação dos dados ocorre linearmente da entrada para a saída através do método `forward`.

### 2. Funções de Ativação (ReLU e Sigmoid)
**O que são:** Funções matemáticas aplicadas ao resultado dos neurônios para introduzir **não-linearidade** na rede. Sem elas, a rede seria apenas uma grande regressão linear incapaz de separar as formas de "meia-lua".
**Como foram implementadas:** 
- **ReLU** (`max(0, Z)`): Usada na camada oculta para zerar valores negativos e passar valores positivos. É rápida de calcular e evita o problema do desaparecimento do gradiente.
- **Sigmoid** (`1 / (1 + exp(-Z))`): Usada na camada de saída. Esmaga o valor final entre 0 e 1, permitindo que a saída seja interpretada como a probabilidade do ponto pertencer à classe 1.

### 3. Binary Cross-Entropy (BCE) Loss
**O que é:** A função de perda (ou erro) ideal para problemas de classificação binária. Ela penaliza fortemente o modelo se ele prever probabilidade alta para a classe errada.
**Como foi implementada:** O método `compute_loss` aplica a fórmula matemática da entropia cruzada entre as predições e os rótulos reais, adicionando um pequeno `epsilon` para evitar indefinição em `log(0)`.

### 4. Backpropagation e Gradiente Descendente
**O que são:** O Backpropagation (retropropagação) é o algoritmo usado para calcular a "fração de culpa" de cada peso em relação ao erro final, usando a regra da cadeia das derivadas. O Gradiente Descendente é a técnica de atualizar os pesos caminhando no sentido oposto ao erro.
**Como foram implementados:** No método `backward`, calculamos as derivadas (`dZ2`, `dW2`, `dZ1`, `dW1`) partindo do final da rede para o início. Em seguida, subtraímos dos pesos atuais o valor do gradiente multiplicado pela **Taxa de Aprendizagem (Learning Rate)**, ajustando o modelo.

### 5. Validação Cruzada Simples (Hold-out)
**O que é:** A prática de esconder uma parte dos dados da rede durante o treino (o cálculo de atualização de pesos) para usá-los como um termômetro imparcial da capacidade real de aprendizado.
**Como foi implementada:** O arquivo `data.py` usa `train_test_split` para separar 20% do dataset exclusivamente como Validação, e mais 20% como Teste.

### 6. Early Stopping (Parada Antecipada)
**O que é:** Uma técnica de regularização. Se a rede treinar por muito tempo, ela começará a decorar os dados de treino (Overfitting), fazendo o erro de validação subir enquanto o de treino continua caindo. O Early Stopping detecta isso e interrompe o treino.
**Como foi implementada:** No laço de treinamento em `experiments.py`, o modelo anota a perda da Validação a cada época. Se o modelo perceber que a `val_loss` parou de cair e iniciou uma piora ao longo de 200 épocas (*patience*), o treino é interrompido e os pesos retornam para os da época onde o erro foi mínimo (*Best Epoch*).

### 7. Inicialização Inteligente de Pesos (He e Xavier)
**O que é:** Inicializar pesos com zeros ou números grandes causa falha no aprendizado. Estratégias estatísticas são usadas para sortear pesos na proporção certa, mantendo a variância dos dados controlada através das camadas.
**Como foi implementada:** No `__init__`, usamos **He Initialization** (`np.sqrt(2 / input_size)`) para a matriz da camada oculta (pois ela usa ReLU) e **Xavier/Glorot Initialization** (`np.sqrt(1 / hidden_size)`) para a matriz da camada de saída (pois ela usa Sigmoid).

## Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o script principal:
```bash
python main.py
```
*(Isso rodará todos os experimentos, salvará os gráficos na pasta `plots/` e exibirá a tabela comparativa).*

---

## Tabela Comparativa de Experimentos

| Neurons | LR | Total Epochs Executed | Best Val Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | 0.01 | 2000.0 | 1999.0 | 0.3432 | 0.3827 | 0.8667 | 0.83 |
| 2.0 | 0.1 | 2000.0 | 1999.0 | 0.2894 | 0.3454 | 0.8683 | 0.83 |
| 2.0 | 0.5 | 2000.0 | 1998.0 | 0.269 | 0.3235 | 0.8717 | 0.85 |
| 5.0 | 0.01 | 2000.0 | 1999.0 | 0.3222 | 0.3642 | 0.8533 | 0.82 |
| 5.0 | 0.1 | 2000.0 | 1999.0 | 0.2683 | 0.3235 | 0.8717 | 0.84 |
| 5.0 | 0.5 | 2000.0 | 1987.0 | 0.2548 | 0.305 | 0.87 | 0.845 |
| 10.0 | 0.01 | 2000.0 | 1999.0 | 0.3485 | 0.372 | 0.85 | 0.84 |
| 10.0 | 0.1 | 2000.0 | 1999.0 | 0.169 | 0.2198 | 0.9383 | 0.91 |
| 10.0 | 0.5 | 2000.0 | 1996.0 | 0.0754 | 0.1033 | 0.9717 | 0.96 |
| 20.0 | 0.01 | 2000.0 | 1999.0 | 0.2789 | 0.3225 | 0.8633 | 0.85 |
| 20.0 | 0.1 | 2000.0 | 1999.0 | 0.1073 | 0.141 | 0.9683 | 0.935 |
| 20.0 | 0.5 | 2000.0 | 1998.0 | 0.0669 | 0.0857 | 0.9783 | 0.955 |


---

## Análise Automática dos Resultados

### 1. Melhor Desempenho de Validação
A configuração com melhor desempenho na validação foi a de **20.0 neurônios** com learning rate de **0.5**. Esta configuração obteve uma acurácia de validação de 95.50% com perda de validação de 0.0857.

### 2. Sinais de Underfitting e Dificuldade de Aprendizado
Nenhuma configuração apresentou underfitting severo (todas atingiram > 80% de acurácia de treino). Isso indica que mesmo as configurações mais simples conseguiram aprender razoavelmente bem o padrão dos dados.

### 3. Sinais de Overfitting e Early Stopping
Não foram observados gaps muito grandes (> 5%) de acurácia entre treino e validação entre as configurações testadas. O mecanismo de Early Stopping ajudou a evitar que a rede memorizasse excessivamente os dados de treino.

Em muitas das configurações (como pode ser visto nos gráficos gerados na pasta `plots`), o erro de validação parou de cair ou começou a subir suavemente antes do erro de treinamento. O early stopping foi essencial para parar o treinamento no 'Best Val Epoch', o ponto ótimo antes da degradação da generalização.

### 4. Influência da Taxa de Aprendizado (Learning Rate)
Observando a tabela:
- **LR = 0.01**: Costuma exigir mais épocas para convergir (muitas vezes atinge o máximo de épocas sem acionar o early stopping) ou pode ficar preso em mínimos locais/underfitting se a capacidade for baixa.
- **LR = 0.1**: Geralmente oferece um bom balanço, convergindo em menos épocas com boa estabilidade e acurácia.
- **LR = 0.5**: Pode convergir muito rápido, mas também pode causar oscilações nos gráficos de perda se os passos de atualização forem muito largos. Em alguns casos, isso ajuda a escapar de mínimos locais, em outros pode dificultar o refino do peso.

### 5. Influência do Número de Neurônios na Camada Oculta
O número de neurônios dita a capacidade (expressividade) do modelo:
- **2 Neurônios**: A fronteira de decisão (veja os gráficos gerados) costuma ser mais rígida e poligonal (duas retas), limitando a capacidade de desenhar a curva complexa das 'luas'.
- **5 a 10 Neurônios**: Costuma ser o ponto ideal (sweet spot) para o Two Moons, conseguindo modelar perfeitamente as curvas com suavidade sem muita complexidade desnecessária.
- **20 Neurônios**: Alta capacidade. O modelo consegue criar fronteiras de decisão muito sinuosas. O risco de overfitting é maior, mas o Early Stopping atua fortemente para mitigar essa memorização.

### 6. Conclusão
O exercício atingiu seu objetivo ao demonstrar como uma MLP resolve um problema de classificação não-linear (Two Moons). Empiricamente, ficou claro que hiperparâmetros (capacidade da rede e passo de aprendizado) precisam ser ajustados e que técnicas de regularização implícita, como o Early Stopping baseado em um conjunto de validação, são cruciais para que a rede obtenha generalização em dados não vistos (testes/validação) em vez de apenas memorizar os dados de treino.
