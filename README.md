## Móudlo Machine Learning (ML) - Especialização em Análise e Ciência de dados (UFN)

Repositório destinado aos materiais de aula. Tópicos abordados:

1. Introdução (O que é Machine Learning, tipos de aprendizado, dados treino e teste, problemas de regressão, classificação, agrupamento, metodologia CRISP-DM, Algoritmos de ML e Aplicações, validação de modelos, Underfitting/Overfitting)
2. **Algortimos de Regressão**: previsão de custos em saúde e preenchimento de nulos com KNN em um dataset de Real Esate (cases com Python)
3. **Algortimos de Classificação**: previsão de churn (case com Python)
4. **Algoritmos de Agrupamento**: segmentação de uma base clientes (case com Python)
5. Redução de dimensionalidade com **Análise de Componentes Principais (PCA)** (case com Python)
6. **Ferramentas de autoML**: comparando e tunando modelos de ML com a biblioteca PyCaret (case com Python)
7. Uso de ML em pesquisa científica (cases publicados) 
8. **Redes Neurais Artificiais** (foco em Multi-Layer Perceptron)
9. **Deploy** de máquinas preditivas com **Streamlit**
    
<br>

Cases Hands-on (desenvolvimento do zero em aula):
- Previsão do tempo de vida útil restante em baterias (regressão)
- Previsão do risco de crédito para uma base de clientes (classificação)
- Segmentação de clientes de um shopping (agrupamento)

<br>

Onde encontrar bases de dados para praticar:

> **Web:**

- Kaggle: https://www.kaggle.com/datasets
- Dados gov br: https://dados.gov.br/dados/conjuntos-dados
- Data playground da Maven Analytics: https://mavenanalytics.io/data-playground
- Real World: https://data.world/datasets/open-data
  
> **Seaborn:**

```python
import seaborn as sns
sns.get_dataset_names()
# retorna uma lista de datasets disponíveis na biblioteca seaborn
```

Exemplo de uso:

```python
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
tips.head()
```  
