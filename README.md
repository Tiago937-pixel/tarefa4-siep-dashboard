# 📊 Dashboard - Previsão de Reclamações com Modelos Supervisionados

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.2-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![Status](https://img.shields.io/badge/status-working-green.svg)

## 🎯 Sobre o Projeto

Dashboard interativo desenvolvido para a **Tarefa 4 - SIEP** da disciplina de Sistemas de Informação Empresarial e Produção da Universidade de Brasília. O objetivo é criar um modelo preditivo para identificar clientes com maior probabilidade de terem feito reclamações nos últimos 2 anos.

**Autores:**
- Rafael Leivas Bisi (231013467)
- Tiago André Gondim (231013476)

**Professor:** João Gabriel de Moraes Souza

## 🚀 Funcionalidades

### 🔍 Análise de Dados
- **Filtros Interativos:** Filtragem por idade, renda e outras variáveis
- **Visualização Dinâmica:** Gráficos interativos com Plotly
- **Estatísticas Descritivas:** Análise automática dos dados

### 🎯 Modelagem
- **Seleção de Variáveis:** Manual ou automática via RFE
- **Balanceamento:** SMOTE para dados desbalanceados (quando disponível)
- **Múltiplos Modelos:** 8 algoritmos de Machine Learning

### 📈 Modelos Implementados

#### Baseados em Distância
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

#### Bagging
- Decision Tree
- Random Forest

#### Boosting
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

### 📊 Métricas e Visualizações
- **Métricas:** Acurácia, Precisão, Recall, F1-Score, AUC
- **Curva ROC:** Comparação entre modelos
- **Matriz de Confusão:** Análise detalhada dos resultados
- **Importância de Features:** Ranking das variáveis mais relevantes

### 🧠 Interpretação Automatizada
- Análise automatizada do melhor modelo
- Recomendações gerenciais baseadas nos resultados
- Insights estratégicos para tomada de decisão

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit 1.28.2** - Interface web interativa
- **Scikit-learn 1.3.2** - Algoritmos de Machine Learning
- **XGBoost & LightGBM** - Algoritmos de Boosting avançados
- **Plotly 5.17.0** - Visualizações interativas
- **Pandas & NumPy** - Manipulação de dados
- **Imbalanced-learn 0.11.0** - Balanceamento de dados com SMOTE

## 📦 Instalação e Execução

### Pré-requisitos
- Python 3.8 ou superior
- Git

### Passo a Passo

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/tarefa4-siep-dashboard.git
cd tarefa4-siep-dashboard
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Execute o dashboard:**
```bash
streamlit run app.py
```

4. **Acesse no navegador:**
```
http://localhost:8501
```

## 📁 Estrutura do Projeto

```
tarefa4-siep-dashboard/
│
├── app.py                    # Aplicação principal do Streamlit
├── requirements.txt          # Dependências do projeto
├── README.md                # Documentação
├── marketing_campaign.csv   # Dataset (opcional - gera sintético se não existir)
└── .streamlit/
    └── config.toml          # Configurações do Streamlit
```

## 📊 Dataset

O projeto utiliza o dataset **Customer Personality Analysis** do Kaggle, que contém informações sobre:

- **Demográficas:** Idade, educação, estado civil
- **Comportamentais:** Gastos por categoria, canais de compra
- **Interação:** Campanhas anteriores, reclamações

**Variável Target:** `Complain` (1 = reclamou, 0 = não reclamou)

## 🎮 Como Usar

1. **Configure os filtros** na sidebar para explorar subconjuntos dos dados
2. **Selecione as variáveis** manualmente ou use RFE para seleção automática
3. **Escolha os modelos** que deseja treinar e comparar
4. **Execute a análise** e explore os resultados interativamente

## 📈 Resultados

O dashboard fornece:

- **Comparação de Modelos:** Tabela e gráficos comparativos
- **Melhor Modelo:** Identificação automática baseada em AUC
- **Interpretação:** Análise das variáveis mais importantes
- **Recomendações:** Insights para aplicação prática

## 🎯 Requisitos da Tarefa

### ✅ Implementado

- [x] **Balanceamento:** SMOTE para variável desbalanceada
- [x] **Seleção de Variáveis:** RFE e seleção manual
- [x] **Múltiplos Modelos:** Todos os algoritmos solicitados
- [x] **Métricas Completas:** AUC, ROC, Precisão, Recall, F1-score, Matriz de Confusão
- [x] **Interpretação:** Análise de importância e recomendações gerenciais

### 🎁 Bônus de Inovação (+2 pontos)

- [x] **Dashboard Interativo:** Streamlit Cloud
- [x] **Filtros Dinâmicos:** Exploração interativa dos dados
- [x] **Seleção de Variáveis:** Interface dinâmica para modelagem
- [x] **Métricas em Tempo Real:** Visualizações atualizadas automaticamente
- [x] **Interpretação Automatizada:** Insights gerados automaticamente

## 🔧 Troubleshooting

### Problema com SMOTE
Se encontrar erros relacionados ao `imbalanced-learn`:
- O dashboard detecta automaticamente a disponibilidade
- Continua funcionando com dados originais se SMOTE não estiver disponível
- Usa versões compatíveis especificadas no requirements.txt

### Dataset Não Encontrado
Se o arquivo `marketing_campaign.csv` não estiver presente:
- O dashboard gera automaticamente um dataset sintético
- Mantém todas as funcionalidades ativas
- Exibe aviso sobre o uso de dados sintéticos

### Problemas de Compatibilidade
- Use as versões exatas especificadas no `requirements.txt`
- Execute `pip install -r requirements.txt` para garantir compatibilidade

## 🚀 Deploy no Streamlit Cloud

1. **Faça o push para o GitHub**
2. **Acesse [share.streamlit.io](https://share.streamlit.io)**
3. **Conecte seu repositório GitHub**
4. **Configure:**
   - Repository: `seu-usuario/tarefa4-siep-dashboard`
   - Branch: `main`
   - Main file path: `app.py`
5. **Deploy!**

## 🤝 Contribuição

Este projeto foi desenvolvido como trabalho acadêmico. Para sugestões ou melhorias:

1. Faça um fork do projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Contato

- **Rafael Leivas Bisi:** 231013467@aluno.unb.br
- **Tiago André Gondim:** 231013476@aluno.unb.br

---

**Universidade de Brasília - Faculdade de Tecnologia**  
**Departamento de Engenharia de Produção**  
**Disciplina:** Sistemas de Informação Empresarial e Produção  
**Professor:** João Gabriel de Moraes Souza
