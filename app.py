"""
Dashboard Interativo - Previsão de Reclamações com Modelos Supervisionados
Tarefa 4 - SIEP
Alunos: Rafael Leivas Bisi (231013467) | Tiago André Gondim (231013476)
Professor: João Gabriel de Moraes Souza
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Balanceamento
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    st.warning("⚠️ SMOTE não disponível. Usando dados originais.")

import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard - Previsão de Reclamações",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .info-card {
        background: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho principal
st.markdown("""
<div class="main-header">
    <h1>📊 Dashboard - Previsão de Reclamações</h1>
    <p style="text-align: center; color: white; margin: 0;">
        <strong>Tarefa 4 - SIEP</strong> | Rafael Leivas Bisi & Tiago André Gondim
    </p>
</div>
""", unsafe_allow_html=True)

# Cache para funções pesadas
@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados"""
    try:
        # Tentar carregar o arquivo
        df = pd.read_csv('marketing_campaign.csv')
        
        # Se não encontrar a variável Complain, procurar alternativas
        target_var = None
        if 'Complain' in df.columns:
            target_var = 'Complain'
        else:
            # Procurar variáveis similares
            possible_targets = [col for col in df.columns if 'complain' in col.lower() or 'response' in col.lower()]
            if possible_targets:
                target_var = possible_targets[0]
            else:
                # Criar uma variável target sintética para demonstração
                st.warning("Variável 'Complain' não encontrada. Criando variável sintética para demonstração.")
                np.random.seed(42)
                df['Complain'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
                target_var = 'Complain'
        
        # Processar dados
        df_processed = df.copy()
        
        # Remover valores nulos
        df_processed = df_processed.dropna()
        
        # Encoding de variáveis categóricas
        categorical_cols = df_processed.select_dtypes(include=[object]).columns.tolist()
        if target_var in categorical_cols:
            categorical_cols.remove(target_var)
            
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
            
        # Converter tipos para numérico
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Remover linhas com NaN resultantes da conversão
        df_processed = df_processed.dropna()
        
        return df, df_processed, target_var, le_dict
        
    except FileNotFoundError:
        # Gerar dataset sintético para demonstração
        st.warning("Arquivo 'marketing_campaign.csv' não encontrado. Gerando dados sintéticos para demonstração.")
        
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'Age': np.random.normal(40, 12, n_samples).astype(int),
            'Income': np.random.normal(50000, 15000, n_samples),
            'Recency': np.random.poisson(50, n_samples),
            'MntWines': np.random.poisson(300, n_samples),
            'MntFruits': np.random.poisson(50, n_samples),
            'MntMeatProducts': np.random.poisson(150, n_samples),
            'MntFishProducts': np.random.poisson(40, n_samples),
            'MntSweetProducts': np.random.poisson(25, n_samples),
            'MntGoldProds': np.random.poisson(60, n_samples),
            'NumDealsPurchases': np.random.poisson(3, n_samples),
            'NumWebPurchases': np.random.poisson(5, n_samples),
            'NumCatalogPurchases': np.random.poisson(3, n_samples),
            'NumStorePurchases': np.random.poisson(6, n_samples),
            'NumWebVisitsMonth': np.random.poisson(7, n_samples),
            'Education': np.random.choice(['Graduation', 'PhD', 'Master', 'Basic'], n_samples),
            'Marital_Status': np.random.choice(['Married', 'Single', 'Divorced', 'Widow'], n_samples),
            'Kidhome': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
            'Teenhome': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Criar variável target baseada em algumas características
        complaint_prob = (
            (df['Recency'] > 80) * 0.3 +
            (df['Income'] < 30000) * 0.2 +
            (df['NumWebVisitsMonth'] > 10) * 0.2 +
            (df['MntWines'] < 100) * 0.1 +
            0.1
        )
        df['Complain'] = np.random.binomial(1, complaint_prob)
        
        # Processar dados
        df_processed = df.copy()
        categorical_cols = ['Education', 'Marital_Status']
        le_dict = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            le_dict[col] = le
            
        return df, df_processed, 'Complain', le_dict

def safe_smote_and_rfe(df_processed, target_var, n_features=15, apply_smote=True):
    """Aplica SMOTE e RFE aos dados com tratamento de erro robusto"""
    try:
        X = df_processed.drop(columns=[target_var])
        y = df_processed[target_var]
        
        # Garantir que todos os dados são numéricos
        X = X.select_dtypes(include=[np.number])
        
        # Verificar se temos dados suficientes
        if len(X) == 0 or len(y) == 0:
            st.error("Erro: Dados vazios após processamento")
            return None, None, None
        
        # Verificar se X tem colunas
        if X.shape[1] == 0:
            st.error("Erro: Nenhuma variável numérica encontrada")
            return None, None, None
        
        # Garantir que todos os dados são finitos
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        
        if len(X) == 0:
            st.error("Erro: Dados vazios após limpeza")
            return None, None, None
        
        # Verificar se temos classes suficientes para SMOTE
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        
        if SMOTE_AVAILABLE and apply_smote and min_samples >= 6 and len(class_counts) > 1:
            try:
                k_neighbors = min(5, min_samples - 1)
                smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
                X_balanced, y_balanced = smote.fit_resample(X, y)
                st.success(f"✅ SMOTE aplicado com sucesso. Dataset balanceado: {len(y_balanced)} amostras")
            except Exception as e:
                st.warning(f"Erro ao aplicar SMOTE: {str(e)}. Usando dados originais.")
                X_balanced, y_balanced = X, y
        else:
            X_balanced, y_balanced = X, y
            if apply_smote:
                st.warning("SMOTE não aplicado: dados insuficientes ou biblioteca indisponível")
        
        # RFE
        try:
            n_features = min(n_features, X_balanced.shape[1])
            if n_features > 0:
                estimator = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
                rfe = RFE(estimator=estimator, n_features_to_select=n_features)
                rfe.fit(X_balanced, y_balanced)
                
                selected_features = X_balanced.columns[rfe.support_].tolist()
                X_selected = X_balanced[selected_features]
                
                st.success(f"✅ RFE aplicado: {len(selected_features)} features selecionadas")
            else:
                selected_features = X_balanced.columns.tolist()
                X_selected = X_balanced
                st.warning("RFE não aplicado: número de features inválido")
                
        except Exception as e:
            st.warning(f"Erro ao aplicar RFE: {str(e)}. Usando todas as features.")
            selected_features = X_balanced.columns.tolist()
            X_selected = X_balanced
        
        return X_selected, y_balanced, selected_features
        
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {str(e)}")
        return None, None, None

def train_models_safe(X_train, X_test, y_train, y_test, selected_models):
    """Treina os modelos selecionados com tratamento de erro robusto"""
    
    # Verificar se os dados são válidos
    if X_train is None or len(X_train) == 0 or X_test is None or len(X_test) == 0:
        st.error("Erro: Dados de treino ou teste vazios")
        return {}
    
    if y_train is None or len(y_train) == 0 or y_test is None or len(y_test) == 0:
        st.error("Erro: Variável target vazia")
        return {}
    
    # Verificar se as dimensões coincidem
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        st.error("Erro: Dimensões incompatíveis entre X e y")
        return {}
    
    # Converter para array numpy e verificar tipos
    try:
        X_train = np.array(X_train, dtype=np.float64)
        X_test = np.array(X_test, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.int64)
        y_test = np.array(y_test, dtype=np.int64)
    except Exception as e:
        st.error(f"Erro na conversão de tipos: {str(e)}")
        return {}
    
    # Verificar se há valores infinitos ou NaN
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        st.error("Erro: Dados contêm valores infinitos ou NaN")
        return {}
    
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42, C=1.0, kernel='rbf'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=50),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', max_depth=6),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, max_depth=6),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    }
    
    results = {}
    
    # Padronizar dados apenas para modelos que precisam
    scale_models = ['KNN', 'SVM', 'Logistic Regression']
    
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        st.error(f"Erro na padronização: {str(e)}")
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    for model_name in selected_models:
        if model_name in models:
            try:
                model = models[model_name]
                
                # Usar dados padronizados para modelos específicos
                if model_name in scale_models:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calcular métricas
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'auc': roc_auc_score(y_test, y_proba),
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'model': model
                }
                
                st.success(f"✅ Modelo {model_name} treinado com sucesso")
                
            except Exception as e:
                st.error(f"Erro ao treinar {model_name}: {str(e)}")
                continue
    
    return results

# Carregar dados
try:
    df, df_processed, target_var, le_dict = load_and_process_data()
    
    # Layout com abas principais
    tab1, tab2, tab3 = st.tabs(["📊 Configuração & Dados", "🤖 Modelagem", "📈 Resultados"])
    
    with tab1:
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configurações")

            # Informações do dataset
            st.subheader("📊 Informações do Dataset")
            st.info(f"""
            **Dimensões:** {df.shape[0]} linhas × {df.shape[1]} colunas
            **Variável Target:** {target_var}
            **Missing Values:** {df.isnull().sum().sum()}
            """)

            # Análise da distribuição do target
            target_dist = df[target_var].value_counts()
            st.subheader(f"🎯 Distribuição de {target_var}")
            
            fig_target = px.pie(
                values=target_dist.values, 
                names=[f"Classe {i}" for i in target_dist.index],
                title=f"Distribuição de {target_var}"
            )
            st.plotly_chart(fig_target, use_container_width=True)

            # Configurações de modelagem
            st.subheader("🔧 Configurações de Modelagem")

            # Seleção de features via RFE
            n_features = st.slider(
                "Número de features (RFE)", 
                min_value=5, 
                max_value=min(20, len(df_processed.columns)-1), 
                value=15
            )

            # Aplicar SMOTE
            if SMOTE_AVAILABLE:
                apply_smote = st.checkbox("Aplicar SMOTE", value=True)
            else:
                apply_smote = False

            # Seleção de modelos
            st.subheader("🤖 Seleção de Modelos")
            available_models = ['KNN', 'SVM', 'Decision Tree', 'Random Forest', 'AdaBoost', 
                              'Gradient Boosting', 'XGBoost', 'LightGBM', 'Logistic Regression']
            selected_models = st.multiselect(
                "Escolha os modelos para treinar:",
                available_models,
                default=['Random Forest', 'XGBoost', 'LightGBM']
            )

            # Tamanho do conjunto de teste
            test_size = st.slider("Tamanho do conjunto de teste (%)", 10, 50, 30) / 100

        # Área principal da Tab 1
        st.subheader("📊 Visão Geral dos Dados")
        
        # Filtros interativos
        st.subheader("🔍 Filtros Interativos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtro por idade (se existir)
            if 'Age' in df.columns:
                age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
                age_range = st.slider("Faixa de Idade", age_min, age_max, (age_min, age_max))
                df_filtered = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
            else:
                df_filtered = df.copy()
        
        with col2:
            # Filtro por renda (se existir)
            if 'Income' in df.columns:
                income_min, income_max = float(df['Income'].min()), float(df['Income'].max())
                income_range = st.slider("Faixa de Renda", income_min, income_max, (income_min, income_max))
                df_filtered = df_filtered[(df_filtered['Income'] >= income_range[0]) & (df_filtered['Income'] <= income_range[1])]

        # Mostrar estatísticas dos dados filtrados
        st.write(f"**Dados após filtros:** {len(df_filtered)} registros")
        
        # Visualizações dos dados filtrados
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Age' in df_filtered.columns:
                fig_age_dist = px.histogram(df_filtered, x='Age', color=target_var, 
                                          title="Distribuição de Idade por Reclamação")
                st.plotly_chart(fig_age_dist, use_container_width=True)
        
        with col2:
            if 'Income' in df_filtered.columns:
                fig_income_dist = px.box(df_filtered, x=target_var, y='Income', 
                                       title="Distribuição de Renda por Reclamação")
                st.plotly_chart(fig_income_dist, use_container_width=True)

        # Preview dos dados
        st.subheader("👀 Preview dos Dados")
        st.dataframe(df_filtered.head(), use_container_width=True)
        
        # Estatísticas básicas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Registros", f"{len(df_filtered):,}")
        
        with col2:
            st.metric("Número de Features", len(df_filtered.columns) - 1)
        
        with col3:
            target_balance = df_filtered[target_var].value_counts(normalize=True).min()
            st.metric("Balanceamento", f"{target_balance:.1%}")

    with tab2:
        st.header("🤖 Treinamento de Modelos")
        
        # Botão para executar análise
        run_analysis = st.button("🚀 Executar Análise", type="primary")

        if run_analysis and selected_models:
            
            # Aplicar SMOTE e RFE
            with st.spinner("Processando dados..."):
                processed_data = safe_smote_and_rfe(df_processed, target_var, n_features, apply_smote)
                
                if processed_data[0] is not None:
                    X_processed, y_processed, selected_features = processed_data
                else:
                    st.error("Erro no processamento dos dados. Verifique o dataset.")
                    st.stop()
            
            # Dividir dados com tratamento de erro
            try:
                # Verificar se temos classes suficientes para estratificação
                if y_processed.value_counts().min() >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
                    )
                else:
                    # Se não temos amostras suficientes para estratificação, usar divisão aleatória
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed, test_size=test_size, random_state=42
                    )
                    st.warning("⚠️ Divisão aleatória usada (dados insuficientes para estratificação)")
            except Exception as e:
                st.error(f"Erro na divisão dos dados: {str(e)}")
                st.stop()
            
            # Mostrar features selecionadas
            st.subheader("🎯 Features Selecionadas via RFE")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Total de features selecionadas:** {len(selected_features)}")
                
            with col2:
                with st.expander("Ver lista completa"):
                    st.write(selected_features)
            
            # Treinar modelos
            with st.spinner("Treinando modelos..."):
                results = train_models_safe(X_train, X_test, y_train, y_test, selected_models)
            
            # Salvar resultados no session state
            st.session_state['results'] = results
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['selected_features'] = selected_features
            
            if results:
                st.success(f"✅ {len(results)} modelos treinados com sucesso!")
            else:
                st.error("❌ Nenhum modelo foi treinado com sucesso!")

        elif run_analysis and not selected_models:
            st.error("❌ Selecione pelo menos um modelo para treinar!")
        
        else:
            # Tela inicial
            st.subheader("👋 Bem-vindo ao Sistema de Modelagem!")
            
            st.markdown("""
            Configure os parâmetros na sidebar e clique em "Executar Análise" para iniciar.
            
            ### 🚀 Como usar:
            1. **Configure os parâmetros** na sidebar à esquerda
            2. **Selecione os modelos** que deseja treinar
            3. **Clique em "Executar Análise"** para iniciar o processo
            
            ### 📊 Funcionalidades:
            - ⚖️ **Balanceamento de dados** com SMOTE
            - 🎯 **Seleção de features** com RFE
            - 🤖 **Múltiplos modelos** de Machine Learning
            - 📈 **Visualizações interativas** dos resultados
            - 🧠 **Interpretação automatizada** dos modelos
            """)

    with tab3:
        st.header("📈 Resultados e Análises")
        
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            X_test = st.session_state['X_test'] 
            y_test = st.session_state['y_test']
            selected_features = st.session_state['selected_features']
            
            # Exibir resultados
            st.subheader("📊 Resultados dos Modelos")
            
            # Criar tabela de resultados
            results_df = pd.DataFrame({
                'Modelo': list(results.keys()),
                'Acurácia': [results[model]['accuracy'] for model in results.keys()],
                'Precisão': [results[model]['precision'] for model in results.keys()],
                'Recall': [results[model]['recall'] for model in results.keys()],
                'F1-Score': [results[model]['f1_score'] for model in results.keys()],
                'AUC': [results[model]['auc'] for model in results.keys()]
            }).round(4)
            
            # Destacar melhor modelo
            best_model_idx = results_df['AUC'].idxmax()
            best_model_name = results_df.loc[best_model_idx, 'Modelo']
            
            # Mostrar tabela com destaque
            st.dataframe(
                results_df.style.highlight_max(subset=['AUC'], color='lightgreen'),
                use_container_width=True
            )
            
            # Cards de métricas do melhor modelo
            st.subheader(f"🏆 Melhor Modelo: {best_model_name}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Acurácia", f"{results[best_model_name]['accuracy']:.3f}")
            with col2:
                st.metric("Precisão", f"{results[best_model_name]['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{results[best_model_name]['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{results[best_model_name]['f1_score']:.3f}")
            with col5:
                st.metric("AUC", f"{results[best_model_name]['auc']:.3f}")
            
            # Visualizações
            st.subheader("📈 Visualizações e Métricas")
            
            tab_comp, tab_roc, tab_conf, tab_imp = st.tabs(["📊 Comparação", "📈 Curvas ROC", "🔥 Matriz de Confusão", "🎯 Importância"])
            
            with tab_comp:
                # Gráfico de comparação
                fig_comparison = go.Figure()
                
                metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC']
                for metric in metrics:
                    fig_comparison.add_trace(go.Bar(
                        name=metric,
                        x=results_df['Modelo'],
                        y=results_df[metric],
                        text=results_df[metric].round(3),
                        textposition='auto'
                    ))
                
                fig_comparison.update_layout(
                    title="Comparação de Métricas por Modelo",
                    xaxis_title="Modelos",
                    yaxis_title="Score",
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            with tab_roc:
                # Curvas ROC
                fig_roc = go.Figure()
                
                for model_name in results.keys():
                    fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_proba'])
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC = {results[model_name]["auc"]:.3f})',
                        line=dict(width=3)
                    ))
                
                # Linha diagonal
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Aleatório',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig_roc.update_layout(
                    title='Curvas ROC - Comparação de Modelos',
                    xaxis_title='Taxa de Falsos Positivos',
                    yaxis_title='Taxa de Verdadeiros Positivos',
                    height=500
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with tab_conf:
                # Matriz de confusão do melhor modelo
                y_pred_best = results[best_model_name]['y_pred']
                cm = confusion_matrix(y_test, y_pred_best)
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title=f'Matriz de Confusão - {best_model_name}'
                )
                
                fig_cm.update_layout(
                    xaxis_title='Predito',
                    yaxis_title='Real',
                    height=400
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Interpretação da matriz
                if cm.size == 4:  # Matriz 2x2
                    tn, fp, fn, tp = cm.ravel()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        **Interpretação da Matriz:**
                        - Verdadeiros Negativos: {tn}
                        - Falsos Positivos: {fp}
                        - Falsos Negativos: {fn}
                        - Verdadeiros Positivos: {tp}
                        """)
                    
                    with col2:
                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        
                        st.markdown(f"""
                        **Métricas Calculadas:**
                        - Acurácia: {accuracy:.3f}
                        - Precisão: {precision:.3f}
                        - Recall: {recall:.3f}
                        """)
            
            with tab_imp:
                # Importância das features
                best_model_obj = results[best_model_name]['model']
                
                if hasattr(best_model_obj, 'feature_importances_'):
                    importances = best_model_obj.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importância': importances
                    }).sort_values('Importância', ascending=True)
                    
                    fig_importance = px.bar(
                        importance_df.tail(15),
                        x='Importância',
                        y='Feature',
                        orientation='h',
                        title=f'Top 15 Features Mais Importantes - {best_model_name}',
                        height=600
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Tabela de importância
                    st.subheader("📋 Tabela de Importância")
                    st.dataframe(
                        importance_df.sort_values('Importância', ascending=False),
                        use_container_width=True
                    )
                    
                else:
                    st.info("Importância de features não disponível para este modelo.")
            
            # Interpretação Automatizada
            st.subheader("🧠 Interpretação Automatizada")
            
            # Análise do melhor modelo
            auc_performance = "excelente (>0.9)" if results[best_model_name]['auc'] > 0.9 else \
                             "muito boa (>0.8)" if results[best_model_name]['auc'] > 0.8 else \
                             "boa (>0.7)" if results[best_model_name]['auc'] > 0.7 else \
                             "precisa de melhorias"
            
            interpretation = f"""
            ### 📊 Análise do Modelo {best_model_name}
            
            **Performance Geral:**
            - O modelo {best_model_name} foi selecionado como o melhor com AUC de {results[best_model_name]['auc']:.3f}
            - Acurácia de {results[best_model_name]['accuracy']:.1%} indica que o modelo acerta {results[best_model_name]['accuracy']:.1%} das predições
            - F1-Score de {results[best_model_name]['f1_score']:.3f} mostra um bom equilíbrio entre precisão e recall
            
            **Capacidade Discriminatória:**
            - AUC de {results[best_model_name]['auc']:.3f} {auc_performance}
            - O modelo consegue distinguir bem entre clientes que irão ou não fazer reclamações
            
            **Recomendações Gerenciais:**
            1. **Implementação:** O modelo está pronto para deployment em produção
            2. **Monitoramento:** Acompanhar a performance com novos dados regularmente
            3. **Ação Proativa:** Usar as predições para identificar clientes de risco
            4. **Foco nas Features:** Investir em melhorias nas variáveis mais importantes
            """
            
            st.markdown(interpretation)
            
            # Análise das features mais importantes (se disponível)
            if hasattr(best_model_obj, 'feature_importances_'):
                top_features = importance_df.tail(5)['Feature'].tolist()
                
                st.subheader("🎯 Análise das Top 5 Features")
                
                for i, feature in enumerate(reversed(top_features), 1):
                    with st.expander(f"{i}. {feature}"):
                        # Análise estatística da feature se os dados estiverem disponíveis
                        if feature in df_processed.columns:
                            feature_stats = df_processed.groupby(target_var)[feature].agg(['mean', 'median', 'std']).round(3)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Estatísticas por classe:**")
                                st.dataframe(feature_stats)
                            
                            with col2:
                                # Boxplot da feature por classe
                                fig_box = px.box(
                                    df_processed, 
                                    x=target_var, 
                                    y=feature,
                                    title=f'Distribuição de {feature} por classe'
                                )
                                st.plotly_chart(fig_box, use_container_width=True)

        else:
            st.markdown("""
            ### ⚠️ Nenhum Resultado Disponível
            Execute a análise na aba "Modelagem" para ver os resultados aqui.
            """)

except Exception as e:
    st.error(f"Erro ao carregar o dashboard: {str(e)}")
    st.markdown("""
    ### Possíveis soluções:
    1. Verifique se o arquivo 'marketing_campaign.csv' está na pasta correta
    2. Verifique se todas as bibliotecas estão instaladas
    3. Recarregue a página
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Dashboard desenvolvido para a Tarefa 4 - SIEP</p>
    <p><strong>Alunos:</strong> Rafael Leivas Bisi (231013467) | Tiago André Gondim (231013476)</p>
    <p><strong>Professor:</strong> João Gabriel de Moraes Souza</p>
</div>
""", unsafe_allow_html=True)
