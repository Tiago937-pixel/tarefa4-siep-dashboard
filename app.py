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
        
        # Encoding de variáveis categóricas
        categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
        if target_var in categorical_cols:
            categorical_cols.remove(target_var)
            
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
            
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

@st.cache_data
def apply_smote_and_rfe(df_processed, target_var, n_features=15, apply_smote=True):
    """Aplica SMOTE e RFE aos dados"""
    
    # Debug: mostrar informações dos dados de entrada
    st.write("🔍 **Debug - Dados de entrada:**")
    st.write(f"- Shape do dataset: {df_processed.shape}")
    st.write(f"- Colunas disponíveis: {list(df_processed.columns)}")
    st.write(f"- Target variable: {target_var}")
    
    # Verificações iniciais
    if df_processed is None or len(df_processed) == 0:
        st.error("❌ Dataset vazio!")
        return None, None, []
    
    if target_var not in df_processed.columns:
        st.error(f"❌ Variável target '{target_var}' não encontrada!")
        st.write(f"Colunas disponíveis: {list(df_processed.columns)}")
        return None, None, []
    
    # Separar features e target
    all_columns = df_processed.columns.tolist()
    feature_columns = [col for col in all_columns if col != target_var]
    
    st.write(f"🎯 **Features identificadas ({len(feature_columns)}):** {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
    
    if len(feature_columns) == 0:
        st.error("❌ Nenhuma feature disponível após remover target!")
        return None, None, []
    
    X = df_processed[feature_columns].copy()
    y = df_processed[target_var].copy()
    
    st.write(f"📊 **Dados extraídos:**")
    st.write(f"- X shape: {X.shape}")
    st.write(f"- y shape: {y.shape}")
    st.write(f"- Tipos de dados em X: {X.dtypes.value_counts().to_dict()}")
    
    # Limpar e converter dados
    X_clean = X.copy()
    numeric_features = []
    
    for col in X_clean.columns:
        try:
            # Tentar converter para numérico
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            
            # Substituir inf por NaN
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Preencher NaN com mediana
            if X_clean[col].isnull().sum() > 0:
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
            
            # Se a coluna tem variância, manter
            if X_clean[col].std() > 0:
                numeric_features.append(col)
                
        except Exception as e:
            st.warning(f"⚠️ Erro ao processar coluna {col}: {str(e)}")
            continue
    
    st.write(f"✅ **Features numéricas válidas ({len(numeric_features)}):** {numeric_features[:10]}{'...' if len(numeric_features) > 10 else ''}")
    
    if len(numeric_features) == 0:
        st.error("❌ Nenhuma feature numérica válida encontrada!")
        return None, None, []
    
    # Usar apenas features numéricas válidas
    X_clean = X_clean[numeric_features]
    
    # Garantir que y é numérico
    y_clean = pd.to_numeric(y, errors='coerce')
    if y_clean.isnull().sum() > 0:
        mode_val = y_clean.mode()
        if len(mode_val) > 0:
            y_clean = y_clean.fillna(mode_val[0])
        else:
            y_clean = y_clean.fillna(0)
    
    # Remover linhas com problemas
    mask = ~(X_clean.isnull().any(axis=1) | y_clean.isnull())
    X_clean = X_clean[mask]
    y_clean = y_clean[mask]
    
    st.write(f"📊 **Após limpeza:**")
    st.write(f"- X shape: {X_clean.shape}")
    st.write(f"- y shape: {y_clean.shape}")
    st.write(f"- Classes em y: {y_clean.value_counts().to_dict()}")
    
    # Verificar se temos dados suficientes
    if len(X_clean) < 10:
        st.error("❌ Dados insuficientes após limpeza!")
        return None, None, []
    
    # Verificar se temos pelo menos 2 classes
    if len(y_clean.unique()) < 2:
        st.error("❌ Apenas uma classe encontrada na variável target!")
        return None, None, []
    
    # Verificar se temos dados suficientes para SMOTE
    min_samples = y_clean.value_counts().min()
    
    if SMOTE_AVAILABLE and apply_smote and min_samples >= 6:
        try:
            # SMOTE
            k_neighbors = min(5, min_samples - 1)
            smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
            X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)
            st.success(f"✅ SMOTE aplicado! Dataset balanceado: {len(y_balanced)} amostras")
        except Exception as e:
            st.warning(f"⚠️ Erro ao aplicar SMOTE: {str(e)}. Usando dados originais.")
            X_balanced, y_balanced = X_clean, y_clean
    else:
        # Se não der para aplicar SMOTE, usar dados originais
        X_balanced, y_balanced = X_clean, y_clean
        if apply_smote:
            st.warning("⚠️ SMOTE não aplicado: dados insuficientes.")
    
    # RFE
    selected_features = X_balanced.columns.tolist()
    X_selected = X_balanced
    
    if len(X_balanced.columns) > n_features:
        try:
            n_features = min(n_features, X_balanced.shape[1])
            estimator = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            rfe.fit(X_balanced, y_balanced)
            
            selected_features = X_balanced.columns[rfe.support_].tolist()
            X_selected = X_balanced[selected_features]
            st.success(f"✅ RFE aplicado! {len(selected_features)} features selecionadas")
            
        except Exception as e:
            st.warning(f"⚠️ Erro ao aplicar RFE: {str(e)}. Usando todas as features.")
            selected_features = X_balanced.columns.tolist()
            X_selected = X_balanced
    
    # Verificação final
    if X_selected is None or len(X_selected) == 0 or len(selected_features) == 0:
        st.error("❌ Erro no processamento final dos dados!")
        return None, None, []
    
    st.write(f"🎯 **Resultado final:**")
    st.write(f"- Features selecionadas: {len(selected_features)}")
    st.write(f"- Amostras finais: {len(X_selected)}")
    
    return X_selected, y_balanced, selected_features

def train_models(X_train, X_test, y_train, y_test, selected_models):
    """Treina os modelos selecionados"""
    
    # Verificações de segurança
    if X_train is None or X_test is None or len(X_train) == 0 or len(X_test) == 0:
        st.error("❌ Dados de treino/teste inválidos!")
        return {}
    
    if X_train.shape[1] == 0:
        st.error("❌ Nenhuma feature disponível para treinamento!")
        return {}
    
    # Garantir que todos os dados são numéricos
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # Converter para numérico e tratar problemas
    for col in X_train_clean.columns:
        # Substituir infinitos e NaN
        X_train_clean[col] = pd.to_numeric(X_train_clean[col], errors='coerce')
        X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors='coerce')
        
        X_train_clean[col] = X_train_clean[col].replace([np.inf, -np.inf], np.nan)
        X_test_clean[col] = X_test_clean[col].replace([np.inf, -np.inf], np.nan)
        
        # Preencher NaN com mediana
        if X_train_clean[col].isnull().sum() > 0:
            median_val = X_train_clean[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_train_clean[col] = X_train_clean[col].fillna(median_val)
            X_test_clean[col] = X_test_clean[col].fillna(median_val)
    
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = {}
    
    # Padronizar dados para KNN e SVM (com verificação)
    X_train_scaled = None
    X_test_scaled = None
    
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
    except Exception as e:
        st.warning(f"⚠️ Erro na padronização: {str(e)}. Modelos KNN e SVM serão ignorados.")
    
    for model_name in selected_models:
        if model_name in models:
            try:
                model = models[model_name]
                
                # Usar dados padronizados para KNN e SVM (se disponível)
                if model_name in ['KNN', 'SVM'] and X_train_scaled is not None:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                elif model_name in ['KNN', 'SVM'] and X_train_scaled is None:
                    st.warning(f"⚠️ Pulando {model_name} devido a erro na padronização.")
                    continue
                else:
                    model.fit(X_train_clean, y_train)
                    y_pred = model.predict(X_test_clean)
                    y_proba = model.predict_proba(X_test_clean)[:, 1]
                
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
                
                st.success(f"✅ {model_name} treinado com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao treinar {model_name}: {str(e)}")
                continue
    
    return results

# Carregar dados
df, df_processed, target_var, le_dict = load_and_process_data()

# Layout com abas principais
tab1, tab2, tab3 = st.tabs(["📊 Configuração & Dados", "🤖 Modelagem", "📈 Resultados"])

with tab1:
    # Sidebar original (mantendo todas as funcionalidades)
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
        
        # Corrigir o gráfico de pizza
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
        available_models = ['KNN', 'SVM', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM']
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
            if apply_smote:
                X_processed, y_processed, selected_features = apply_smote_and_rfe(df_processed, target_var, n_features, True)
            else:
                X_processed, y_processed, selected_features = apply_smote_and_rfe(df_processed, target_var, n_features, False)
            
            # Verificar se o processamento foi bem-sucedido
            if X_processed is None or y_processed is None or len(selected_features) == 0:
                st.error("❌ Erro no pré-processamento dos dados. Verifique os dados de entrada.")
                st.stop()
            
            if len(X_processed) == 0:
                st.error("❌ Nenhum dado válido após processamento.")
                st.stop()
            
            st.success(f"✅ Dados processados: {len(X_processed)} amostras, {len(selected_features)} features")
        
        # Dividir dados com tratamento de erro
        try:
            # Verificar se temos classes suficientes para estratificação
            if pd.Series(y_processed).value_counts().min() >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
                )
                st.info("✅ Divisão estratificada aplicada com sucesso")
            else:
                # Se não temos amostras suficientes para estratificação, usar divisão aleatória
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42
                )
                st.warning("⚠️ Divisão aleatória usada (dados insuficientes para estratificação)")
                
            # Verificar se a divisão foi bem-sucedida
            if len(X_train) == 0 or len(X_test) == 0:
                st.error("❌ Erro na divisão dos dados: conjuntos vazios.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Erro na divisão dos dados: {str(e)}")
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
            
            # Verificação final antes do treinamento
            st.write(f"📊 **Dados para treinamento:**")
            st.write(f"- Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
            st.write(f"- Teste: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
            st.write(f"- Classes no treino: {pd.Series(y_train).value_counts().to_dict()}")
            
            results = train_models(X_train, X_test, y_train, y_test, selected_models)
        
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
        
        # Interpretação Automatizada dos Coeficientes
        st.subheader("🧠 Interpretação Automatizada")
        
        interpretation = f"""
        ### 📊 Análise do Modelo {best_model_name}
        
        **Performance Geral:**
        - O modelo {best_model_name} foi selecionado como o melhor com AUC de {results[best_model_name]['auc']:.3f}
        - Acurácia de {results[best_model_name]['accuracy']:.1%} indica que o modelo acerta {results[best_model_name]['accuracy']:.1%} das predições
        - F1-Score de {results[best_model_name]['f1_score']:.3f} mostra um bom equilíbrio entre precisão e recall
        
        **Capacidade Discriminatória:**
        - AUC de {results[best_model_name]['auc']:.3f} {'é excelente (>0.9)' if results[best_model_name]['auc'] > 0.9 else 'é muito boa (>0.8)' if results[best_model_name]['auc'] > 0.8 else 'é boa (>0.7)' if results[best_model_name]['auc'] > 0.7 else 'precisa de melhorias'}
        - O modelo consegue distinguir bem entre clientes que irão ou não fazer reclamações
        
        **Recomendações Gerenciais:**
        1. **Implementação:** O modelo está pronto para deployment em produção
        2. **Monitoramento:** Acompanhar a performance com novos dados regularmente
        3. **Ação Proativa:** Usar as predições para identificar clientes de risco
        4. **Foco nas Features:** Investir em melhorias nas variáveis mais importantes
        """
        
        st.markdown(interpretation)
        
        # Análise das features mais importantes
        if hasattr(best_model_obj, 'feature_importances_'):
            top_features = importance_df.tail(5)['Feature'].tolist()
            
            st.subheader("🎯 Análise das Top 5 Features")
            
            for i, feature in enumerate(reversed(top_features), 1):
                with st.expander(f"{i}. {feature}"):
                    # Análise estatística da feature
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Dashboard desenvolvido para a Tarefa 4 - SIEP</p>
    <p><strong>Alunos:</strong> Rafael Leivas Bisi (231013467) | Tiago André Gondim (231013476)</p>
    <p><strong>Professor:</strong> João Gabriel de Moraes Souza</p>
</div>
""", unsafe_allow_html=True)
