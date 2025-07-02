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
import warnings
warnings.filterwarnings('ignore')

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

# Configuração da página
st.set_page_config(
    page_title="Dashboard - Previsão de Reclamações",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado melhorado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(33,150,243,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(76,175,80,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(255,152,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        color: #495057;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102,126,234,0.4);
    }
    .feature-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #bbdefb;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho principal
st.markdown("""
<div class="main-header">
    <h1>📊 Dashboard - Previsão de Reclamações</h1>
    <p><strong>Tarefa 4 - SIEP</strong> | Rafael Leivas Bisi (231013467) & Tiago André Gondim (231013476)</p>
    <p><em>Professor: João Gabriel de Moraes Souza</em></p>
</div>
""", unsafe_allow_html=True)

# Função para carregar dados melhorada
@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados"""
    try:
        df = pd.read_csv('marketing_campaign.csv')
        st.success("✅ Dataset real carregado com sucesso!")
        
        if 'Complain' not in df.columns:
            np.random.seed(42)
            complaint_prob = (
                (df.get('Recency', 50) > df.get('Recency', 50).quantile(0.7)) * 0.3 +
                (df.get('Income', 50000) < df.get('Income', 50000).quantile(0.3)) * 0.2 +
                (df.get('NumWebVisitsMonth', 5) > df.get('NumWebVisitsMonth', 5).quantile(0.8)) * 0.2 +
                0.1
            )
            df['Complain'] = np.random.binomial(1, np.clip(complaint_prob, 0, 1))
        
    except FileNotFoundError:
        st.warning("📁 Dataset não encontrado. Gerando dados sintéticos...")
        
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'Year_Birth': np.random.normal(1975, 12, n_samples).astype(int),
            'Education': np.random.choice(['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'], 
                                        n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
            'Marital_Status': np.random.choice(['Married', 'Single', 'Together', 'Divorced', 'Widow'], 
                                             n_samples, p=[0.4, 0.2, 0.25, 0.1, 0.05]),
            'Income': np.random.lognormal(10.5, 0.6, n_samples),
            'Kidhome': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
            'Teenhome': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.25, 0.05]),
            'Recency': np.random.poisson(50, n_samples),
            'MntWines': np.random.poisson(300, n_samples),
            'MntFruits': np.random.poisson(26, n_samples),
            'MntMeatProducts': np.random.poisson(166, n_samples),
            'MntFishProducts': np.random.poisson(37, n_samples),
            'MntSweetProducts': np.random.poisson(27, n_samples),
            'MntGoldProds': np.random.poisson(44, n_samples),
            'NumDealsPurchases': np.random.poisson(2, n_samples),
            'NumWebPurchases': np.random.poisson(4, n_samples),
            'NumCatalogPurchases': np.random.poisson(2, n_samples),
            'NumStorePurchases': np.random.poisson(5, n_samples),
            'NumWebVisitsMonth': np.random.poisson(7, n_samples),
            'AcceptedCmp1': np.random.binomial(1, 0.06, n_samples),
            'AcceptedCmp2': np.random.binomial(1, 0.01, n_samples),
            'AcceptedCmp3': np.random.binomial(1, 0.07, n_samples),
            'AcceptedCmp4': np.random.binomial(1, 0.07, n_samples),
            'AcceptedCmp5': np.random.binomial(1, 0.07, n_samples),
            'Response': np.random.binomial(1, 0.15, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['Age'] = 2023 - df['Year_Birth']
        
        complaint_prob = (
            (df['Recency'] > df['Recency'].quantile(0.75)) * 0.25 +
            (df['Income'] < df['Income'].quantile(0.25)) * 0.2 +
            (df['NumWebVisitsMonth'] > df['NumWebVisitsMonth'].quantile(0.8)) * 0.15 +
            (df['Age'] > 65) * 0.1 +
            (df['MntWines'] < df['MntWines'].quantile(0.3)) * 0.1 +
            0.05
        )
        
        df['Complain'] = np.random.binomial(1, np.clip(complaint_prob, 0, 1))
    
    # Processar dados
    df_processed = df.copy()
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
    if 'Complain' in categorical_cols:
        categorical_cols.remove('Complain')
    
    le_dict = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    return df, df_processed, 'Complain', le_dict

# Função para SMOTE e RFE
@st.cache_data
def apply_preprocessing(df_processed, target_var, selected_features, apply_smote=True, n_features=15):
    """Aplica pré-processamento"""
    
    # Se features específicas foram selecionadas, usar apenas elas
    if selected_features:
        available_features = [f for f in selected_features if f in df_processed.columns and f != target_var]
        if not available_features:
            available_features = [col for col in df_processed.columns if col != target_var]
    else:
        available_features = [col for col in df_processed.columns if col != target_var]
    
    X = df_processed[available_features]
    y = df_processed[target_var]
    
    # Limpar dados
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    # Remover linhas problemáticas
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Aplicar SMOTE se disponível e solicitado
    if SMOTE_AVAILABLE and apply_smote and len(y_clean.unique()) > 1:
        min_samples = y_clean.value_counts().min()
        if min_samples >= 6:  # Garantir que temos amostras suficientes
            try:
                k_neighbors = min(5, min_samples - 1)
                smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
                X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)
                st.success(f"✅ SMOTE aplicado! Dataset balanceado: {len(y_balanced)} amostras")
            except Exception as e:
                st.warning(f"⚠️ Erro no SMOTE: {str(e)}. Usando dados originais.")
                X_balanced, y_balanced = X_clean, y_clean
        else:
            st.warning("⚠️ Poucas amostras para SMOTE. Usando dados originais.")
            X_balanced, y_balanced = X_clean, y_clean
    else:
        X_balanced, y_balanced = X_clean, y_clean
    
    # Aplicar RFE se solicitado
    if len(X_balanced.columns) > n_features:
        try:
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, len(X_balanced.columns)))
            rfe.fit(X_balanced, y_balanced)
            selected_cols = X_balanced.columns[rfe.support_].tolist()
            X_final = X_balanced[selected_cols]
            st.info(f"🎯 RFE aplicado: {len(selected_cols)} features selecionadas")
        except Exception as e:
            st.warning(f"⚠️ Erro no RFE: {str(e)}. Usando todas as features.")
            X_final = X_balanced
            selected_cols = X_balanced.columns.tolist()
    else:
        X_final = X_balanced
        selected_cols = X_balanced.columns.tolist()
    
    return X_final, y_balanced, selected_cols

# Função para treinar modelos
def train_models(X_train, X_test, y_train, y_test, selected_models):
    """Treina os modelos selecionados"""
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    progress_bar = st.progress(0)
    
    for i, model_name in enumerate(selected_models):
        if model_name in models:
            try:
                model = models[model_name]
                
                if model_name in ['KNN', 'SVM', 'Logistic Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                
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
                
                progress_bar.progress((i + 1) / len(selected_models))
                
            except Exception as e:
                st.error(f"❌ Erro em {model_name}: {str(e)}")
    
    progress_bar.empty()
    return results

# Carregar dados
df, df_processed, target_var, le_dict = load_and_process_data()

# Layout principal com abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploração de Dados", "⚙️ Configuração", "🤖 Modelagem", "📈 Resultados"])

with tab1:
    st.header("📊 Exploração dos Dados")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total de Registros", f"{len(df):,}")
    with col2:
        st.metric("📋 Número de Variáveis", len(df.columns))
    with col3:
        complain_rate = df[target_var].mean()
        st.metric("⚠️ Taxa de Reclamação", f"{complain_rate:.1%}")
    with col4:
        balance_ratio = df[target_var].value_counts().min() / df[target_var].value_counts().max()
        st.metric("⚖️ Balanceamento", f"{balance_ratio:.3f}")
    
    # Visualizações
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição do target
        target_counts = df[target_var].value_counts()
        fig_target = px.pie(
            values=target_counts.values, 
            names=['Não Reclamou', 'Reclamou'],
            title="🎯 Distribuição de Reclamações",
            color_discrete_sequence=['#28a745', '#dc3545']
        )
        fig_target.update_layout(height=400)
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        # Distribuição de idade (se existir)
        if 'Age' in df.columns:
            fig_age = px.histogram(
                df, x='Age', color=target_var,
                title="📊 Distribuição de Idade por Reclamação",
                marginal="box",
                color_discrete_sequence=['#28a745', '#dc3545']
            )
        elif 'Year_Birth' in df.columns:
            df_temp = df.copy()
            df_temp['Age'] = 2023 - df_temp['Year_Birth']
            fig_age = px.histogram(
                df_temp, x='Age', color=target_var,
                title="📊 Distribuição de Idade por Reclamação",
                marginal="box",
                color_discrete_sequence=['#28a745', '#dc3545']
            )
        else:
            fig_age = px.bar(x=['Dados'], y=[1], title="Dados de Idade não disponíveis")
        
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_var in numeric_cols:
        numeric_cols.remove(target_var)
    
    if numeric_cols:
        stats_cols = numeric_cols[:6]  # Mostrar apenas as primeiras 6 colunas
        
        stats_summary = df.groupby(target_var)[stats_cols].agg(['mean', 'std']).round(2)
        
        # Mostrar em formato mais limpo
        for col in stats_cols:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"📊 {col} - Não Reclamou", 
                    f"{stats_summary.loc[0, (col, 'mean')]:.2f}",
                    f"±{stats_summary.loc[0, (col, 'std')]:.2f}"
                )
            with col2:
                st.metric(
                    f"📊 {col} - Reclamou", 
                    f"{stats_summary.loc[1, (col, 'mean')]:.2f}",
                    f"±{stats_summary.loc[1, (col, 'std')]:.2f}"
                )

with tab2:
    st.header("⚙️ Configuração do Modelo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Seleção de Variáveis")
        
        # Lista de features disponíveis (excluindo target e IDs)
        available_features = [col for col in df_processed.columns 
                            if col != target_var and not col.lower().startswith('id')]
        
        # Seleção manual de features
        manual_selection = st.checkbox("✋ Seleção Manual de Variáveis", value=False)
        
        if manual_selection:
            selected_features = st.multiselect(
                "Escolha as variáveis para o modelo:",
                available_features,
                default=available_features[:10] if len(available_features) > 10 else available_features,
                help="Selecione as variáveis que deseja incluir no modelo"
            )
        else:
            selected_features = []
        
        # RFE
        apply_rfe = st.checkbox("🎯 Aplicar RFE (Recursive Feature Elimination)", value=True)
        if apply_rfe:
            n_features = st.slider(
                "Número de features para selecionar:",
                min_value=5,
                max_value=min(20, len(available_features)),
                value=15,
                help="RFE selecionará automaticamente as melhores features"
            )
        else:
            n_features = len(selected_features) if selected_features else len(available_features)
    
    with col2:
        st.subheader("⚖️ Balanceamento e Divisão")
        
        # SMOTE
        if SMOTE_AVAILABLE:
            apply_smote = st.checkbox("⚖️ Aplicar SMOTE", value=True, help="Balancear classes usando SMOTE")
        else:
            apply_smote = False
            st.warning("⚠️ SMOTE não disponível")
        
        # Tamanho do teste
        test_size = st.slider("Tamanho do conjunto de teste:", 0.1, 0.5, 0.3, 0.05)
        
        # Modelos para treinar
        st.subheader("🤖 Modelos para Treinar")
        
        model_groups = {
            "Distância": ['KNN', 'SVM'],
            "Árvores": ['Decision Tree', 'Random Forest'],
            "Boosting": ['AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
            "Linear": ['Logistic Regression']
        }
        
        selected_models = []
        for group_name, models in model_groups.items():
            st.write(f"**{group_name}:**")
            cols = st.columns(len(models))
            for i, model in enumerate(models):
                with cols[i]:
                    if st.checkbox(model, value=(model in ['Random Forest', 'XGBoost', 'Logistic Regression'])):
                        selected_models.append(model)

with tab3:
    st.header("🤖 Treinamento dos Modelos")
    
    if st.button("🚀 Executar Análise", type="primary", use_container_width=True):
        if not selected_models:
            st.error("❌ Selecione pelo menos um modelo para treinar!")
        else:
            with st.spinner("🔄 Processando dados..."):
                # Aplicar pré-processamento
                X_processed, y_processed, final_features = apply_preprocessing(
                    df_processed, target_var, selected_features, apply_smote, n_features
                )
                
                # Verificar se temos dados suficientes e classes balanceadas para train_test_split
                min_class_count = pd.Series(y_processed).value_counts().min()
                
                if min_class_count < 2:
                    st.error("❌ Dados insuficientes para divisão em treino/teste. Tente com mais dados.")
                    st.stop()
                
                # Tentar divisão estratificada, se falhar usar divisão normal
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed, 
                        test_size=test_size, 
                        random_state=42, 
                        stratify=y_processed
                    )
                except ValueError:
                    st.warning("⚠️ Divisão estratificada não possível. Usando divisão aleatória.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed, 
                        test_size=test_size, 
                        random_state=42
                    )
                
                # Armazenar no session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['final_features'] = final_features
                st.session_state['selected_models'] = selected_models
            
            with st.spinner("🤖 Treinando modelos..."):
                results = train_models(X_train, X_test, y_train, y_test, selected_models)
                st.session_state['results'] = results
            
            if results:
                st.success(f"✅ {len(results)} modelos treinados com sucesso!")
                
                # Preview dos resultados
                results_df = pd.DataFrame({
                    'Modelo': list(results.keys()),
                    'AUC': [results[model]['auc'] for model in results.keys()],
                    'Acurácia': [results[model]['accuracy'] for model in results.keys()],
                    'F1-Score': [results[model]['f1_score'] for model in results.keys()]
                }).round(4)
                
                st.dataframe(results_df.sort_values('AUC', ascending=False), use_container_width=True)
            else:
                st.error("❌ Nenhum modelo foi treinado com sucesso!")
    
    # Mostrar features selecionadas se disponível
    if 'final_features' in st.session_state:
        st.subheader("🎯 Features Selecionadas")
        
        features_html = ""
        for feature in st.session_state['final_features']:
            features_html += f'<span class="feature-tag">{feature}</span>'
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 {len(st.session_state['final_features'])} features selecionadas:</h4>
            {features_html}
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.header("📈 Resultados e Análises")
    
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Identificar melhor modelo
        results_df = pd.DataFrame({
            'Modelo': list(results.keys()),
            'Acurácia': [results[model]['accuracy'] for model in results.keys()],
            'Precisão': [results[model]['precision'] for model in results.keys()],
            'Recall': [results[model]['recall'] for model in results.keys()],
            'F1-Score': [results[model]['f1_score'] for model in results.keys()],
            'AUC': [results[model]['auc'] for model in results.keys()]
        }).round(4)
        
        best_model_name = results_df.loc[results_df['AUC'].idxmax(), 'Modelo']
        best_results = results[best_model_name]
        
        # Métricas do melhor modelo
        st.subheader(f"🏆 Melhor Modelo: {best_model_name}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🎯 AUC", f"{best_results['auc']:.3f}")
        with col2:
            st.metric("✅ Acurácia", f"{best_results['accuracy']:.3f}")
        with col3:
            st.metric("🔍 Precisão", f"{best_results['precision']:.3f}")
        with col4:
            st.metric("📊 Recall", f"{best_results['recall']:.3f}")
        with col5:
            st.metric("⚖️ F1-Score", f"{best_results['f1_score']:.3f}")
        
        # Subtabs para diferentes análises
        subtab1, subtab2, subtab3, subtab4 = st.tabs(["📊 Comparação", "📈 ROC", "🎯 Confusão", "🔍 Importância"])
        
        with subtab1:
            # Comparação de modelos
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
                title="📊 Comparação de Métricas por Modelo",
                xaxis_title="Modelos",
                yaxis_title="Score",
                barmode='group',
                height=600
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Tabela detalhada
            st.subheader("📋 Tabela Detalhada de Resultados")
            st.dataframe(
                results_df.sort_values('AUC', ascending=False).style.highlight_max(subset=['AUC'], color='lightgreen'),
                use_container_width=True
            )
        
        with subtab2:
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
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Aleatório',
                line=dict(dash='dash', color='gray', width=2)
            ))
            
            fig_roc.update_layout(
                title='📈 Curvas ROC - Comparação de Modelos',
                xaxis_title='Taxa de Falsos Positivos (FPR)',
                yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
                height=600
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with subtab3:
            # Matriz de confusão
            cm = confusion_matrix(y_test, best_results['y_pred'])
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                title=f'🎯 Matriz de Confusão - {best_model_name}',
                labels=dict(x="Predito", y="Real")
            )
            fig_cm.update_layout(height=500)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Interpretação
            tn, fp, fn, tp = cm.ravel()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="info-box">
                    <h4>📊 Interpretação da Matriz</h4>
                    <ul>
                        <li><strong>Verdadeiros Negativos:</strong> {tn}</li>
                        <li><strong>Falsos Positivos:</strong> {fp}</li>
                        <li><strong>Falsos Negativos:</strong> {fn}</li>
                        <li><strong>Verdadeiros Positivos:</strong> {tp}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Métricas Derivadas</h4>
                    <ul>
                        <li><strong>Acurácia:</strong> {accuracy:.3f}</li>
                        <li><strong>Precisão:</strong> {precision:.3f}</li>
                        <li><strong>Recall:</strong> {recall:.3f}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with subtab4:
            # Importância das features
            best_model = best_results['model']
            
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_names = st.session_state['final_features']
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importância': importances
                }).sort_values('Importância', ascending=True)
                
                fig_importance = px.bar(
                    importance_df.tail(15),
                    x='Importância',
                    y='Feature',
                    orientation='h',
                    title=f'🎯 Top 15 Features Mais Importantes - {best_model_name}',
                    height=600
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Interpretação automatizada
                st.subheader("🧠 Interpretação Automatizada")
                
                top_features = importance_df.tail(3)['Feature'].tolist()
                
                interpretation = f"""
                <div class="info-box">
                    <h4>📊 Análise do Modelo {best_model_name}</h4>
                    <p><strong>Performance Geral:</strong></p>
                    <ul>
                        <li>AUC de {best_results['auc']:.3f} - {'Excelente discriminação' if best_results['auc'] > 0.9 else 'Muito boa discriminação' if best_results['auc'] > 0.8 else 'Boa discriminação'}</li>
                        <li>Acurácia de {best_results['accuracy']:.1%} - O modelo acerta {best_results['accuracy']:.1%} das predições</li>
                        <li>F1-Score de {best_results['f1_score']:.3f} - Bom equilíbrio entre precisão e recall</li>
                    </ul>
                    
                    <p><strong>Principais Fatores de Risco:</strong></p>
                    <ol>
                        <li><strong>{top_features[2]}</strong> - Fator mais importante</li>
                        <li><strong>{top_features[1]}</strong> - Segundo mais importante</li>
                        <li><strong>{top_features[0]}</strong> - Terceiro mais importante</li>
                    </ol>
                </div>
                """
                
                st.markdown(interpretation, unsafe_allow_html=True)
            
            elif hasattr(best_model, 'coef_'):
                coefficients = best_model.coef_[0]
                feature_names = st.session_state['final_features']
                
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coeficiente': coefficients,
                    'Importância': np.abs(coefficients)
                }).sort_values('Importância', ascending=True)
                
                fig_coef = px.bar(
                    coef_df.tail(15),
                    x='Coeficiente',
                    y='Feature',
                    orientation='h',
                    title=f'🎯 Coeficientes do Modelo - {best_model_name}',
                    height=600
                )
                st.plotly_chart(fig_coef, use_container_width=True)
            
            else:
                st.info("ℹ️ Importância de features não disponível para este modelo.")
        
        # Recomendações estratégicas
        st.subheader("💼 Recomendações Estratégicas")
        
        recommendations = f"""
        <div class="success-box">
            <h4>🎯 Ações Recomendadas</h4>
            <ol>
                <li><strong>Implementação Imediata:</strong> Deploy do modelo {best_model_name} em produção</li>
                <li><strong>Segmentação por Risco:</strong>
                    <ul>
                        <li>Alto risco (score > 0.7): Atendimento personalizado</li>
                        <li>Médio risco (0.3-0.7): Programa de retenção</li>
                        <li>Baixo risco (< 0.3): Acompanhamento padrão</li>
                    </ul>
                </li>
                <li><strong>Monitoramento:</strong> Reavaliar modelo mensalmente</li>
                <li><strong>Ação Proativa:</strong> Focar nas variáveis mais importantes identificadas</li>
            </ol>
        </div>
        """
        
        st.markdown(recommendations, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Nenhum Resultado Disponível</h4>
            <p>Execute a análise na aba "Modelagem" para ver os resultados aqui.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Dashboard desenvolvido para a Tarefa 4 - SIEP</strong></p>
    <p><strong>Alunos:</strong> Rafael Leivas Bisi (231013467) | Tiago André Gondim (231013476)</p>
    <p><strong>Professor:</strong> João Gabriel de Moraes Souza</p>
    <p><em>Universidade de Brasília - Engenharia de Produção</em></p>
</div>
""", unsafe_allow_html=True)
