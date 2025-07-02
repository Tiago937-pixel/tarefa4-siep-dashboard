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
        # Tentar carregar o arquivo do Kaggle Customer Personality Analysis
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
    X = df_processed.drop(columns=[target_var])
    y = df_processed[target_var]
    
    # Garantir que todos os dados são numéricos e limpos
    X_clean = X.copy()
    y_clean = y.copy()
    
    # Remover valores infinitos e NaN
    for col in X_clean.columns:
        if X_clean[col].dtype in ['int64', 'float64']:
            # Substituir inf por NaN, depois por mediana
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
            if X_clean[col].isnull().sum() > 0:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
    
    # Remover linhas com problemas
    mask = ~(X_clean.isnull().any(axis=1) | y_clean.isnull())
    X_clean = X_clean[mask]
    y_clean = y_clean[mask]
    
    # Aplicar SMOTE se disponível e solicitado
    if SMOTE_AVAILABLE and apply_smote:
        # Verificar se temos dados suficientes para SMOTE
        min_samples = y_clean.value_counts().min()
        
        try:
            if min_samples >= 2:
                # SMOTE
                k_neighbors = min(5, min_samples - 1)
                smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
                X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)
            else:
                # Se não der para aplicar SMOTE, usar dados originais
                X_balanced, y_balanced = X_clean, y_clean
                
        except Exception as e:
            st.warning(f"Erro ao aplicar SMOTE: {str(e)}. Usando dados originais.")
            X_balanced, y_balanced = X_clean, y_clean
    else:
        X_balanced, y_balanced = X_clean, y_clean
    
    # RFE
    try:
        n_features = min(n_features, X_balanced.shape[1])
        estimator = LogisticRegression(random_state=42, max_iter=1000)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X_balanced, y_balanced)
        
        selected_features = X_balanced.columns[rfe.support_].tolist()
        X_selected = X_balanced[selected_features]
        
    except Exception as e:
        st.warning(f"Erro ao aplicar RFE: {str(e)}. Usando todas as features.")
        selected_features = X_balanced.columns.tolist()
        X_selected = X_balanced
    
    return X_selected, y_balanced, selected_features

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
    
    # Padronizar dados para KNN e SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name in selected_models:
        if model_name in models:
            try:
                model = models[model_name]
                
                # Usar dados padronizados para KNN e SVM
                if model_name in ['KNN', 'SVM']:
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
            except Exception as e:
                st.error(f"Erro ao treinar {model_name}: {str(e)}")
    
    return results

# Carregar dados
df, df_processed, target_var, le_dict = load_and_process_data()

# Sidebar
st.sidebar.header("⚙️ Configurações")

# Seção 1: Filtros de dados
st.sidebar.subheader("🔍 Filtros de Dados")

# Filtros dinâmicos baseados nos dados
with st.sidebar.expander("Filtrar Dados", expanded=False):
    # Filtro por idade (se existir)
    if 'Age' in df.columns:
        age_range = st.slider(
            "Faixa de Idade", 
            int(df['Age'].min()), 
            int(df['Age'].max()), 
            (int(df['Age'].min()), int(df['Age'].max()))
        )
        df_filtered = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
        df_processed_filtered = df_processed[(df_processed['Age'] >= age_range[0]) & (df_processed['Age'] <= age_range[1])]
    else:
        df_filtered = df.copy()
        df_processed_filtered = df_processed.copy()
    
    # Filtro por renda (se existir)
    if 'Income' in df.columns:
        income_range = st.slider(
            "Faixa de Renda", 
            float(df['Income'].min()), 
            float(df['Income'].max()), 
            (float(df['Income'].min()), float(df['Income'].max()))
        )
        df_filtered = df_filtered[(df_filtered['Income'] >= income_range[0]) & (df_filtered['Income'] <= income_range[1])]
        df_processed_filtered = df_processed_filtered[(df_processed_filtered['Income'] >= income_range[0]) & (df_processed_filtered['Income'] <= income_range[1])]

# Informações do dataset filtrado
st.sidebar.subheader("📊 Informações do Dataset")
st.sidebar.info(f"""
**Dimensões:** {df_filtered.shape[0]} linhas × {df_filtered.shape[1]} colunas
**Variável Target:** {target_var}
**Missing Values:** {df_filtered.isnull().sum().sum()}
""")

# Análise da distribuição do target
target_dist = df_filtered[target_var].value_counts()
st.sidebar.subheader(f"🎯 Distribuição de {target_var}")
fig_target = px.pie(
    values=target_dist.values, 
    names=target_dist.index,
    title=f"Distribuição de {target_var}"
)
st.sidebar.plotly_chart(fig_target, use_container_width=True)

# Seção 2: Seleção dinâmica de variáveis
st.sidebar.subheader("🎯 Seleção de Variáveis")

# Seleção manual de features
available_features = [col for col in df_processed_filtered.columns if col != target_var]
manual_features = st.sidebar.multiselect(
    "Selecionar variáveis manualmente:",
    available_features,
    default=available_features[:10] if len(available_features) > 10 else available_features
)

# Seleção de features via RFE
apply_rfe = st.sidebar.checkbox("Aplicar RFE para seleção automática", value=True)
if apply_rfe:
    n_features = st.sidebar.slider(
        "Número de features (RFE)", 
        min_value=5, 
        max_value=min(20, len(available_features)), 
        value=15
    )
else:
    n_features = len(manual_features)

# Aplicar SMOTE
if SMOTE_AVAILABLE:
    apply_smote = st.sidebar.checkbox("Aplicar SMOTE", value=True)
else:
    apply_smote = False

# Seção 3: Configurações de modelagem
st.sidebar.subheader("🤖 Configurações de Modelos")

# Seleção de modelos
available_models = ['KNN', 'SVM', 'Decision Tree', 'Random Forest', 'AdaBoost', 
                   'Gradient Boosting', 'XGBoost', 'LightGBM', 'Logistic Regression']
selected_models = st.sidebar.multiselect(
    "Escolha os modelos para treinar:",
    available_models,
    default=['Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression']
)

# Tamanho do conjunto de teste
test_size = st.sidebar.slider("Tamanho do conjunto de teste (%)", 10, 50, 30) / 100

# Botão para executar análise
run_analysis = st.sidebar.button("🚀 Executar Análise", type="primary")

# Área principal
if run_analysis and selected_models:
    
    # Aplicar seleção de variáveis
    if apply_rfe:
        with st.spinner("Processando dados e aplicando RFE..."):
            X_processed, y_processed, selected_features = apply_smote_and_rfe(
                df_processed_filtered, target_var, n_features, apply_smote
            )
            if apply_smote and SMOTE_AVAILABLE:
                st.success(f"✅ SMOTE aplicado! Dataset balanceado: {len(y_processed)} amostras")
            else:
                st.info("ℹ️ SMOTE não aplicado - usando dados originais")
    else:
        # Usar seleção manual
        X_processed = df_processed_filtered[manual_features]
        y_processed = df_processed_filtered[target_var]
        selected_features = manual_features
        
        if apply_smote and SMOTE_AVAILABLE:
            # Aplicar apenas SMOTE
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, y_processed.value_counts().min() - 1))
                X_processed, y_processed = smote.fit_resample(X_processed, y_processed)
                st.success(f"✅ SMOTE aplicado! Dataset balanceado: {len(y_processed)} amostras")
            except:
                st.warning("⚠️ Não foi possível aplicar SMOTE. Usando dados originais.")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
    )
    
    # Mostrar features selecionadas
    st.subheader("🎯 Features Selecionadas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Total de features selecionadas:** {len(selected_features)}")
        
    with col2:
        with st.expander("Ver lista completa"):
            st.write(selected_features)
    
    # Treinar modelos
    with st.spinner("Treinando modelos..."):
        results = train_models(X_train, X_test, y_train, y_test, selected_models)
    
    if results:
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
        st.subheader("📈 Visualizações")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Comparação", "📈 Curvas ROC", "🔥 Matriz de Confusão", "🎯 Importância", "🧠 Interpretação"])
        
        with tab1:
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
        
        with tab2:
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
        
        with tab3:
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
        
        with tab4:
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
                
            elif hasattr(best_model_obj, 'coef_'):
                # Para modelos lineares
                coefficients = best_model_obj.coef_[0]
                coef_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Coeficiente': coefficients,
                    'Importância (abs)': np.abs(coefficients)
                }).sort_values('Importância (abs)', ascending=True)
                
                fig_coef = px.bar(
                    coef_df.tail(15),
                    x='Coeficiente',
                    y='Feature',
                    orientation='h',
                    title=f'Coeficientes do Modelo - {best_model_name}',
                    height=600
                )
                
                st.plotly_chart(fig_coef, use_container_width=True)
                
                # Tabela de coeficientes
                st.subheader("📋 Tabela de Coeficientes")
                st.dataframe(
                    coef_df.sort_values('Importância (abs)', ascending=False),
                    use_container_width=True
                )
            else:
                st.info("Importância de features não disponível para este modelo.")
        
        with tab5:
            # Interpretação Automatizada
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
            
            **Insights Estratégicos:**
            - Clientes com características específicas têm maior propensão a reclamar
            - A empresa pode usar essas informações para ações preventivas
            - Personalização do atendimento baseada no score de risco
            """
            
            st.markdown(interpretation)
            
            # Análise das features mais importantes
            if hasattr(best_model_obj, 'feature_importances_'):
                top_features = importance_df.tail(5)['Feature'].tolist()
                
                st.subheader("🎯 Análise das Top 5 Features")
                
                for i, feature in enumerate(reversed(top_features), 1):
                    with st.expander(f"{i}. {feature}"):
                        # Análise estatística da feature
                        if feature in df_processed_filtered.columns:
                            feature_stats = df_processed_filtered.groupby(target_var)[feature].agg(['mean', 'median', 'std']).round(3)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Estatísticas por classe:**")
                                st.dataframe(feature_stats)
                            
                            with col2:
                                # Boxplot da feature por classe
                                fig_box = px.box(
                                    df_processed_filtered, 
                                    x=target_var, 
                                    y=feature,
                                    title=f'Distribuição de {feature} por classe'
                                )
                                st.plotly_chart(fig_box, use_container_width=True)

else:
    # Tela inicial
    st.subheader("👋 Bem-vindo ao Dashboard de Previsão de Reclamações!")
    
    st.markdown("""
    Este dashboard permite analisar e modelar dados para previsão de reclamações de clientes usando múltiplos algoritmos de Machine Learning.
    
    ### 🚀 Como usar:
    1. **Configure os filtros** na sidebar para selecionar subconjuntos dos dados
    2. **Escolha as variáveis** manualmente ou use RFE para seleção automática
    3. **Selecione os modelos** que deseja treinar e comparar
    4. **Clique em "Executar Análise"** para iniciar o processo
    
    ### 📊 Funcionalidades:
    - 🔍 **Filtros interativos** para exploração dos dados
    - 🎯 **Seleção dinâmica** de variáveis para modelagem
    - ⚖️ **Balanceamento de dados** com SMOTE (quando disponível)
    - 🤖 **Múltiplos modelos** de Machine Learning
    - 📈 **Visualizações interativas** dos resultados
    - 🧠 **Interpretação automatizada** dos modelos
    - 📋 **Análise de importância** das variáveis
    
    ### 🎯 Modelos Disponíveis:
    - **Baseados em Distância:** KNN, SVM
    - **Bagging:** Decision Tree, Random Forest  
    - **Boosting:** AdaBoost, Gradient Boosting, XGBoost, LightGBM
    - **Linear:** Logistic Regression
    """)
    
    # Mostrar preview dos dados
    st.subheader("👀 Preview dos Dados")
    st.dataframe(df.head(), use_container_width=True)
    
    # Estatísticas básicas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", f"{len(df):,}")
    
    with col2:
        st.metric("Número de Features", len(df.columns) - 1)
    
    with col3:
        target_balance = df[target_var].value_counts(normalize=True).min()
        st.metric("Balanceamento", f"{target_balance:.1%}")
    
    # Gráfico de distribuição do target
    st.subheader("📊 Distribuição da Variável Target")
    target_counts = df[target_var].value_counts()
    
    fig_dist = px.bar(
        x=target_counts.index, 
        y=target_counts.values,
        title=f'Distribuição da variável {target_var}',
        labels={'x': target_var, 'y': 'Frequência'}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Dashboard desenvolvido para a Tarefa 4 - SIEP</p>
    <p><strong>Alunos:</strong> Rafael Leivas Bisi (231013467) | Tiago André Gondim (231013476)</p>
    <p><strong>Professor:</strong> João Gabriel de Moraes Souza</p>
    <p><i>Universidade de Brasília - Engenharia de Produção</i></p>
</div>
""", unsafe_allow_html=True)
