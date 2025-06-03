from sklearn.neural_network import MLPClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve, precision_recall_curve, f1_score,
                           precision_score, recall_score, matthews_corrcoef)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import (RandomUnderSampler, ClusterCentroids, 
                                    NearMiss, TomekLinks)
from imblearn.pipeline import Pipeline as ImbPipeline
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import time
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import os
import json
from datetime import datetime
from fpdf import FPDF
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
from sklearn.calibration import calibration_curve
import sys
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder

# Konfiguracja katalogów
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Konfiguracja Streamlit
st.set_page_config(layout="wide", page_title="Zaawansowana analiza kardiologiczna")
st.title("""🔍 Zaawansowana analiza danych kardiologicznych 
         \n💓 Predykcja chorób serca z interpretacją wyników""")

# 1. Wczytanie i przygotowanie danych
@st.cache_data
def load_data(uploaded_file=None):
    """
    Wczytuje dane z pliku CSV lub domyślnego datasetu.
    
    Parametry:
        uploaded_file (file-like object, opcjonalne): Przesłany plik CSV
        
    Zwraca:
        pd.DataFrame: Wczytane dane lub None jeśli błąd
        
    Przetwarzanie:
        - Konwersja wieku z dni na lata
        - Mapowanie wartości kategorycznych
        - Obliczanie BMI i ciśnienia tętna
    """
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, sep=';')
        except:
            data = pd.read_csv(uploaded_file)
    else:
        try:
            data = pd.read_csv("cardio_train.csv", sep=';')
        except:
            st.warning("Nie znaleziono domyślnego pliku cardio_train.csv. Proszę załadować dane.")
            return None
    
    # Konwersja wieku z dni na lata
    if 'age' in data.columns:
        data['age'] = data['age'] // 365
    
    # Mapowanie płci
    if 'gender' in data.columns:
        data['gender'] = data['gender'].map({1: 'kobieta', 2: 'mężczyzna'})
    
    # Mapowanie cholesterolu i glukozy
    if 'cholesterol' in data.columns:
        data['cholesterol'] = data['cholesterol'].map({1: 'normalny', 2: 'podwyższony', 3: 'wysoki'})
    if 'gluc' in data.columns:
        data['gluc'] = data['gluc'].map({1: 'normalny', 2: 'podwyższony', 3: 'wysoki'})
    
    # BMI
    if all(col in data.columns for col in ['weight', 'height']):
        data['bmi'] = data['weight'] / (data['height']/100)**2
    
    # Ciśnienie tętna
    if all(col in data.columns for col in ['ap_hi', 'ap_lo']):
        data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
    
    return data

# 2. Rozbudowana EDA
def perform_eda(data):
    st.header("📊 1. Eksploracyjna analiza danych (EDA)")
    
    with st.expander("🔍 Pełna analiza danych", expanded=True):
        # Podstawowe informacje
        st.subheader("📌 Podstawowe informacje")
        col1, col2, col3 = st.columns(3)
        col1.metric("Liczba pacjentów", f"{data.shape[0]:,}".replace(",", " "))
        col2.metric("Liczba cech", data.shape[1])
        col3.metric("Brakujące wartości", f"{data.isnull().sum().sum()}")
        
        # Analiza wartości brakujących
        st.subheader("🔎 Analiza wartości brakujących")
        missing = data.isnull().sum()
        if missing.sum() > 0:
            fig = px.bar(missing, title="Brakujące wartości w kolumnach",
                        labels={'index': 'Kolumna', 'value': 'Liczba brakujących'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Brak wartości brakujących w danych")
            
        # Wykrywanie outlierów
        st.subheader("📊 Wykrywanie outlierów")
        num_cols = [col for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure'] if col in data.columns]
        if num_cols:
            fig = px.box(data[num_cols], title="Rozkład cech numerycznych - wykrywanie outlierów")
            st.plotly_chart(fig, use_container_width=True)
        
        # Wyświetlanie danych
        if st.checkbox("Pokaż surowe dane"):
            sample_size = st.slider("Liczba wierszy do wyświetlenia", 100, 5000, 1000, 100)
            st.dataframe(data.sample(min(sample_size, len(data))).style.background_gradient(cmap='Blues'))
        
        # Statystyki opisowe
        st.subheader("📈 Statystyki opisowe")
        st.dataframe(data.describe().T.style.background_gradient(cmap='YlOrBr'))
        
        # Rozkład zmiennej docelowej
        if 'cardio' in data.columns:
            st.subheader("🎯 Rozkład zmiennej docelowej (cardio)")
            fig = px.pie(data, names='cardio', title='Rozkład chorób serca w badanej populacji',
                        color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
            
            # Rozkład klas w podgrupach
            st.subheader("👥 Rozkład klas w podgrupach")
            group_col = st.selectbox("Wybierz cechę do grupowania", 
                                   [col for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'] if col in data.columns])
            if group_col:
                fig = px.histogram(data, x=group_col, color='cardio', barmode='group',
                                 title=f'Rozkład chorób serca wg {group_col}',
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
        
        # Rozkłady cech numerycznych
        st.subheader("📉 Rozkłady cech numerycznych")
        num_cols = [col for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure'] if col in data.columns]
        if num_cols:
            selected_num_col = st.selectbox("Wybierz cechę numeryczną", num_cols)
            
            fig = px.histogram(data, x=selected_num_col, color='cardio' if 'cardio' in data.columns else None, 
                              marginal="box", nbins=50, barmode='overlay', opacity=0.7,
                              title=f'Rozkład {selected_num_col}',
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rozkłady cech kategorycznych
        st.subheader("📊 Rozkłady cech kategorycznych")
        cat_cols = [col for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'] if col in data.columns]
        if cat_cols:
            selected_cat_col = st.selectbox("Wybierz cechę kategoryczną", cat_cols)
            
            fig = px.bar(data, x=selected_cat_col, color='cardio' if 'cardio' in data.columns else None, barmode='group',
                        title=f'Rozkład {selected_cat_col}',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        # Analiza korelacji
        st.subheader("🔗 Analiza korelacji")
        numeric_cols = [col for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure', 'cardio'] 
                       if col in data.columns and data[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            numeric_data = data[numeric_cols].copy()
            corr = numeric_data.corr()
            
            fig = px.imshow(corr, text_auto=True, aspect="auto", 
                           color_continuous_scale='RdBu_r',
                           title='Macierz korelacji cech numerycznych')
            st.plotly_chart(fig, use_container_width=True)
        
        # Interaktywna analiza zależności
        st.subheader("📌 Interaktywna analiza zależności")
        x_axis = st.selectbox("Oś X", num_cols if num_cols else [], index=0)
        y_axis = st.selectbox("Oś Y", num_cols if num_cols else [], index=1)
        color_options = ['cardio'] + cat_cols if 'cardio' in data.columns else cat_cols
        color_by = st.selectbox("Kolor wg", color_options if color_options else [])
        
        if x_axis and y_axis:
            fig = px.scatter(data.sample(10000) if len(data) > 10000 else data,
                            x=x_axis, y=y_axis, color=color_by if color_by else None,
                            marginal_x="box", marginal_y="violin",
                            title=f'{x_axis} vs {y_axis}' + (f' wg {color_by}' if color_by else ''),
                            color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig, use_container_width=True)

# 3. Zaawansowany preprocessing
def perform_preprocessing(data):
    st.header("⚙️ 2. Przygotowanie danych (Preprocessing)")
    
    if 'cardio' not in data.columns:
        st.error("Brak kolumny docelowej 'cardio' w danych. Nie można kontynuować.")
        return None
    
    with st.expander("🔧 Konfiguracja preprocessingu", expanded=True):
        # Podział na cechy i target
        X = data.drop(['cardio'], axis=1)
        if 'id' in X.columns:
            X = X.drop(['id'], axis=1)
        y = data['cardio']
        
        # Definicja kolumn
        numeric_features = [col for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure'] if col in X.columns]
        categorical_features = [col for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'] if col in X.columns]
        
        # Obsługa wartości brakujących
        st.subheader("Obsługa wartości brakujących")
        missing_strategy_num = st.selectbox(
            "Strategia dla wartości brakujących (numeryczne)",
            ['median', 'mean', 'constant', 'usun'])
        
        missing_strategy_cat = st.selectbox(
            "Strategia dla wartości brakujących (kategoryczne)",
            ['most_frequent', 'constant', 'usun'])
        
        # Wybór metod skalowania
        st.subheader("Skalowanie cech numerycznych")
        scaling_method = st.selectbox(
            "Wybierz metodę skalowania",
            ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Brak skalowania'])
        
        # Transformacje dla cech numerycznych
        numeric_transformer_steps = []
        
        if missing_strategy_num != 'usun':
            numeric_transformer_steps.append(('imputer', SimpleImputer(strategy=missing_strategy_num)))
        
        if scaling_method == 'StandardScaler':
            numeric_transformer_steps.append(('scaler', StandardScaler()))
        elif scaling_method == 'MinMaxScaler':
            numeric_transformer_steps.append(('scaler', MinMaxScaler()))
        elif scaling_method == 'RobustScaler':
            numeric_transformer_steps.append(('scaler', RobustScaler()))
        
        numeric_transformer = Pipeline(steps=numeric_transformer_steps) if numeric_transformer_steps else None
        
        # Transformacje dla cech kategorycznych
        st.subheader("Kodowanie cech kategorycznych")
        encoder_type = st.selectbox(
            "Wybierz metodę kodowania",
            ['One-Hot Encoding', 'Ordinal Encoding', 'Target Encoding'],
            index=0
        )
        
        categorical_transformer_steps = []
        if missing_strategy_cat != 'usun':
            categorical_transformer_steps.append(('imputer', SimpleImputer(strategy=missing_strategy_cat, fill_value='brak')))
        
        # Wybór enkodera
        if encoder_type == 'One-Hot Encoding':
            categorical_transformer_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
        elif encoder_type == 'Ordinal Encoding':
            categorical_transformer_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        elif encoder_type == 'Target Encoding':
            categorical_transformer_steps.append(('encoder', TargetEncoder(target_type='binary')))
        
        categorical_transformer = Pipeline(steps=categorical_transformer_steps)
        
        # Połączenie transformacji
        transformers = []
        if numeric_features and numeric_transformer:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        
        # Balansowanie klas
        st.subheader("Balansowanie klas")
        balancing_method = st.selectbox(
            "Wybierz metodę balansowania klas",
            ['Brak', 'SMOTE (oversampling)', 'ADASYN (oversampling)', 
            'BorderlineSMOTE', 'RandomUnderSampler', 'ClusterCentroids',
            'TomekLinks', 'NearMiss', 'SMOTEENN (hybrydowe)'])

        if balancing_method == 'SMOTE (oversampling)':
            resampler = SMOTE(random_state=42)
        elif balancing_method == 'ADASYN (oversampling)':
            resampler = ADASYN(random_state=42)
        elif balancing_method == 'BorderlineSMOTE':
            resampler = BorderlineSMOTE(random_state=42)
        elif balancing_method == 'RandomUnderSampler':
            resampler = RandomUnderSampler(random_state=42)
        elif balancing_method == 'ClusterCentroids':
            resampler = ClusterCentroids(random_state=42)
        elif balancing_method == 'TomekLinks':
            resampler = TomekLinks()
        elif balancing_method == 'NearMiss':
            resampler = NearMiss(version=3)
        elif balancing_method == 'SMOTEENN (hybrydowe)':
            from imblearn.combine import SMOTEENN
            resampler = SMOTEENN(random_state=42)
        else:
            resampler = None
        
        # Selekcja cech
        st.subheader("Selekcja cech")
        feature_selection_method = st.selectbox(
            "Wybierz metodę selekcji cech",
            ['Brak', 'SelectFromModel (Lasso)', 'RFE', 'SelectKBest (chi2)', 'PCA', 'Permutation Importance'])
        
        if feature_selection_method == 'SelectFromModel (Lasso)':
            feature_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
        elif feature_selection_method == 'RFE':
            feature_selector = RFE(RandomForestClassifier(random_state=42))
        elif feature_selection_method == 'SelectKBest (chi2)':
            k_features = st.slider("Liczba cech do wyboru", 1, len(numeric_features + categorical_features), 
                                 min(10, len(numeric_features + categorical_features)))
            feature_selector = SelectKBest(chi2, k=k_features)
        elif feature_selection_method == 'PCA':
            n_components = st.slider("Liczba komponentów PCA", 1, len(numeric_features), min(5, len(numeric_features)))
            feature_selector = PCA(n_components=n_components, random_state=42)
        elif feature_selection_method == 'Permutation Importance':
            feature_selector = 'permutation'
        else:
            feature_selector = None
        
        # Podział na zbiór treningowy i testowy
        st.subheader("Podział danych")
        test_size = st.slider("Rozmiar zbioru testowego", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Ziarno losowości", min_value=0, value=42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        preprocessing_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'resampler': resampler,
            'feature_selector': feature_selector,
            'feature_selection_method': feature_selection_method,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'scaling_method': scaling_method,
            'balancing_method': balancing_method,
            'missing_strategy_num': missing_strategy_num,
            'missing_strategy_cat': missing_strategy_cat
        }
        
        st.session_state['preprocessing_data'] = preprocessing_data
        
        if st.button("Zastosuj preprocessing"):
            with st.spinner("Przetwarzanie danych..."):
                time.sleep(2)
                st.success("Preprocessing zakończony pomyślnie!")
                return preprocessing_data

# 4. Zaawansowane modelowanie
def generate_model_plots(model, model_name, preprocessing_data):
    """Funkcja generująca wszystkie wykresy dla modelu"""
    
    # Predykcje
    y_pred = model.predict(preprocessing_data['X_test'])
    y_proba = model.predict_proba(preprocessing_data['X_test'])[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Macierz pomyłek
    st.subheader("Macierz pomyłek")
    cm = confusion_matrix(preprocessing_data['y_test'], y_pred)
    fig = px.imshow(cm, text_auto=True, 
                   labels=dict(x="Predykcja", y="Rzeczywistość", color="Liczba"),
                   x=['Brak choroby', 'Choroba'],
                   y=['Brak choroby', 'Choroba'],
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Krzywa ROC
    if y_proba is not None:
        st.subheader("Krzywa ROC")
        fpr, tpr, _ = roc_curve(preprocessing_data['y_test'], y_proba)
        roc_auc = roc_auc_score(preprocessing_data['y_test'], y_proba)
        fig = px.area(x=fpr, y=tpr, 
                     labels=dict(x="False Positive Rate", y="True Positive Rate"),
                     title=f'Krzywa ROC (AUC = {roc_auc:.3f})')
        fig.add_shape(type='line', line=dict(dash='dash'),
                     x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Krzywa Precision-Recall
    if y_proba is not None:
        st.subheader("Krzywa Precision-Recall")
        precision_curve, recall_curve, _ = precision_recall_curve(
            preprocessing_data['y_test'], y_proba)
        fig = px.area(x=recall_curve, y=precision_curve,
                     labels=dict(x="Recall", y="Precision"),
                     title=f'Krzywa Precision-Recall (AP = {np.mean(precision_curve):.3f})')
        st.plotly_chart(fig, use_container_width=True)
    
    # Ważność cech
    if model_name in st.session_state.get('feature_importances', {}):
        st.subheader("Ważność cech")
        feature_imp = st.session_state['feature_importances'][model_name]
        fig = px.bar(feature_imp.head(20), x='Importance', y='Feature', 
                    orientation='h', title='Top 20 najważniejszych cech',
                    color='Importance', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Krzywa uczenia (dla wczytanych modeli nie mamy danych, więc pomijamy)
    st.warning("Krzywa uczenia nie jest dostępna dla wczytanych modeli - wymaga danych z procesu treningowego")
    
    # Parametry modelu
    if hasattr(model, 'best_params_'):
        st.subheader("Parametry modelu")
        st.json(model.best_params_)
    elif hasattr(model.named_steps['classifier'], 'get_params'):
        st.subheader("Parametry modelu")
        st.json(model.named_steps['classifier'].get_params())

def perform_modeling(preprocessing_data):
    st.header("🤖 3. Zaawansowane modelowanie")
    
    # Wybór czy trenować nowe modele czy użyć istniejących
    train_option = st.radio("Wybierz opcję:", 
                           ["Wytrenuj nowe modele", "Użyj wcześniej wytrenowanych modeli"],
                           horizontal=True)
    
    if train_option == "Użyj wcześniej wytrenowanych modeli":
        model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
        
        if not model_files:
            st.warning("Brak zapisanych modeli. Proszę najpierw wytrenować modele.")
            return
        
        selected_models = st.multiselect("Wybierz modele do wczytania", 
                                       [f.replace("_model.pkl", "").replace("_", " ") for f in model_files])
     
        if st.button("Wczytaj wybrane modele"):
            trained_models = {}
            model_metadata = {}
            results = []
            feature_importances = {}
            
            for model_name in selected_models:
                try:
                    # 1. Wczytaj model z metadanymi
                    model_path = os.path.join("models", f"{model_name.replace(' ', '_')}_model.pkl")
                    model_data = joblib.load(model_path)
                    
                    # Walidacja struktury
                    # if not isinstance(model_data, dict) or 'model' not in model_data:
                    #     st.error(f"Nieprawidłowy format pliku modelu {model_name}")
                    #     continue


                    # if not isinstance(model_data, dict) :
                    #     st.error(f"Nieprawidłowy format pliku modelu {model_name}")
                    #     continue
                
                        
                    # trained_models[model_name] = model_data['model']
                    if isinstance(model_data, dict) and 'model' in model_data:
                        trained_models[model_name] = model_data['model']
                    else:
                        trained_models[model_name] = model_data

                    # 2. Wczytaj pełne wyniki z pliku JSON
                    results_path = os.path.join("results", f"{model_name.replace(' ', '_')}_full_results.json")
                    if os.path.exists(results_path):
                        with open(results_path, "r") as f:
                            result_data = json.load(f)
                            
                            # Zapisz metadane
                            model_metadata[model_name] = result_data.get('metadata', {})
                            
                            # Zapisz wyniki performance
                            results.append({
                                'Model': model_name,
                                'Accuracy': result_data['performance_metrics'].get('Accuracy'),
                                'ROC-AUC': result_data['performance_metrics'].get('ROC-AUC'),
                                'F1-Score': result_data['performance_metrics'].get('F1-Score'),
                                'Precision': result_data['performance_metrics'].get('Precision'),
                                'Recall': result_data['performance_metrics'].get('Recall'),
                                'Training Time': f"{model_metadata[model_name].get('training_time_seconds', 'N/A')} s",
                                'Params': str(model_metadata[model_name].get('model_parameters', 'N/A'))
                            })
                            
                            # Zapisz ważność cech jeśli istnieje
                            if result_data.get('feature_importances'):
                                try:
                                    feature_importances[model_name] = pd.DataFrame.from_dict(
                                        result_data['feature_importances'])
                                except Exception as e:
                                    st.warning(f"Problem z konwersją ważności cech dla {model_name}: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Błąd podczas wczytywania modelu {model_name}: {str(e)}")
                    continue
            
            # Zapisz w sesji
            st.session_state['trained_models'] = trained_models
            st.session_state['model_metadata'] = model_metadata
            st.session_state['model_results'] = pd.DataFrame(results)
            st.session_state['feature_importances'] = feature_importances
                        
            # Wygeneruj wykresy dla każdego modelu
            model_tabs = st.tabs([f"Model {i+1}: {name}" for i, name in enumerate(selected_models)])
            
            for i, (model_name, model_tab) in enumerate(zip(selected_models, model_tabs)):
                with model_tab:
                    model = trained_models[model_name]
                    
                    # Wyświetl podstawowe informacje
                    st.subheader(f"Model: {model_name}")
                    
                    # Pobierz metryki
                    model_result = next((r for r in results if r.get('Model') == model_name), {})
                    
                    # Wyświetl metryki
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{model_result.get('Accuracy', 'N/A'):.3f}")
                        st.metric("F1-Score", f"{model_result.get('F1-Score', 'N/A'):.3f}")
                    with col2:
                        st.metric("ROC-AUC", f"{model_result.get('ROC-AUC', 'N/A'):.3f}")
                        st.metric("Precision", f"{model_result.get('Precision', 'N/A'):.3f}")
                    
                    # Generuj wykresy używając tej samej funkcji co dla nowych modeli
                    generate_model_plots(model, model_name, st.session_state['preprocessing_data'])
                    
                    # Wyświetl parametry modelu jeśli dostępne
                    if model_name in model_metadata and 'model_parameters' in model_metadata[model_name]:
                        st.subheader("Parametry modelu")
                        st.json(model_metadata[model_name]['model_parameters'])
            
            st.success(f"Pomyślnie wczytano {len(selected_models)} modeli!")
            return
    
    # Konfiguracja modeli - tylko jeśli wybrano trenowanie nowych
    with st.expander("⚙️ Konfiguracja modeli", expanded=True):
        # Definicja modeli z parametrami
        model_options = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True, False]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'scale_pos_weight': [1, (len(preprocessing_data['y_train']) / sum(preprocessing_data['y_train'])) if sum(preprocessing_data['y_train']) > 0 else 1]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200],
                    'num_leaves': [31, 63],
                    'learning_rate': [0.01, 0.1],
                    'feature_fraction': [0.8, 1.0]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced'),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'K-NN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': list(range(3, 21, 2)),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(verbose=0, random_state=42),
                'params': {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.05],
                    'depth': [4, 6]
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 20],
                    'min_samples_split': [2, 10],
                    'min_samples_leaf': [1, 4]
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {}
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'MLP (Neural Network)': {
                'model': MLPClassifier(random_state=42, early_stopping=True),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Wybór modeli
        selected_models = st.multiselect(
            "Wybierz modele do porównania (zalecane max 3-4 dla wydajności)",
            list(model_options.keys()),
            default=['Random Forest', 'XGBoost', 'Logistic Regression'])
        
        # Zaawansowane opcje
        st.subheader("Zaawansowane opcje trenowania")
        cv_folds = st.slider("Liczba foldów walidacji krzyżowej", 3, 10, 5)
        scoring_metric = st.selectbox(
            "Metryka optymalizacji",
            ['roc_auc', 'f1', 'accuracy', 'precision', 'recall'],
            index=0)
        
        # Wybór metody optymalizacji hiperparametrów
        optimization_method = st.selectbox(
            "Metoda optymalizacji hiperparametrów",
            ['RandomizedSearchCV', 'GridSearchCV', 'Brak optymalizacji'],
            index=0)
        
        # Opcja kalibracji modeli
        calibrate_models = st.checkbox("Kalibruj modele (dla lepszych prawdopodobieństw)", value=False)
    
    if st.button("Uruchom modelowanie") and train_option == "Wytrenuj nowe modele":
        if not selected_models:
            st.warning("Proszę wybrać co najmniej jeden model")
            return
            
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Utworzenie tabów dla każdego modelu
        model_tabs = st.tabs([f"Model {i+1}: {name}" for i, name in enumerate(selected_models)])
        
        results = []
        trained_models = {}
        feature_importances = {}
        all_metadata = {}
        
        for i, (model_name, model_tab) in enumerate(zip(selected_models, model_tabs)):
            with model_tab:
                status_text.text(f"Trenowanie modelu {model_name}...")
                progress_bar.progress((i + 1) / len(selected_models))
                model_start_time = time.time()
                    
                # Budowa pipeline'u
                preprocessor = preprocessing_data['preprocessor']
                model_info = model_options[model_name]
                model = model_info['model']
                params = model_info['params']
                
                if preprocessing_data['resampler'] is not None:
                    steps = [
                        ('preprocessor', preprocessor),
                        ('resampler', preprocessing_data['resampler'])
                    ]
                    
                    if preprocessing_data['feature_selector'] is not None and preprocessing_data['feature_selector'] != 'permutation':
                        steps.append(('feature_selector', preprocessing_data['feature_selector']))
                    
                    steps.append(('classifier', model))
                    pipeline = ImbPipeline(steps)
                else:
                    steps = [('preprocessor', preprocessor)]
                    
                    if preprocessing_data['feature_selector'] is not None and preprocessing_data['feature_selector'] != 'permutation':
                        steps.append(('feature_selector', preprocessing_data['feature_selector']))
                    
                    steps.append(('classifier', model))
                    pipeline = Pipeline(steps)
                
                # Optymalizacja hiperparametrów
                if optimization_method == 'GridSearchCV' and params:
                    param_grid = {f'classifier__{key}': value for key, value in params.items()}
                    search = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring=scoring_metric, n_jobs=-1, verbose=1)
                elif optimization_method == 'RandomizedSearchCV' and params:
                    param_grid = {f'classifier__{key}': value for key, value in params.items()}
                    search = RandomizedSearchCV(pipeline, param_grid, cv=cv_folds, scoring=scoring_metric, 
                                            n_iter=10, random_state=42, n_jobs=-1, verbose=1)
                else:
                    search = pipeline
                
                # Trenowanie modelu
                search.fit(preprocessing_data['X_train'], preprocessing_data['y_train'])
                
                if hasattr(search, 'best_estimator_'):
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    best_model = search
                    best_params = "Brak optymalizacji"
                
                # Kalibracja modelu
                if calibrate_models and hasattr(best_model.named_steps['classifier'], 'predict_proba'):
                    from sklearn.calibration import CalibratedClassifierCV
                    calibrated_model = CalibratedClassifierCV(best_model.named_steps['classifier'], cv='prefit', method='isotonic')
                    
                    # Przetworzenie danych treningowych
                    X_train_processed = best_model[:-1].transform(preprocessing_data['X_train'])
                    calibrated_model.fit(X_train_processed, preprocessing_data['y_train'])
                    
                    # Zastąpienie oryginalnego modelu skalibrowanym
                    best_model.named_steps['classifier'] = calibrated_model
                
                # Predykcje
                y_pred = best_model.predict(preprocessing_data['X_test'])
                y_proba = best_model.predict_proba(preprocessing_data['X_test'])[:, 1] if hasattr(best_model, 'predict_proba') else None
                
                # Obliczenie metryk
                accuracy = accuracy_score(preprocessing_data['y_test'], y_pred)
                roc_auc = roc_auc_score(preprocessing_data['y_test'], y_proba) if y_proba is not None else None
                f1 = f1_score(preprocessing_data['y_test'], y_pred)
                precision = precision_score(preprocessing_data['y_test'], y_pred)
                recall = recall_score(preprocessing_data['y_test'], y_pred)
                mcc = matthews_corrcoef(preprocessing_data['y_test'], y_pred)
                
                # Macierz pomyłek
                cm = confusion_matrix(preprocessing_data['y_test'], y_pred)
                
                # Ważność cech
                feature_imp = None
                if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                    try:
                        if preprocessing_data['feature_selector'] is not None:
                            if preprocessing_data['feature_selection_method'] == 'PCA':
                                features = [f"PC{i+1}" for i in range(preprocessing_data['feature_selector'].n_components_)]
                            else:
                                features = preprocessing_data['X_train'].columns[
                                    best_model.named_steps['feature_selector'].get_support()]
                        else:
                            features = preprocessing_data['X_train'].columns
                        
                        importances = best_model.named_steps['classifier'].feature_importances_
                        feature_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
                        feature_imp = feature_imp.sort_values('Importance', ascending=False)
                        feature_importances[model_name] = feature_imp
                    except Exception as e:
                        st.warning(f"Nie udało się wyodrębnić ważności cech: {str(e)}")
                
                # Zapisz wyniki
                model_result = {
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'ROC-AUC': roc_auc if roc_auc is not None else 'N/A',
                    'F1-Score': f1,
                    'Precision': precision,
                    'Recall': recall,
                    'MCC': mcc,
                    'Params': str(best_params)
                }
                results.append(model_result)
                
                # Oblicz czas treningu
                training_time = time.time() - model_start_time
                
                # Stwórz metadane
                model_metadata = {
                    'model_name': model_name,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'training_time_seconds': round(training_time, 2),
                    'preprocessing': {
                        'numeric_features': preprocessing_data['numeric_features'],
                        'categorical_features': preprocessing_data['categorical_features'],
                        'scaling_method': preprocessing_data['scaling_method'],
                        'balancing_method': preprocessing_data['balancing_method'],
                        'missing_strategy_num': preprocessing_data['missing_strategy_num'],
                        'missing_strategy_cat': preprocessing_data['missing_strategy_cat'],
                        'feature_selection_method': preprocessing_data['feature_selection_method'],
                        'test_size': preprocessing_data.get('test_size', 0.2)
                    },
                    'training_parameters': {
                        'optimization_method': optimization_method,
                        'scoring_metric': scoring_metric,
                        'cv_folds': cv_folds,
                        'calibration': calibrate_models
                    },
                    'model_parameters': best_params if hasattr(search, 'best_params_') else model.get_params(),
                    'environment': {
                        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                        'libraries': {
                            'sklearn': sklearn.__version__,
                            'pandas': pd.__version__,
                            'numpy': np.__version__,
                            'streamlit': st.__version__
                        }
                    }
                }
                all_metadata[model_name] = model_metadata
                
                # Zapisz model i metadane
                model_path = os.path.join("models", f"{model_name.replace(' ', '_')}_model.pkl")
                joblib.dump({
                    'model': best_model,
                    'metadata': model_metadata,
                    'results': model_result,
                    'feature_importances': feature_imp.to_dict() if feature_imp is not None else None
                }, model_path)
                
                def convert_to_serializable(obj):
                    """Konwertuje obiekt do postaci możliwej do serializacji do JSON."""
                    # Obsługa podstawowych typów numerycznych
                    if isinstance(obj, (np.integer, np.int64, int)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, float)):
                        return float(obj)
                    elif isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    
                    # Obsługa tablic i struktur danych
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict()
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, (list, tuple, set)):
                        return [convert_to_serializable(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    
                    # Specjalna obsługa obiektów scikit-learn i keras
                    elif isinstance(obj, (ColumnTransformer, Pipeline, BaseEstimator)):
                        return {
                            'type': str(type(obj)),
                            'params': convert_to_serializable(obj.get_params()),
                            'steps': [str(step) for step in getattr(obj, 'steps', [])]
                        }
                    
                    # Obsługa specjalnych typów Pythona
                    elif str(type(obj)) in ("<class 'mappingproxy'>", "<class 'wrapper_descriptor'>", 
                                        "<class 'method_descriptor'>"):
                        return str(obj)  # Zwracamy reprezentację stringową
                    
                    # Obsługa obiektów z __dict__
                    elif hasattr(obj, '__dict__'):
                        return convert_to_serializable(obj.__dict__)
                    
                    # Obsługa obiektów z get_params()
                    elif hasattr(obj, 'get_params'):
                        return convert_to_serializable(obj.get_params())
                    
                    # Domyślna obsługa - konwersja na string
                    else:
                        try:
                            return str(obj)
                        except Exception:
                            return None  # W ostateczności zwracamy None


                full_results = {
                    'metadata': convert_to_serializable(model_metadata),
                    'performance_metrics': convert_to_serializable(model_result),
                    'feature_importances': convert_to_serializable(feature_imp.to_dict() if feature_imp is not None else None),
                    'confusion_matrix': convert_to_serializable(cm.tolist()),
                    'training_parameters': {
                        'X_train_shape': convert_to_serializable(preprocessing_data['X_train'].shape),
                        'y_train_distribution': convert_to_serializable(dict(pd.Series(preprocessing_data['y_train']).value_counts())),
                        'X_test_shape': convert_to_serializable(preprocessing_data['X_test'].shape),
                        'y_test_distribution': convert_to_serializable(dict(pd.Series(preprocessing_data['y_test']).value_counts()))
                    }
                }

                
                results_path = os.path.join("results", f"{model_name.replace(' ', '_')}_full_results.json")
                with open(results_path, "w") as f:
                    json.dump(full_results, f, indent=4)
                
                # Wizualizacja wyników
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Macierz pomyłek")
                    fig = px.imshow(cm, text_auto=True, 
                                   labels=dict(x="Predykcja", y="Rzeczywistość", color="Liczba"),
                                   x=['Brak choroby', 'Choroba'],
                                   y=['Brak choroby', 'Choroba'],
                                   color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if y_proba is not None:
                        st.subheader("Krzywa ROC")
                        fpr, tpr, _ = roc_curve(preprocessing_data['y_test'], y_proba)
                        fig = px.area(x=fpr, y=tpr, 
                                     labels=dict(x="False Positive Rate", y="True Positive Rate"),
                                     title=f'Krzywa ROC (AUC = {roc_auc:.3f})' if roc_auc is not None else 'Krzywa ROC')
                        fig.add_shape(type='line', line=dict(dash='dash'),
                                     x0=0, x1=1, y0=0, y1=1)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Krzywa Precision-Recall
                if y_proba is not None:
                    st.subheader("Krzywa Precision-Recall")
                    precision_curve, recall_curve, _ = precision_recall_curve(
                        preprocessing_data['y_test'], y_proba)
                    fig = px.area(x=recall_curve, y=precision_curve,
                                 labels=dict(x="Recall", y="Precision"),
                                 title=f'Krzywa Precision-Recall (AP = {np.mean(precision_curve):.3f})')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Krzywa kalibracji
                if calibrate_models and y_proba is not None:
                    st.subheader("Krzywa kalibracji")
                    prob_true, prob_pred = calibration_curve(preprocessing_data['y_test'], y_proba, n_bins=10)
                    fig = px.line(x=prob_pred, y=prob_true, 
                                 labels={'x': 'Średnie przewidywane prawdopodobieństwo', 
                                         'y': 'Frakcja pozytywnych przypadków'},
                                 title='Krzywa kalibracji')
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                 x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Ważność cech
                if feature_imp is not None:
                    st.subheader("Ważność cech")
                    fig = px.bar(feature_imp.head(20), x='Importance', y='Feature', 
                                orientation='h', title='Top 20 najważniejszych cech',
                                color='Importance', color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Krzywa uczenia
                st.subheader("Krzywa uczenia")
                try:
                    train_sizes, train_scores, test_scores = learning_curve(
                        best_model, preprocessing_data['X_train'], preprocessing_data['y_train'], 
                        cv=5, scoring='accuracy', n_jobs=-1)
                    
                    train_mean = np.mean(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, name='Training score'))
                    fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, name='Cross-validation score'))
                    fig.update_layout(title='Krzywa uczenia', 
                                    xaxis_title='Training size', 
                                    yaxis_title='Accuracy')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Nie udało się wygenerować krzywej uczenia: {str(e)}")
                
                # Parametry modelu
                st.subheader("Parametry modelu")
                st.json(best_params if hasattr(search, 'best_params_') else model.get_params())
                
                st.success(f"Model {model_name} zapisany! Czas treningu: {training_time:.2f}s")
        
        # Zapisz podsumowanie całego procesu
        summary = {
            'total_training_time': round(time.time() - start_time, 2),
            'models_trained': selected_models,
            'average_training_time': round((time.time() - start_time) / len(selected_models), 2),
            'preprocessing_config': {
                k: convert_to_serializable(v) 
                for k, v in preprocessing_data.items() 
                if k not in ['X_train', 'X_test', 'y_train', 'y_test']
            },
            'training_parameters': {
                'optimization_method': optimization_method,
                'scoring_metric': scoring_metric,
                'cv_folds': cv_folds,
                'calibration': calibrate_models
            },
            'data_shapes': {
                'X_train_shape': convert_to_serializable(preprocessing_data['X_train'].shape),
                'X_test_shape': convert_to_serializable(preprocessing_data['X_test'].shape),
                'y_train_distribution': convert_to_serializable(
                    dict(pd.Series(preprocessing_data['y_train']).value_counts())
                ),
                'y_test_distribution': convert_to_serializable(
                    dict(pd.Series(preprocessing_data['y_test']).value_counts())
                )
            }
        }

        summary_path = os.path.join("results", "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(convert_to_serializable(summary), f, indent=4)
                
        # Zapisz wytrenowane modele w sesji
        st.session_state['trained_models'] = trained_models
        st.session_state['model_metadata'] = all_metadata
        st.session_state['model_results'] = pd.DataFrame(results)
        st.session_state['feature_importances'] = feature_importances
        
        # Podsumowanie wyników
        st.header("📊 Podsumowanie wyników")
        results_df = pd.DataFrame(results)
        
        # Formatowanie wyników
        metrics = ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'MCC']
        for metric in metrics:
            if metric in results_df.columns:
                results_df[metric] = results_df[metric].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
        
        # Wyświetlanie wyników
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tabela wyników")
            st.dataframe(results_df.set_index('Model'))
        
        with col2:
            st.subheader("Ranking modeli")
            rank_df = results_df.copy()
            
            # Konwersja metryk na liczby (pomijając 'N/A')
            for metric in metrics:
                if metric in rank_df.columns:
                    rank_df[metric] = rank_df[metric].apply(lambda x: float(x) if x != 'N/A' else np.nan)
            
            # Obliczenie rankingu tylko dla dostępnych metryk
            available_metrics = [m for m in metrics if m in rank_df.columns and not rank_df[m].isna().all()]
            
            for metric in available_metrics:
                rank_df[f'{metric}_rank'] = rank_df[metric].rank(ascending=False)
            
            if available_metrics:
                rank_df['Średni ranking'] = rank_df[[f'{m}_rank' for m in available_metrics]].mean(axis=1)
                rank_df = rank_df.sort_values('Średni ranking').reset_index(drop=True)
                st.dataframe(rank_df[['Model'] + available_metrics + ['Średni ranking']].set_index('Model'))
        
        # Wykres porównawczy
        st.subheader("📈 Porównanie modeli")
        if available_metrics:
            metric_to_compare = st.selectbox("Wybierz metrykę do porównania", available_metrics)
            
            fig = px.bar(results_df, x='Model', y=metric_to_compare, 
                         color='Model', text=metric_to_compare,
                         title=f'Porównanie modeli wg {metric_to_compare}',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        # Wyświetl podsumowanie treningu
        with st.expander("📋 Podsumowanie procesu treningowego", expanded=False):
            st.write(f"**Całkowity czas treningu:** {summary['total_training_time']:.2f}s")
            st.write(f"**Średni czas na model:** {summary['average_training_time']:.2f}s")
            st.write("**Wytrenowane modele:**")
            st.write(selected_models)
            
            st.write("**Parametry treningu:**")
            st.json(summary['training_parameters'])



# 5. Analiza reguł asocjacyjnych
def perform_association_rules(data):
    st.header("🔗 Analiza reguł asocjacyjnych")
    
    if 'cardio' not in data.columns:
        st.warning("Brak kolumny docelowej 'cardio' w danych. Nie można przeprowadzić analizy.")
        return
    
    with st.expander("⚙️ Konfiguracja analizy", expanded=True):
        # Przygotowanie danych
        bin_data = data.copy()
        
        # Dyskretyzacja wieku
        st.subheader("Podział na przedziały wiekowe")
        age_bin1 = st.slider("Pierwsza granica wieku", 30, 70, 40)
        age_bin2 = st.slider("Druga granica wieku", age_bin1+1, 75, 50)
        
        age_bins = [age_bin1, age_bin2]
        age_labels = [
            f"<{age_bins[0]} lat",
            f"{age_bins[0]}-{age_bins[1]-1} lat",
            f">={age_bins[1]} lat"
        ]
        bin_data['age'] = pd.cut(
            bin_data['age'],
            bins=[0, age_bin1, age_bin2, 120],
            labels=age_labels,
            right=False
        )
        
        # Dyskretyzacja ciśnienia krwi
        st.subheader("Podział na przedziały ciśnienia skurczowego")
        bp_bin1 = st.slider("Granica ciśnienia", 80, 160, 120)
        
        bp_labels = [
            f"<{bp_bin1} mmHg",
            f">={bp_bin1} mmHg"
        ]
        bin_data['ap_hi'] = pd.cut(
            bin_data['ap_hi'],
            bins=[0, bp_bin1, 300],
            labels=bp_labels,
            right=False
        )
        
        # Dyskretyzacja BMI
        st.subheader("Podział na przedziały BMI")
        bmi_bin1 = st.slider("Dolna granica BMI", 15, 25, 18)
        bmi_bin2 = st.slider("Górna granica BMI", bmi_bin1+1, 35, 25)
        
        bmi_labels = [
            f"<{bmi_bin1} (niedowaga)",
            f"{bmi_bin1}-{bmi_bin2-1} (normalne)",
            f">={bmi_bin2} (nadwaga/otyłość)"
        ]
        bin_data['bmi'] = pd.cut(
            bin_data['bmi'],
            bins=[0, bmi_bin1, bmi_bin2, 100],
            labels=bmi_labels,
            right=False
        )
        
        # Parametry reguł
        st.subheader("Parametry reguł asocjacyjnych")
        min_support = st.slider("Minimalne wsparcie", 0.01, 0.5, 0.1, 0.01)
        min_confidence = st.slider("Minimalna ufność", 0.1, 1.0, 0.5, 0.05)
        min_lift = st.slider("Minimalny lift", 0.5, 5.0, 1.2, 0.1)

        if st.button("Generuj reguły asocjacyjne"):
            with st.spinner("Przetwarzanie reguł..."):
                # One-hot encoding
                cat_cols = [col for col in ['age', 'ap_hi', 'bmi', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'] 
                           if col in bin_data.columns]
                bin_data = pd.get_dummies(bin_data[cat_cols + ['cardio']], columns=cat_cols)
                
                # Generowanie częstych zestawów itemsetów
                frequent_items = apriori(bin_data, min_support=min_support, use_colnames=True)
                
                if not frequent_items.empty:
                    # Generowanie reguł
                    rules = association_rules(frequent_items, metric="confidence", 
                                            min_threshold=min_confidence)
                    rules = rules[rules['lift'] >= min_lift]
                    rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
                    
                    st.success(f"Znaleziono {len(rules)} reguł asocjacyjnych spełniających kryteria")
                    
                    if len(rules) > 0:
                        # ---------------------------
                        # 1. NAJLEPSZE REGUŁY - TABELA
                        # ---------------------------
                        st.subheader("🏆 Najlepsze reguły asocjacyjne")
                        
                        # Przygotowanie przyjaznych nazw dla cech
                        feature_names = {
                            'age': 'Wiek',
                            'ap_hi': 'Ciśnienie',
                            'bmi': 'BMI',
                            'gender': 'Płeć',
                            'cholesterol': 'Cholesterol',
                            'gluc': 'Glukoza',
                            'smoke': 'Palenie',
                            'alco': 'Alkohol',
                            'active': 'Aktywność',
                            'cardio': 'Choroba serca'
                        }
                        
                        # Funkcja do formatowania reguł
                        def format_rule(items):
                            formatted = []
                            for item in items:
                                for orig, new in feature_names.items():
                                    if orig in item:
                                        val = item.replace(f"{orig}_", "")
                                        formatted.append(f"{new}: {val}")
                                        break
                            return " + ".join(formatted)
                        
                        # Formatowanie reguł
                        rules['sformatowana_regula'] = rules.apply(
                            lambda x: f"{format_rule(x['antecedents'])} → {format_rule(x['consequents'])}", 
                            axis=1
                        )
                        
                        # Wybór i formatowanie kolumn do wyświetlenia
                        show_cols = {
                            'sformatowana_regula': 'Reguła',
                            'support': 'Wsparcie',
                            'confidence': 'Ufność',
                            'lift': 'Lift',
                            'conviction': 'Przekonanie'
                        }
                        
                        # Stylowanie tabeli
                        styled_rules = (
                            rules[list(show_cols.keys())]
                            .rename(columns=show_cols)
                            .style
                            .format({
                                'Wsparcie': '{:.2%}'.format,
                                'Ufność': '{:.2%}'.format,
                                'Lift': '{:.2f}'.format,
                                'Przekonanie': '{:.2f}'.format
                            })
                            .bar(subset=['Lift', 'Ufność'], color='#5fba7d')
                            .set_properties(**{
                                'text-align': 'left',
                                'white-space': 'wrap',
                                'max-width': '800px'
                            })
                        )
                        
                        st.dataframe(styled_rules, height=min(35 * len(rules), 500))
                        
                        # ---------------------------
                        # 2. WIZUALIZACJA SIECIOWA
                        # ---------------------------
                        st.subheader("🕸️ Diagram zależności między regułami")
                        
                        # Przygotowanie danych do wizualizacji
                        top_rules = rules.head(10)  # Ogranicz do 10 najlepszych reguł dla czytelności
                        
                        # Tworzenie grafu
                        fig = go.Figure()
                        
                        # Dodanie krawędzi
                        for _, row in top_rules.iterrows():
                            antecedents = format_rule(row['antecedents'])
                            consequents = format_rule(row['consequents'])
                            
                            fig.add_trace(go.Scatter(
                                x=[antecedents, consequents, None],
                                y=[1, 2, None],
                                mode='lines',
                                line=dict(width=row['lift']*0.3, color='rgba(100,100,100,0.4)'),
                                hoverinfo='text',
                                text=f"Lift: {row['lift']:.2f}<br>Ufność: {row['confidence']:.2%}",
                                name=''
                            ))
                        
                        # Dodanie węzłów
                        all_nodes = set()
                        for _, row in top_rules.iterrows():
                            all_nodes.add(format_rule(row['antecedents']))
                            all_nodes.add(format_rule(row['consequents']))
                        
                        for i, node in enumerate(all_nodes):
                            fig.add_trace(go.Scatter(
                                x=[node],
                                y=[1 if "→" not in node else 2],
                                mode='markers+text',
                                marker=dict(size=20, color='LightSkyBlue'),
                                text=[node],
                                textposition="middle center",
                                hoverinfo='text',
                                name=node
                            ))
                        
                        fig.update_layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # ---------------------------
                        # 3. STATYSTYKI PODSUMOWUJĄCE
                        # ---------------------------
                        st.subheader("📊 Podsumowanie statystyczne")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Najsilniejsza reguła (Lift)", f"{rules['lift'].max():.2f}")
                        with col2:
                            st.metric("Średnia ufność reguł", f"{rules['confidence'].mean():.2%}")
                        with col3:
                            st.metric("Liczba istotnych reguł", len(rules))
                        
                        # ---------------------------
                        # 4. EKSPORT WYNIKÓW
                        # ---------------------------
                        st.subheader("💾 Eksport wyników")
                        
                        # Przygotowanie danych do eksportu
                        export_rules = rules.copy()
                        export_rules['support'] = export_rules['support'].apply(lambda x: f"{x:.2%}")
                        export_rules['confidence'] = export_rules['confidence'].apply(lambda x: f"{x:.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Eksport do CSV
                            csv = export_rules[list(show_cols.keys())].to_csv(index=False, sep=';').encode('utf-8')
                            st.download_button(
                                "Pobierz reguły (CSV)",
                                csv,
                                "reguly_asocjacyjne.csv",
                                "text/csv"
                            )
                        
                        with col2:
                            # Eksport do JSON
                            json_data = export_rules[list(show_cols.keys())].to_json(orient='records', force_ascii=False)
                            st.download_button(
                                "Pobierz reguły (JSON)",
                                json_data,
                                "reguly_asocjacyjne.json",
                                "application/json"
                            )
                else:
                    st.warning("Nie znaleziono częstych zestawów itemsetów dla podanych parametrów")


# 6. Interpretacja modeli
def model_interpretation():
    st.header("🧠 5. Interpretacja modeli")
    
    if 'trained_models' not in st.session_state:
        st.warning("Najpierw wytrenuj modele w sekcji Modelowanie")
        return
    
    model_names = list(st.session_state['trained_models'].keys())
    selected_model = st.selectbox("Wybierz model do interpretacji", model_names)
    
    model = st.session_state['trained_models'][selected_model]
    preprocessing_data = st.session_state['preprocessing_data']
    
    st.subheader(f"Interpretacja modelu {selected_model}")

    # LIME
    st.subheader("🍋 LIME (Lokalna interpretacja)")
    try:
        # Create a preprocessing pipeline without the classifier
        preprocessor = model.named_steps['preprocessor']
        
        # Transform the training data for LIME
        X_train_processed = preprocessor.transform(preprocessing_data['X_train'])
        
        # Get feature names after preprocessing
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                cat_feature_names = trans.named_steps['onehot'].get_feature_names_out(cols)
                feature_names.extend(cat_feature_names)
        
        # Get categorical indices for LIME
        categorical_features = preprocessing_data['categorical_features']
        categorical_idx = [i for i, col in enumerate(preprocessing_data['X_train'].columns) 
                        if col in categorical_features]
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_processed,
            feature_names=feature_names,
            class_names=['Brak choroby', 'Choroba'],
            categorical_features=categorical_idx,
            verbose=True,
            mode='classification')
        
        # Select observation to explain
        sample_idx = st.slider("Wybierz obserwację do wyjaśnienia", 0, len(preprocessing_data['X_test'])-1, 0)
        X_test_processed = preprocessor.transform(preprocessing_data['X_test'].iloc[[sample_idx]])
        
        exp = explainer.explain_instance(
            X_test_processed[0], 
            lambda x: model.named_steps['classifier'].predict_proba(x),
            num_features=10)
        
        # Display explanation
        st.write(exp.as_list())
        
        # Visualization
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"LIME nie jest dostępny dla tego modelu: {str(e)}")
    
    # Permutation Importance
    if preprocessing_data.get('feature_selection_method') == 'Permutation Importance':
        st.subheader("🔀 Permutation Importance")
        try:
            X_processed = model.named_steps['preprocessor'].transform(preprocessing_data['X_test'])
            result = permutation_importance(
                model.named_steps['classifier'], X_processed, preprocessing_data['y_test'], 
                n_repeats=10, random_state=42)
            
            # Pobranie nazw cech
            feature_names = []
            for name, trans, cols in model.named_steps['preprocessor'].transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    cat_feature_names = trans.named_steps['onehot'].get_feature_names_out(cols)
                    feature_names.extend(cat_feature_names)
            
            perm_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': result.importances_mean,
                'Std': result.importances_std
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(perm_imp.head(20), x='Importance', y='Feature', 
                         error_x='Std', orientation='h',
                         title='Permutation Importance (Top 20 cech)',
                         color='Importance', color_continuous_scale='Purples')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Nie udało się obliczyć Permutation Importance: {str(e)}")
    
    # Przykładowe predykcje
    st.subheader("📌 Przykładowe predykcje")
    sample_size = st.slider("Liczba przykładów", 1, 20, 5)
    
    sample_data = preprocessing_data['X_test'].sample(sample_size, random_state=42)
    sample_true = preprocessing_data['y_test'].loc[sample_data.index]
    sample_pred = model.predict(sample_data)
    sample_proba = model.predict_proba(sample_data) if hasattr(model, 'predict_proba') else None
    
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        with st.expander(f"Przykład {i+1}: {'Choroba' if sample_true.loc[idx] == 1 else 'Zdrowy'}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dane wejściowe:**")
                st.json(row.to_dict())
            
            with col2:
                st.write("**Predykcja:**")
                if sample_proba is not None:
                    proba = sample_proba[i][1] * 100
                    if sample_pred[i] == 1:
                        st.error(f"Ryzyko choroby: {proba:.1f}%")
                    else:
                        st.success(f"Ryzyko choroby: {proba:.1f}%")
                    
                    # Wykres prawdopodobieństwa
                    fig = px.bar(x=['Brak choroby', 'Choroba'], 
                                 y=[100-proba, proba],
                                 labels={'x': 'Stan', 'y': 'Prawdopodobieństwo'},
                                 color=['Brak choroby', 'Choroba'],
                                 color_discrete_sequence=['green', 'red'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(f"Predykcja: {'Choroba' if sample_pred[i] == 1 else 'Brak choroby'}")


# 7. Predykcja dla nowych danych
def predict_new_data():
    st.header("🔮 6. Predykcja dla nowych danych")
    
    if 'trained_models' not in st.session_state or 'preprocessing_data' not in st.session_state:
        st.warning("Najpierw wytrenuj modele w sekcji Modelowanie")
        return
    
    model_names = list(st.session_state['trained_models'].keys())
    selected_model = st.selectbox("Wybierz model do predykcji", model_names)
    
    model = st.session_state['trained_models'][selected_model]
    preprocessing_data = st.session_state['preprocessing_data']
    
    # Opcje wprowadzania danych
    input_method = st.radio("Wybierz metodę wprowadzania danych",
                           ["Formularz", "Wczytaj plik CSV"],
                           horizontal=True)
    
    if input_method == "Formularz":
        # Formularz do wprowadzania danych
        st.subheader("👤 Wprowadź dane pacjenta")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Podstawowe informacje")
            age = st.slider("Wiek (lata)", 20, 100, 50)
            gender = st.radio("Płeć", ['kobieta', 'mężczyzna'])
            height = st.slider("Wzrost (cm)", 140, 220, 170)
            weight = st.slider("Waga (kg)", 40, 200, 70)
            bmi = weight / (height/100)**2
            st.metric("BMI", f"{bmi:.1f}")
        
        with col2:
            st.subheader("Ciśnienie krwi")
            ap_hi = st.slider("Skurczowe (mmHg)", 80, 250, 120)
            ap_lo = st.slider("Rozkurczowe (mmHg)", 50, 150, 80)
            pulse_pressure = ap_hi - ap_lo
            st.metric("Ciśnienie tętna", pulse_pressure)
            
            st.subheader("Aktywność")
            active = st.checkbox("Aktywny fizycznie")
        
        with col3:
            st.subheader("Wyniki badań")
            cholesterol = st.selectbox("Cholesterol", ['normalny', 'podwyższony', 'wysoki'])
            gluc = st.selectbox("Glukoza", ['normalny', 'podwyższony', 'wysoki'])
            
            st.subheader("Używki")
            smoke = st.checkbox("Palący")
            alco = st.checkbox("Spożywa alkohol")
        
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'height': [height],
            'weight': [weight],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'bmi': [bmi],
            'pulse_pressure': [pulse_pressure],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active]
        })
    else:
        # Wczytywanie danych z pliku
        st.subheader("📁 Wczytaj dane z pliku CSV")
        uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                st.write("Podgląd danych:")
                st.dataframe(input_data)
                
                # Sprawdź wymagane kolumny
                required_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo']
                missing_cols = [col for col in required_cols if col not in input_data.columns]
                
                if missing_cols:
                    st.error(f"Brak wymaganych kolumn: {', '.join(missing_cols)}")
                    return
            except Exception as e:
                st.error(f"Błąd podczas wczytywania pliku: {str(e)}")
                return
    
    if st.button("Wykonaj predykcję", type="primary") and 'input_data' in locals():
        # Przygotowanie danych wejściowych
        try:
            # Predykcja
            with st.spinner("Przetwarzanie..."):
                time.sleep(1)
                prediction = model.predict(input_data)
                proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
                
                # Wyświetlenie wyników
                st.subheader("🎯 Wynik predykcji")
                
                if proba is not None:
                    if input_data.shape[0] == 1:
                        # Pojedyncza predykcja
                        if prediction[0] == 1:
                            st.error(f"""
                            ### Ryzyko choroby serca: WYSOKIE ({proba[0][1]*100:.1f}% prawdopodobieństwo)
                            \nZalecana konsultacja z kardiologiem!
                            """)
                        else:
                            st.success(f"""
                            ### Ryzyko choroby serca: NISKIE ({proba[0][0]*100:.1f}% prawdopodobieństwo)
                            \nProfilaktyka wystarczająca.
                            """)
                        
                        # Wykres prawdopodobieństwa
                        fig = px.bar(x=['Brak choroby', 'Choroba'], 
                                     y=[proba[0][0]*100, proba[0][1]*100],
                                     labels={'x': 'Stan', 'y': 'Prawdopodobieństwo [%]'},
                                     title='Rozkład prawdopodobieństwa',
                                     color=['Brak choroby', 'Choroba'],
                                     color_discrete_sequence=['green', 'red'],
                                     text=[f"{proba[0][0]*100:.1f}%", f"{proba[0][1]*100:.1f}%"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Wiele predykcji
                        results = input_data.copy()
                        results['Predykcja'] = prediction
                        if proba is not None:
                            results['Prawdopodobieństwo choroby'] = proba[:, 1]
                        
                        st.dataframe(results)
                        
                        # Podsumowanie
                        st.subheader("Podsumowanie predykcji")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Liczba pacjentów", len(results))
                            st.metric("Przewidywane przypadki choroby", sum(results['Predykcja']))
                        with col2:
                            if 'Prawdopodobieństwo choroby' in results.columns:
                                st.metric("Średnie prawdopodobieństwo", f"{results['Prawdopodobieństwo choroby'].mean():.1f}%")
                                st.metric("Maksymalne prawdopodobieństwo", f"{results['Prawdopodobieństwo choroby'].max():.1f}%")
                else:
                    # Modele bez predict_proba
                    if input_data.shape[0] == 1:
                        st.write(f"### Predykcja: {'Choroba' if prediction[0] == 1 else 'Brak choroby'}")
                    else:
                        results = input_data.copy()
                        results['Predykcja'] = prediction
                        st.dataframe(results)
                        
                        st.subheader("Podsumowanie predykcji")
                        st.metric("Liczba pacjentów", len(results))
                        st.metric("Przewidywane przypadki choroby", sum(results['Predykcja']))
                
                # Wyjaśnienie czynników ryzyka
                if input_data.shape[0] == 1 and proba is not None:
                    st.subheader("📌 Analiza czynników ryzyka")
                    
                    # Nagłówek z informacją o prawdopodobieństwie
                    risk_level = "WYSOKIE" if proba[0][1] > 0.5 else "NISKIE"
                    risk_color = "red" if risk_level == "WYSOKIE" else "green"
                    st.markdown(f"<h4 style='color:{risk_color}'>Ryzyko choroby: {risk_level} ({proba[0][1]*100:.1f}%)</h4>", 
                                unsafe_allow_html=True)
                    
                    try:
                        classifier = model.named_steps['classifier']
                        
                        # 1. Analiza dla modeli drzewiastych
                        if hasattr(classifier, 'feature_importances_'):
                            # Pobierz nazwy cech po przetworzeniu
                            feature_names = []
                            for name, trans, cols in model.named_steps['preprocessor'].transformers_:
                                if name == 'num':
                                    feature_names.extend(cols)
                                elif name == 'cat':
                                    if hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                                        feature_names.extend(trans.named_steps['onehot'].get_feature_names_out(cols))
                            
                            importances = classifier.feature_importances_
                            feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                            feature_imp = feature_imp.sort_values('Importance', ascending=False)
                            
                            # Wyświetl top 5 cech
                            st.write("**Najważniejsze cechy modelu:**")
                            st.dataframe(feature_imp.head(5).style.format({'Importance': '{:.3f}'}))
                            
                            # Sprawdź wartości pacjenta dla ważnych cech
                            st.write("**Wartości pacjenta dla kluczowych cech:**")
                            risk_factors = []
                            
                            for feature in feature_imp.head(10)['Feature']:
                                # Mapowanie z powrotem do oryginalnych cech
                                original_feature = None
                                if 'gender' in feature:
                                    original_feature = 'gender'
                                    value = input_data['gender'].iloc[0]
                                    display_value = value
                                elif 'cholesterol' in feature:
                                    original_feature = 'cholesterol'
                                    value = input_data['cholesterol'].iloc[0]
                                    display_value = value
                                elif feature in input_data.columns:
                                    original_feature = feature
                                    value = input_data[feature].iloc[0]
                                    display_value = f"{value:.2f}" if isinstance(value, (int, float)) else value
                                
                                if original_feature:
                                    # Ocena ryzyka dla konkretnych wartości
                                    risk_comment = ""
                                    if original_feature == 'age' and value > 60:
                                        risk_comment = "⚠️ Ryzyko wzrasta po 60 roku życia"
                                    elif original_feature == 'ap_hi' and value > 140:
                                        risk_comment = "⚠️ Nadciśnienie (≥140 mmHg)"
                                    elif original_feature == 'bmi' and value > 30:
                                        risk_comment = "⚠️ Otyłość (BMI >30)"
                                    
                                    if risk_comment:
                                        risk_factors.append((original_feature, display_value, risk_comment))
                            
                            if risk_factors:
                                st.warning("**Potencjalne czynniki ryzyka:**")
                                for factor in risk_factors:
                                    st.write(f"- **{factor[0]}**: {factor[1]} → {factor[2]}")
                            # Wyświetlanie szczegółowej analizy czynników ryzyka
                            st.subheader("🔍 Szczegółowa analiza profilu pacjenta")

                            # Utwórz kolumny dla lepszego układu
                            col1, col2 = st.columns(2)

                            with col1:
                                # Panel z podstawowymi parametrami
                                st.markdown("**📊 Twoje kluczowe parametry:**")
                                
                                # Wiek
                                age = input_data['age'].iloc[0]
                                age_status = "✅ W normie" if age < 60 else "⚠️ Podwyższone ryzyko (wiek >60)"
                                st.write(f"- **Wiek**: {age} lat | {age_status}")
                                
                                # BMI
                                bmi = input_data['bmi'].iloc[0]
                                if bmi < 18.5:
                                    bmi_status = "❌ Niedowaga"
                                elif 18.5 <= bmi < 25:
                                    bmi_status = "✅ Prawidłowe"
                                elif 25 <= bmi < 30:
                                    bmi_status = "⚠️ Nadwaga"
                                else:
                                    bmi_status = "❌ Otyłość"
                                st.write(f"- **BMI**: {bmi:.1f} | {bmi_status}")
                                
                                # Ciśnienie krwi
                                ap_hi = input_data['ap_hi'].iloc[0]
                                if ap_hi < 120:
                                    bp_status = "✅ Optymalne"
                                elif 120 <= ap_hi < 140:
                                    bp_status = "⚠️ Wysokie prawidłowe"
                                else:
                                    bp_status = "❌ Nadciśnienie"
                                st.write(f"- **Ciśnienie skurczowe**: {ap_hi} mmHg | {bp_status}")

                            with col2:
                                # Panel z dodatkowymi czynnikami ryzyka
                                st.markdown("**📌 Dodatkowe czynniki:**")
                                
                                # Cholesterol
                                chol = input_data['cholesterol'].iloc[0]
                                chol_status = "⚠️ Ryzyko" if chol in ['podwyższony', 'wysoki'] else "✅ W normie"
                                st.write(f"- **Cholesterol**: {chol} | {chol_status}")
                                
                                # Palenie
                                smoke = input_data['smoke'].iloc[0]
                                st.write(f"- **Palenie**: {'❌ Tak' if smoke else '✅ Nie'}")
                                
                                # Aktywność fizyczna
                                active = input_data['active'].iloc[0]
                                st.write(f"- **Aktywność fizyczna**: {'✅ Tak' if active else '⚠️ Brak'}")

                            # Spersonalizowane zalecenia
                            st.subheader("💡 Spersonalizowane zalecenia profilaktyczne")

                            recommendations = []

                            # Zalecenia na podstawie wieku
                            if age >= 45:
                                recommendations.append("- Raz w roku wykonaj podstawowe badania krwi (morfologia, glukoza, lipidogram)")
                                
                            # Zalecenia dot. BMI
                            if bmi >= 25:
                                recommendations.append(f"- Rozważ redukcję masy ciała o {max(1, round(bmi-24))} kg dla osiągnięcia prawidłowego BMI")
                            elif bmi < 18.5:
                                recommendations.append("- Skonsultuj z lekarzem lub dietetykiem niedowagę")

                            # Zalecenia dot. ciśnienia
                            if ap_hi >= 130:
                                recommendations.append("- Mierz ciśnienie regularnie (2-3 razy w tygodniu)")
                                if ap_hi >= 140:
                                    recommendations.append("- Skonsultuj z lekarzem wyniki pomiarów ciśnienia")

                            # Inne czynniki
                            if chol in ['podwyższony', 'wysoki']:
                                recommendations.append("- Ogranicz tłuszcze nasycone w diecie (tłuste mięsa, smalec, masło)")
                                
                            if smoke:
                                recommendations.append("- Rozważ program rzucania palenia - korzyści zdrowotne pojawiają się już po 24h!")

                            if not active:
                                recommendations.append("- Zacznij od 30-minutowych spacerów 3-4 razy w tygodniu")

                            # Domyślne zalecenia jeśli wszystko w normie
                            if not recommendations:
                                recommendations.extend([
                                    "- Kontynuuj zdrowy styl życia!",
                                    "- Raz na 2 lata wykonaj podstawowe badania kontrolne",
                                    "- Raz na 5 lat sprawdź poziom cholesterolu"
                                ])

                            for rec in recommendations:
                                st.write(rec)

                            # Wykres porównawczy
                            st.subheader("📈 Twoje wyniki vs wartości referencyjne")

                            fig = go.Figure()

                            # Dodaj wartości pacjenta
                            fig.add_trace(go.Bar(
                                x=['Wiek', 'BMI', 'Ciśnienie'],
                                y=[age, bmi, ap_hi],
                                name='Twoje wartości',
                                marker_color='indianred'
                            ))

                            # Dodaj wartości referencyjne
                            fig.add_trace(go.Scatter(
                                x=['Wiek', 'BMI', 'Ciśnienie'],
                                y=[60, 24.9, 120],  # Wartości docelowe
                                name='Wartości optymalne',
                                mode='markers',
                                marker=dict(size=15, color='lightgreen')
                            ))

                            fig.update_layout(
                                title='Porównanie z wartościami referencyjnymi',
                                yaxis_title='Wartość'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Linki do dodatkowych zasobów
                            st.markdown("""
                            **📚 Przydatne zasoby:**
                            - [Wytyczne dot. zdrowego stylu życia (WHO)](https://www.who.int)
                            - [Kalkulator BMI i porady żywieniowe](https://www.nhlbi.nih.gov/health/educational/lose_wt/index.htm)
                            - [Poradnik dot. nadciśnienia](https://www.heart.org/en/health-topics/high-blood-pressure)
                            """)
                        
                        # 2. Dla modeli liniowych (np. regresja logistyczna)
                        elif hasattr(classifier, 'coef_'):
                            st.warning("""
                            **Uwaga:** Model liniowy wykrywa inne wzorce ryzyka niż modele drzewiaste.
                            Zinterpretuj współczynniki poniżej:
                            """)
                            
                            # Wyświetl najważniejsze współczynniki
                            coef = pd.DataFrame({
                                'Feature': input_data.columns,
                                'Współczynnik': classifier.coef_[0]
                            }).sort_values('Współczynnik', key=abs, ascending=False)
                            
                            st.dataframe(coef.head(10).style.format({'Współczynnik': '{:.3f}'}))
                            
                            # Prosta interpretacja
                            st.info("""
                            **Jak interpretować:**
                            - Wartości dodatnie zwiększają ryzyko choroby
                            - Wartości ujemne zmniejszają ryzyko
                            - Im większa wartość bezwzględna, tym silniejszy efekt
                            """)
                        
                        else:
                            st.warning("""
                            **Informacja:** Ten model nie udostępnia standardowej analizy czynników ryzyka.
                            Możesz spróbować:
                            1. Wybrać inny model (np. Random Forest, XGBoost)
                            2. Użyć narzędzi interpretacyjnych (LIME) w sekcji 'Interpretacja modeli'
                            """)
                    
                    except Exception as e:
                        st.error(f"Techniczny błąd analizy: {str(e)}")
                        st.info("""
                        **Co możesz zrobić:**
                        - Sprawdź czy wszystkie wymagane cechy są wprowadzone
                        - Spróbuj innego modelu
                        - Skonsultuj się z administratorem systemu
                        """)
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {str(e)}")

# 8. Generowanie raportu PDF
def generate_pdf_report(data):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Raport Analizy Kardiologicznej', 0, 1, 'C')
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Strona {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    
    # Funkcja do bezpiecznego formatowania tekstu
    def safe_text(text):
        if not isinstance(text, str):
            text = str(text)
        # Zamiana polskich znaków na podstawowe litery
        replacements = {
            'ł': 'l', 'ą': 'a', 'ę': 'e', 'ś': 's', 'ć': 'c',
            'ź': 'z', 'ż': 'z', 'ń': 'n', 'ó': 'o',
            'Ł': 'L', 'Ą': 'A', 'Ę': 'E', 'Ś': 'S',
            'Ć': 'C', 'Ź': 'Z', 'Ż': 'Z', 'Ń': 'N', 'Ó': 'O'
        }
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
        return text.encode('latin-1', 'replace').decode('latin-1')

    # 1. Podstawowe statystyki
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text("1. Podstawowe statystyki danych"), ln=1)
    pdf.set_font("Arial", size=10)
    
    stats = [
        f"Liczba pacjentow: {data.shape[0]}",
        f"Liczba cech: {data.shape[1]}"
    ]
    
    if 'cardio' in data.columns:
        stats.append(f"Rozklad chorob serca: {data['cardio'].sum()} przypadkow ({(data['cardio'].mean()*100):.1f}%)")
    
    for stat in stats:
        pdf.cell(200, 10, txt=safe_text(stat), ln=1)

    # 2. Wyniki modelowania
    if 'model_results' in st.session_state:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=safe_text("2. Wyniki modelowania"), ln=1)
        pdf.set_font("Arial", size=10)
        
        results = st.session_state['model_results']
        
        try:
            available_metrics = [m for m in ['ROC-AUC', 'Accuracy', 'F1-Score'] 
                               if m in results.columns and results[m].notna().all()]
            
            if not available_metrics:
                pdf.cell(200, 10, txt=safe_text("Brak kompletnych metryk do oceny modeli"), ln=1)
            else:
                for metric in available_metrics:
                    results[f'{metric}_rank'] = results[metric].rank(ascending=False)
                
                results['Sredni ranking'] = results[[f'{m}_rank' for m in available_metrics]].mean(axis=1)
                best_model_row = results.loc[results['Sredni ranking'].idxmin()]
                best_model_name = best_model_row['Model']
                
                pdf.cell(200, 10, txt=safe_text(f"Najlepszy model: {best_model_name}"), ln=1)
                for metric in available_metrics:
                    pdf.cell(200, 10, txt=safe_text(f"{metric}: {best_model_row[metric]:.3f}"), ln=1)
                
                if 'feature_importances' in st.session_state:
                    # Znajdź pasującą nazwę modelu (ignoruj wielkość liter i spacje)
                    matching_keys = [k for k in st.session_state['feature_importances'].keys() 
                                   if best_model_name.lower().replace(" ", "") in k.lower().replace(" ", "")]
                    
                    if matching_keys:
                        feature_imp = st.session_state['feature_importances'][matching_keys[0]].head(10)
                        
                        pdf.ln(10)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt=safe_text("3. Najwazniejsze cechy w najlepszym modelu"), ln=1)
                        pdf.set_font("Arial", size=10)
                        
                        for _, row in feature_imp.iterrows():
                            pdf.cell(200, 10, txt=safe_text(f"- {row['Feature']}: {row['Importance']:.3f}"), ln=1)
        except Exception as e:
            pdf.cell(200, 10, txt=safe_text(f"Blad podczas generowania podsumowania modeli: {str(e)}"), ln=1)

    # 4. Zalecenia
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=safe_text("4. Zalecenia"), ln=1)
    pdf.set_font("Arial", size=10)
    
    report_path = os.path.join("reports", f"raport_kardiologiczny_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    try:
        pdf.output(report_path)
    except UnicodeEncodeError:
        with open(report_path, "wb") as f:
            f.write(pdf.output(dest='S').encode('latin-1', errors='replace'))
    
    return report_path

# 9. Podsumowanie i raport
def show_summary(data):
    st.header("📝 7. Podsumowanie i raport")
    
    with st.expander("📌 Streszczenie analizy", expanded=True):
        # Przygotuj dane numeryczne dla korelacji
        numeric_cols = [col for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure', 'cardio'] 
                       if col in data.columns and data[col].dtype in ['int64', 'float64']]
        
        st.subheader("🔎 Najważniejsze wnioski z EDA")
        
        if numeric_cols:
            numeric_data = data[numeric_cols].copy()
            corr_matrix = numeric_data.corr()
            
            eda_text = [
                f"1. **Rozkład klas**: Dataset zawiera {data['cardio'].sum()} przypadki chorób serca " + 
                f"i {len(data) - data['cardio'].sum()} przypadki zdrowych osób."
            ]
            
            if 'cardio' in corr_matrix.columns:
                corr_with_target = corr_matrix['cardio'].abs().sort_values(ascending=False)
                eda_text.append("2. **Korelacje**: Najsilniejszą korelację z chorobą serca wykazują:")
                for i in range(1, min(4, len(corr_with_target))):
                    eda_text.append(f"   - {corr_with_target.index[i]} ({corr_with_target.iloc[i]:.3f})")
            
            eda_text.append("3. **Rozkłady**: Wykresy pokazują wyraźne różnice w rozkładach wielu cech między grupami chorych i zdrowych.")
            
            st.write("\n".join(eda_text))
        else:
            st.write("1. Brak danych numerycznych do analizy korelacji.")
        
        if 'model_results' in st.session_state:
            st.subheader("🏆 Najlepsze modele")
            results = st.session_state['model_results']
            
            # Znajdź najlepszy model na podstawie dostępnych metryk
            available_metrics = [m for m in ['ROC-AUC', 'Accuracy', 'F1-Score'] if m in results.columns]
            if available_metrics:
                # Konwersja na liczby (pomijając 'N/A')
                results_numeric = results.copy()
                for metric in available_metrics:
                    results_numeric[metric] = results_numeric[metric].apply(lambda x: float(x) if x != 'N/A' else np.nan)
                
                # Obliczenie rankingu
                for metric in available_metrics:
                    results_numeric[f'{metric}_rank'] = results_numeric[metric].rank(ascending=False)
                
                results_numeric['Średni ranking'] = results_numeric[[f'{m}_rank' for m in available_metrics]].mean(axis=1)
                best_model_row = results_numeric.loc[results_numeric['Średni ranking'].idxmin()]
                
                model_text = [
                    f"1. **Najlepszy model**: {best_model_row['Model']}",
                    f"2. **Średni ranking modeli**: {results_numeric['Średni ranking'].mean():.1f}"
                ]
                
                for metric in available_metrics:
                    model_text.append(f"3. **{metric}**: {best_model_row[metric]:.3f}")
                
                st.write("\n".join(model_text))

            st.subheader("⚡ Najszybszy model")
            if 'Training Time' in results.columns or 'training_time_seconds' in results.columns:
                # Konwertuj czas na sekundy dla porównania
                results['Time_sec'] = results['Training Time'].apply(
                    lambda x: float(x.split()[0]) if isinstance(x, str) and 's' in x else np.nan)
                
                if not results['Time_sec'].isna().all():
                    fastest_model_row = results.loc[results['Time_sec'].idxmin()]
                    st.write(f"**Model:** {fastest_model_row['Model']}")
                    st.write(f"- Czas trenowania: {fastest_model_row['Training Time']}")
                    
                else:
                    st.warning("Brak danych o czasie trenowania")
            else:
                st.warning("Brak danych o czasie trenowania")


        st.subheader("💡 Zalecenia praktyczne")
        st.write("""
        1. **Wdrożenie**: Model powinien być wdrożony jako narzędzie wspomagające diagnozę.
        2. **Profilaktyka**: Pacjenci z wynikiem powyżej 50% ryzyka powinni być kierowani na badania.
        3. **Monitorowanie**: Warto rozważyć okresowe aktualizowanie modelu.
        4. **Interpretacja**: Lekarze powinni brać pod uwagę najważniejsze czynniki ryzyka identyfikowane przez model.
        """)
    
    # Generowanie raportu PDF
    st.subheader("📄 Generowanie raportu PDF")
    if st.button("Wygeneruj raport PDF"):
        with st.spinner("Generowanie raportu..."):
            report_path = generate_pdf_report(data)
            
            with open(report_path, "rb") as f:
                pdf_data = f.read()
            
            st.success("Raport wygenerowany pomyślnie!")
            st.download_button(
                label="Pobierz raport",
                data=pdf_data,
                file_name=os.path.basename(report_path),
                mime="application/pdf"
            )

# Główna aplikacja
def main():
    st.sidebar.title("Menu")
    
    # Opcja wczytania danych
    data_option = st.sidebar.radio("Źródło danych",
                                  ["Użyj domyślnego datasetu", "Wczytaj własny plik CSV"])
    
    if data_option == "Wczytaj własny plik CSV":
        uploaded_file = st.sidebar.file_uploader("Wybierz plik CSV", type="csv")
        data = load_data(uploaded_file)
    else:
        data = load_data()
    
    if data is None:
        st.warning("Proszę wczytać dane do analizy")
        return
    
    menu_options = {
        "1. Analiza danych (EDA)": lambda: perform_eda(data),
        "2. Przygotowanie danych": lambda: perform_preprocessing(data),
        "3. Modelowanie": lambda: perform_modeling(st.session_state['preprocessing_data']) 
                                if 'preprocessing_data' in st.session_state 
                                else st.warning("Najpierw wykonaj przygotowanie danych w sekcji 2"),
        "4. Reguły asocjacyjne": lambda: perform_association_rules(data),
        "5. Interpretacja modeli": model_interpretation,
        "6. Predykcja": predict_new_data,
        "7. Podsumowanie": lambda: show_summary(data)
    }
    
    selected_option = st.sidebar.radio("Wybierz sekcję", list(menu_options.keys()))
    
    # Wyświetlenie wybranej sekcji
    menu_options[selected_option]()
    
    # Stopka
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Aplikacja do analizy danych kardiologicznych**  
    Olha Yakymenko
    Ostatnia aktualizacja: {}
    """.format(datetime.now().strftime("%Y-%m-%d")))

if __name__ == "__main__":
    main()














