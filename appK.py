import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Health Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS
# ========================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# LOAD AND CACHE DATA
# ========================
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    
    # Introduce NaN values randomly
    np.random.seed(42)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        # Randomly select 5% of indices to set as NaN
        nan_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        df.loc[nan_indices, col] = np.nan
    
    # Handle missing values with mean
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
    
    return df

@st.cache_resource
def train_model(df):
    # Separate features and target
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    X['sex'] = le_sex.fit_transform(X['sex'])
    X['smoker'] = le_smoker.fit_transform(X['smoker'])
    X['region'] = le_region.fit_transform(X['region'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal k
    k_values = range(1, 31)
    test_scores = []
    
    for k in k_values:
        knn_temp = KNeighborsRegressor(n_neighbors=k)
        knn_temp.fit(X_train_scaled, y_train)
        test_scores.append(knn_temp.score(X_test_scaled, y_test))
    
    best_k = k_values[np.argmax(test_scores)]
    
    # Train final model with best k
    knn_best = KNeighborsRegressor(n_neighbors=best_k)
    knn_best.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = knn_best.predict(X_train_scaled)
    y_test_pred = knn_best.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_r2': r2_score(y_test, y_test_pred),
        'best_k': best_k
    }
    
    return {
        'model': knn_best,
        'scaler': scaler,
        'encoders': {'sex': le_sex, 'smoker': le_smoker, 'region': le_region},
        'metrics': metrics,
        'data': {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        },
        'k_analysis': {'k_values': list(k_values), 'test_scores': test_scores}
    }

# ========================
# MAIN APP
# ========================
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Health Insurance Cost Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
        model_data = train_model(df)
    except FileNotFoundError:
        st.error("⚠️ insurance.csv file not found. Please upload the file first.")
        uploaded_file = st.file_uploader("Upload insurance.csv file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.to_csv("insurance.csv", index=False)
            st.success("✅ File uploaded successfully!")
            st.rerun()
        return
    
    # Sidebar
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 5rem;'>🏥</h1>", unsafe_allow_html=True)
    st.sidebar.title("🎛️ Main Menu")
    page = st.sidebar.radio(
        "Choose Page:",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Performance", "🔮 Prediction", "📈 Advanced Visualizations"]
    )
    
    # ========================
    # HOME PAGE
    # ========================
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📦 Total Records", f"{len(df):,}")
        with col2:
            st.metric("📋 Number of Features", df.shape[1] - 1)
        with col3:
            st.metric("🎯 Best K", model_data['metrics']['best_k'])
        
        st.markdown("---")
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Descriptive Statistics:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Data info
        st.markdown("---")
        st.subheader("ℹ️ Data Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Missing Values': df.isnull().sum().values
            })
            st.dataframe(info_df, use_container_width=True)
        
        with col2:
            st.write("**Categorical Variables Distribution:**")
            for col in ['sex', 'smoker', 'region']:
                st.write(f"**{col}:**")
                st.write(df[col].value_counts())
    
    # ========================
    # DATA ANALYSIS PAGE
    # ========================
    elif page == "📊 Data Analysis":
        st.header("📊 Exploratory Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Relationships", "📊 Statistics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig = px.histogram(df, x='age', nbins=30, title='Age Distribution',
                                 color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)
                
                # BMI distribution
                fig = px.histogram(df, x='bmi', nbins=30, title='BMI Distribution',
                                 color_discrete_sequence=['#ff7f0e'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Charges distribution
                fig = px.histogram(df, x='charges', nbins=50, title='Charges Distribution',
                                 color_discrete_sequence=['#2ca02c'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Children distribution
                fig = px.histogram(df, x='children', title='Children Distribution',
                                 color_discrete_sequence=['#d62728'])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Charges vs Age
                fig = px.scatter(df, x='age', y='charges', color='smoker',
                               title='Charges vs Age Relationship',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
                
                # Charges by Smoker
                fig = px.box(df, x='smoker', y='charges', color='smoker',
                           title='Charges by Smoking Status')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Charges vs BMI
                fig = px.scatter(df, x='bmi', y='charges', color='smoker',
                               title='Charges vs BMI Relationship',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
                
                # Charges by Region
                fig = px.box(df, x='region', y='charges', color='region',
                           title='Charges by Region')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Correlation matrix
            st.subheader("Correlation Matrix")
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                          title='Correlation Matrix',
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================
    # MODEL PERFORMANCE PAGE
    # ========================
    elif page == "🤖 Model Performance":
        st.header("🤖 KNN Model Performance")
        
        metrics = model_data['metrics']
        
        # Metrics cards
        st.subheader(f"🎯 Optimal Model: K = {metrics['best_k']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE (Train)", f"${metrics['train_mae']:,.2f}")
            st.metric("MAE (Test)", f"${metrics['test_mae']:,.2f}")
        
        with col2:
            st.metric("RMSE (Train)", f"${metrics['train_rmse']:,.2f}")
            st.metric("RMSE (Test)", f"${metrics['test_rmse']:,.2f}")
        
        with col3:
            st.metric("R² (Train)", f"{metrics['train_r2']:.4f}")
            st.metric("R² (Test)", f"{metrics['test_r2']:.4f}")
        
        st.markdown("---")
        
        # K vs R² plot
        st.subheader("📈 Optimal K Value Analysis")
        k_values = model_data['k_analysis']['k_values']
        test_scores = model_data['k_analysis']['test_scores']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_values, y=test_scores, mode='lines+markers',
                                name='Test R²', line=dict(color='#1f77b4', width=3)))
        fig.add_vline(x=metrics['best_k'], line_dash="dash", line_color="red",
                        annotation_text=f"Best K={metrics['best_k']}")
        fig.update_layout(title='Impact of K Value on Model Performance',
                            xaxis_title='Number of Neighbors (K)',
                            yaxis_title='R² Score',
                            height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted plots
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Actual vs Predicted (Train)")
            y_train = model_data['data']['y_train']
            y_train_pred = model_data['data']['y_train_pred']
            
            fig = px.scatter(x=y_train, y=y_train_pred, 
                            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                            title=f'Train Set (R² = {metrics["train_r2"]:.4f})')
            fig.add_trace(go.Scatter(x=[y_train.min(), y_train.max()],
                                    y=[y_train.min(), y_train.max()],
                                    mode='lines', line=dict(color='red', dash='dash'),
                                    name='Perfect Prediction'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔵 Actual vs Predicted (Test)")
            y_test = model_data['data']['y_test']
            y_test_pred = model_data['data']['y_test_pred']
            
            fig = px.scatter(x=y_test, y=y_test_pred,
                            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                            title=f'Test Set (R² = {metrics["test_r2"]:.4f})',
                            color_discrete_sequence=['green'])
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                    y=[y_test.min(), y_test.max()],
                                    mode='lines', line=dict(color='red', dash='dash'),
                                    name='Perfect Prediction'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Residual plots
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Residual Distribution (Train)")
            residuals_train = y_train - y_train_pred
            fig = px.scatter(x=y_train_pred, y=residuals_train,
                            labels={'x': 'Predicted Values', 'y': 'Residuals'},
                            title='Residual Plot (Train)')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Residual Distribution (Test)")
            residuals_test = y_test - y_test_pred
            fig = px.scatter(x=y_test_pred, y=residuals_test,
                            labels={'x': 'Predicted Values', 'y': 'Residuals'},
                            title='Residual Plot (Test)',
                            color_discrete_sequence=['green'])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================
    # PREDICTION PAGE
    # ========================
    elif page == "🔮 Prediction":
        st.header("🔮 Insurance Cost Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 30)
            sex = st.selectbox("Sex", ['male', 'female'])
            bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0, 0.1)
        
        with col2:
            children = st.slider("Number of Children", 0, 5, 0)
            smoker = st.selectbox("Smoker", ['no', 'yes'])
            region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])
        
        if st.button("🔮 Calculate Predicted Cost", type="primary"):
            # Encode inputs
            encoders = model_data['encoders']
            sex_encoded = encoders['sex'].transform([sex])[0]
            smoker_encoded = encoders['smoker'].transform([smoker])[0]
            region_encoded = encoders['region'].transform([region])[0]
            
            # Create input array
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            
            # Scale and predict
            input_scaled = model_data['scaler'].transform(input_data)
            prediction = model_data['model'].predict(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.success(f"## 💰 Predicted Cost: ${prediction:,.2f}")
            
            # Show input summary
            st.markdown("---")
            st.subheader("📋 Input Data Summary:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Age:** {age} years")
                st.write(f"**Sex:** {sex}")
            
            with col2:
                st.write(f"**BMI:** {bmi}")
                st.write(f"**Children:** {children}")
            
            with col3:
                st.write(f"**Smoker:** {smoker}")
                st.write(f"**Region:** {region}")
            
            # Comparison with dataset
            st.markdown("---")
            st.subheader("📊 Comparison with Dataset")
            
            avg_cost = df['charges'].mean()
            smoker_avg = df[df['smoker'] == smoker]['charges'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Average Cost", f"${avg_cost:,.2f}",
                         f"{((prediction - avg_cost) / avg_cost * 100):+.1f}%")
            with col2:
                st.metric(f"Average Cost ({smoker})", f"${smoker_avg:,.2f}",
                         f"{((prediction - smoker_avg) / smoker_avg * 100):+.1f}%")
            with col3:
                percentile = (df['charges'] < prediction).mean() * 100
                st.metric("Percentile", f"{percentile:.1f}%")
        
        # Batch prediction
        st.markdown("---")
        st.subheader("📊 Multiple Case Predictions")
        
        examples = [
            (30, 'male', 25.0, 1, 'no', 'southwest'),
            (30, 'male', 25.0, 1, 'yes', 'southwest'),
            (45, 'female', 30.0, 2, 'no', 'northeast'),
            (50, 'male', 32.0, 3, 'yes', 'southeast'),
        ]
        
        predictions_list = []
        for age, sex, bmi, children, smoker, region in examples:
            encoders = model_data['encoders']
            sex_encoded = encoders['sex'].transform([sex])[0]
            smoker_encoded = encoders['smoker'].transform([smoker])[0]
            region_encoded = encoders['region'].transform([region])[0]
            
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            input_scaled = model_data['scaler'].transform(input_data)
            prediction = model_data['model'].predict(input_scaled)[0]
            
            predictions_list.append({
                'Age': age,
                'Sex': sex,
                'BMI': bmi,
                'Children': children,
                'Smoker': smoker,
                'Region': region,
                'Predicted Cost': f"${prediction:,.2f}"
            })
        
        predictions_df = pd.DataFrame(predictions_list)
        st.dataframe(predictions_df, use_container_width=True)
    
    # ========================
    # ADVANCED VISUALIZATIONS PAGE
    # ========================
    elif page == "📈 Advanced Visualizations":
        st.header("📈 Advanced Visualizations")
        
        tab1, tab2 = st.tabs(["🎨 Interactive Charts", "📊 Custom Analysis"])
        
        with tab1:
            # 3D Scatter
            st.subheader("🌐 3D Scatter Plot")
            fig = px.scatter_3d(df, x='age', y='bmi', z='charges',
                                color='smoker', size='children',
                                title='Relationship between Age, BMI, and Charges')
            st.plotly_chart(fig, use_container_width=True)
            
            # Violin plot
            st.subheader("🎻 Cost Distribution by Different Factors")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.violin(df, y='charges', x='smoker', color='sex',
                                box=True, title='Charges by Smoking Status and Sex')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.violin(df, y='charges', x='children', color='smoker',
                                box=True, title='Charges by Number of Children and Smoking')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Custom analysis
            st.subheader("🔍 Custom Analysis")
            
            analysis_type = st.selectbox(
                "Choose Analysis Type:",
                ["Average Cost by Categories", "BMI Distribution", "Age Analysis"]
            )
            
            if analysis_type == "Average Cost by Categories":
                group_by = st.multiselect("Group by:", ['sex', 'smoker', 'region'], default=['smoker'])
                if group_by:
                    grouped = df.groupby(group_by)['charges'].mean().reset_index()
                    fig = px.bar(grouped, x=group_by[0] if len(group_by) == 1 else grouped.index,
                                y='charges', title='Average Costs')
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(grouped, use_container_width=True)
            
            elif analysis_type == "BMI Distribution":
                smoker_filter = st.radio("Filter by Smoking Status:", ['All', 'yes', 'no'])
                filtered_df = df if smoker_filter == 'All' else df[df['smoker'] == smoker_filter]
                
                fig = px.histogram(filtered_df, x='bmi', color='sex',
                                    marginal='box', title='BMI Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Age Analysis":
                age_ranges = pd.cut(df['age'], bins=[0, 30, 45, 60, 100],
                                    labels=['18-30', '31-45', '46-60', '60+'])
                df_temp = df.copy()
                df_temp['age_range'] = age_ranges
                
                fig = px.box(df_temp, x='age_range', y='charges', color='smoker',
                            title='Charges by Age Range')
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with Streamlit | KNN Model for Health Insurance Cost Prediction 🏥</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()