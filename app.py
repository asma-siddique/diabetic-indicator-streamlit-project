# Import all required libraries at the TOP of the file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from scipy import stats
import plotly.express as px

# Rest of your code continues exactly as before...
# [Keep all the existing code below these imports]

# Configure page
st.set_page_config(
    page_title="Diabetes Health Analysis",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stSelectbox, .stSlider {margin-bottom: 1.5rem;}
    .stDataFrame {margin-top: 1rem;}
    .stAlert {margin: 1rem 0;}
    .plot-container {margin: 2rem 0;}
    .stMarkdown h3 {color: #2e86ab;}
    .feature-card {border-radius: 10px; padding: 1rem; margin: 0.5rem 0; background-color: #f8f9fa;}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('diabetes_dataset.csv')
        st.success("Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# ======================
# DATA PREPROCESSING
# ======================
def preprocess_data(df):
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='median')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Create BMI categories for EDA (we'll keep this in the EDA version)
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 35, 40, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 
                                      'Obese I', 'Obese II', 'Obese III'])
    
    # Remove outliers
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numerical_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Prepare features and target
    X = df.drop(['Diabetes_012', 'BMI_Category'], axis=1, errors='ignore')  # Exclude BMI_Category for modeling
    y = df['Diabetes_012']
    
    return df, X, y  # Return both versions


# ======================
# EDA FUNCTIONS
# ======================

def show_summary_statistics(df):
    st.subheader("📊 Summary Statistics")
    
    # Basic statistics
    st.write("**Numerical Features:**")
    st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'))
    
    # Categorical statistics
    st.write("**Categorical Features:**")
    cat_stats = df.select_dtypes(include=['object', 'category']).describe().T
    st.dataframe(cat_stats)
    
    # Unique value counts
    st.write("**Unique Value Counts:**")
    unique_counts = pd.DataFrame(df.nunique(), columns=['Unique Values'])
    st.dataframe(unique_counts)

def show_missing_values_analysis(df):
    st.subheader("🔍 Missing Values Analysis")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.warning(f"Found {missing.sum()} missing values across {len(missing[missing > 0])} columns")
        st.dataframe(missing[missing > 0].rename('Missing Count').sort_values(ascending=False))
        
        # Show missing value heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        ax.set_title('Missing Values Heatmap')
        st.pyplot(fig)

def show_outlier_detection(df):
    st.subheader("📌 Outlier Detection")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    selected_col = st.selectbox("Select feature for outlier analysis:", numerical_cols)
    
    # Boxplot
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df[selected_col], ax=ax1)
    ax1.set_title(f'Boxplot of {selected_col}')
    
    # Histogram with outliers highlighted
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    q1 = df[selected_col].quantile(0.25)
    q3 = df[selected_col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[selected_col] < (q1 - 1.5*iqr)) | (df[selected_col] > (q3 + 1.5*iqr))]
    
    sns.histplot(df[selected_col], kde=True, ax=ax2)
    ax2.axvspan(outliers[selected_col].min(), outliers[selected_col].max(), 
                color='red', alpha=0.3, label='Outliers')
    ax2.set_title(f'Distribution of {selected_col} with Outliers Highlighted')
    ax2.legend()
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
    
    st.write(f"**Outlier Statistics for {selected_col}:**")
    st.write(f"- Lower bound: {q1 - 1.5*iqr:.2f}")
    st.write(f"- Upper bound: {q3 + 1.5*iqr:.2f}")
    st.write(f"- Number of outliers: {len(outliers)}")

def show_feature_distributions(df):
    st.subheader("📈 Feature Distributions")
    
    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox(
            "Select feature to visualize:",
            df.select_dtypes(include=[np.number]).columns
        )
    with col2:
        plot_type = st.selectbox(
            "Select plot type:",
            ['Histogram', 'Boxplot', 'Violin Plot', 'ECDF']
        )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if plot_type == 'Histogram':
        sns.histplot(df[feature], kde=True, ax=ax, bins=30)
    elif plot_type == 'Boxplot':
        sns.boxplot(x=df[feature], ax=ax)
    elif plot_type == 'Violin Plot':
        sns.violinplot(x=df[feature], ax=ax)
    elif plot_type == 'ECDF':
        sns.ecdfplot(df[feature], ax=ax)
    
    ax.set_title(f'{plot_type} of {feature}')
    st.pyplot(fig)
    
    st.write(f"**Statistics for {feature}:**")
    st.dataframe(df[feature].describe().to_frame().T)

def show_correlation_analysis(df):
    st.subheader("🔄 Correlation Analysis")
    
    # Numerical correlation
    st.write("**Numerical Feature Correlation:**")
    corr_matrix = df.select_dtypes(include=['number']).corr()
    fig1, ax1 = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                center=0, linewidths=0.5, ax=ax1)
    ax1.set_title('Feature Correlation Matrix')
    st.pyplot(fig1)
    
    # Top correlations with target
    st.write("**Top Correlations with Diabetes Status:**")
    target_corr = corr_matrix['Diabetes_012'].sort_values(ascending=False)[1:]
    st.dataframe(target_corr.to_frame('Correlation'))
    
    # Pairplot sample
    st.write("**Pairwise Feature Relationships (Sample):**")
    sample_df = df.sample(500) if len(df) > 500 else df
    fig2 = sns.pairplot(sample_df[['Diabetes_012', 'BMI', 'Age', 'GenHlth', 'PhysHlth']], 
                       hue='Diabetes_012', palette='viridis')
    st.pyplot(fig2)

def show_trend_analysis(df):
    st.subheader("📈 Trend Analysis")
    
    # Since we don't have time data, we'll use Age as a proxy
    st.write("**Diabetes Prevalence by Age Group:**")
    
    age_diabetes = df.groupby('Age')['Diabetes_012'].value_counts(normalize=True).unstack()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    age_diabetes.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Diabetes Prevalence by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Percentage')
    ax.legend(title='Diabetes Status', labels=['No', 'Prediabetes', 'Diabetes'])
    st.pyplot(fig)
    
    st.write("**Key Observation:** Diabetes prevalence increases with age, with the highest rates in older age groups.")
# ======================
# MODEL FUNCTIONS
# ======================

def train_and_evaluate_model(X, y):
    st.subheader("🤖 Model Training")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    with st.spinner("Training Random Forest model..."):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    st.success("Model trained successfully!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("F1 Score", f"{f1:.2%}")
    with col3:
        st.metric("Classes", str(model.n_classes_))
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Top 10 Important Features')
    st.pyplot(fig)
    
    return model, scaler
def show_grouped_analysis(df):
    st.subheader("📊 Grouped Analysis")
    
    # Get available grouping columns (only show columns that exist in the DataFrame)
    available_group_cols = []
    for col in ['BMI_Category', 'Age', 'GenHlth', 'Income']:
        if col in df.columns:
            available_group_cols.append(col)
    
    if not available_group_cols:
        st.warning("No valid grouping columns available!")
        return
    
    group_col = st.selectbox(
        "Select grouping feature:",
        available_group_cols
    )
    
    # Get available analysis columns
    available_agg_cols = []
    for col in ['Diabetes_012', 'BMI', 'PhysHlth', 'MentHlth']:
        if col in df.columns:
            available_agg_cols.append(col)
    
    if not available_agg_cols:
        st.warning("No valid analysis columns available!")
        return
    
    agg_col = st.selectbox(
        "Select feature to analyze:",
        available_agg_cols
    )
    
    try:
        # Grouped statistics
        grouped_stats = df.groupby(group_col)[agg_col].agg(['mean', 'median', 'std', 'count'])
        st.write(f"**Grouped Statistics of {agg_col} by {group_col}:**")
        st.dataframe(grouped_stats.style.background_gradient(cmap='Blues'))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        if agg_col == 'Diabetes_012':
            sns.barplot(x=group_col, y=agg_col, data=df, ax=ax, ci=None)
        else:
            sns.boxplot(x=group_col, y=agg_col, data=df, ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f'{agg_col} Distribution by {group_col}')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in grouped analysis: {str(e)}")
        st.error("Please select different grouping or analysis columns")

def prediction_interface(model, scaler):
    st.subheader("🔮 Diabetes Risk Prediction")
    st.write("Fill in all health indicators below to assess diabetes risk:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            highbp = st.selectbox('High Blood Pressure', [0, 1])
            highchol = st.selectbox('High Cholesterol', [0, 1])
            cholcheck = st.selectbox('Cholesterol Check in Past 5 Years', [0, 1])
            bmi = st.slider('BMI', 10, 100, 25)
            smoker = st.selectbox('Smoked at least 100 cigarettes', [0, 1])
            stroke = st.selectbox('Ever had a stroke', [0, 1])
            heartdisease = st.selectbox('Heart disease or attack', [0, 1])
            physactivity = st.selectbox('Physical activity', [0, 1])
            fruits = st.selectbox('Eats fruits daily', [0, 1])
            veggies = st.selectbox('Eats vegetables daily', [0, 1])
            
        with col2:
            hvyalcoholconsump = st.selectbox('Heavy alcohol consumption', [0, 1])
            anyhealthcare = st.selectbox('Has healthcare coverage', [0, 1])
            nodocbccost = st.selectbox('Could not see doctor due to cost', [0, 1])
            genhlth = st.slider('General Health (1=Excellent, 5=Poor)', 1, 5, 3)
            menthlth = st.slider('Days of poor mental health (last 30 days)', 0, 30, 0)
            physhlth = st.slider('Days of poor physical health (last 30 days)', 0, 30, 0)
            diffwalk = st.selectbox('Difficulty walking/climbing stairs', [0, 1])
            sex = st.selectbox('Sex (0=Female, 1=Male)', [0, 1])
            age = st.slider('Age Group', 1, 13, 7)
            education = st.slider('Education Level (1-6)', 1, 6, 3)
            income = st.slider('Income Level (1-8)', 1, 8, 4)
        
        submitted = st.form_submit_button("Predict Diabetes Risk")
        
        if submitted:
            # Create input data as numpy array with correct dtype
            input_data = np.array([[
                int(highbp), int(highchol), int(cholcheck), float(bmi), 
                int(smoker), int(stroke), int(heartdisease),
                int(physactivity), int(fruits), int(veggies), 
                int(hvyalcoholconsump), int(anyhealthcare),
                int(nodocbccost), int(genhlth), int(menthlth), 
                int(physhlth), int(diffwalk), int(sex), int(age),
                int(education), int(income)
            ]], dtype=np.float64)  # Explicitly set dtype
            
            try:
                input_scaled = scaler.transform(input_data)
                prediction = int(model.predict(input_scaled)[0])  # Ensure integer prediction
                proba = model.predict_proba(input_scaled)[0]
                
                diabetes_status = ['No Diabetes', 'Prediabetes', 'Diabetes']
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                cols = st.columns(3)
                with cols[1]:
                    st.metric("Predicted Status", diabetes_status[prediction])
                
                st.write("**Probability Estimates:**")
                proba_df = pd.DataFrame({
                    'Status': diabetes_status,
                    'Probability': proba
                }).set_index('Status')
                st.dataframe(proba_df.style.background_gradient(cmap='Blues'))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=diabetes_status, y=proba, palette='viridis', ax=ax)
                ax.set_ylabel('Probability')
                ax.set_title('Diabetes Risk Probability Distribution')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.error("Please ensure all features are provided as numbers (0-1 for binary, integers for others)")
# ======================
# MAIN APP
# ======================

def main():
    st.title("Diabetes Health Indicators Analysis")
    st.markdown("""
    **A comprehensive analysis of diabetes risk factors** using the CDC's BRFSS survey data.
    This interactive dashboard allows you to explore the dataset and predict diabetes risk.
    """)
    
    # Load and preprocess data
    raw_df = load_data()
    df, X, y = preprocess_data(raw_df)  # Get both versions of the data
    
    menu = ["🏠 Introduction", "🔍 EDA", "🤖 ML Model", "📝 Conclusion"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    if choice == "🏠 Introduction":
        st.header("Introduction to the Project")
        st.markdown("""
        ### Project Overview
        This project analyzes diabetes risk factors from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) data.
        
        **Key Objectives:**
        1. Perform comprehensive exploratory data analysis (EDA)
        2. Identify key risk factors for diabetes
        3. Build and evaluate predictive models
        4. Create an interactive risk assessment tool
        
        **Dataset Information:**
        - Source: [Kaggle Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
        - Rows: 253,680 survey responses
        - Features: 21 health-related variables
        - Target: Diabetes status (0=No, 1=Prediabetes, 2=Diabetes)
        """)
        
        st.subheader("Quick Dataset Overview")
        st.dataframe(df.head())
        
        st.subheader("Data Dictionary")
        st.write("""
        - **HighBP/HighChol**: Binary indicators
        - **BMI**: Continuous body mass index
        - **Smoker/Stroke/HeartDisease**: Binary health indicators
        - **PhysActivity**: Physical activity in past 30 days
        - **Fruits/Veggies**: Daily consumption
        - **GenHlth**: Self-rated health (1-5 scale)
        - **MentHlth/PhysHlth**: Days of poor health
        - **DiffWalk**: Mobility difficulty
        - **Age/Education/Income**: Demographic factors
        """)
        
    elif choice == "🔍 EDA":
        st.header("Exploratory Data Analysis")
        
        analysis = st.selectbox(
            "Select Analysis Type:",
            [
                "Summary Statistics",
                "Missing Values Analysis",
                "Outlier Detection",
                "Feature Distributions",
                "Correlation Analysis",
                "Grouped Analysis",
                "Trend Analysis"
            ]
        )
        
        if analysis == "Summary Statistics":
            show_summary_statistics(df)
        elif analysis == "Missing Values Analysis":
            show_missing_values_analysis(df)
        elif analysis == "Outlier Detection":
            show_outlier_detection(df)
        elif analysis == "Feature Distributions":
            show_feature_distributions(df)
        elif analysis == "Correlation Analysis":
            show_correlation_analysis(df)
        elif analysis == "Grouped Analysis":
            show_grouped_analysis(df)
        elif analysis == "Trend Analysis":
            show_trend_analysis(df)
            
    elif choice == "🤖 ML Model":
        st.header("Machine Learning Model")
        
        tab1, tab2 = st.tabs(["Model Training", "Risk Prediction"])
        
        with tab1:
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    st.session_state.model, st.session_state.scaler = train_and_evaluate_model(X, y)
                    st.session_state.feature_names = X.columns.tolist()
                    st.success("Model trained successfully!")
            
            if 'model' in st.session_state:
                st.write("### Model Information")
                st.write(f"Number of features: {len(st.session_state.feature_names)}")
                st.write("Feature names in order:")
                st.write(st.session_state.feature_names)
                st.write("Top 10 important features:")
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': st.session_state.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(feature_importance.head(10))
        
        with tab2:
            if 'model' in st.session_state:
                st.write("### Prediction Interface")
                st.write(f"Using {len(st.session_state.feature_names)} features:")
                prediction_interface(st.session_state.model, st.session_state.scaler)
            else:
                st.warning("Please train the model first in the 'Model Training' tab.")
    
    elif choice == "📝 Conclusion":
        st.header("Conclusion and Key Findings")
        st.markdown("""
        ### Key Takeaways
        
        **From EDA:**
        - Strongest diabetes predictors: HighBP, HighChol, BMI, GenHlth, PhysHlth, DiffWalk
        - Clear age-related trends in diabetes prevalence
        - Significant income-based disparities in diabetes rates
        - Physical activity shows protective effects across all groups
        
        **From Modeling:**
        - Random Forest achieved good performance despite class imbalance
        - Top important features align with EDA findings
        - Model can effectively stratify risk levels
        
        **Recommendations:**
        - Target high-risk populations for preventive interventions
        - Focus on modifiable risk factors (BMI, physical activity)
        - Consider socioeconomic factors in public health planning
        
        ### Future Work
        - Address class imbalance with advanced techniques
        - Experiment with ensemble methods
        - Develop personalized risk scoring system
        - Integrate with clinical decision support systems
        """)

if __name__ == '__main__':
    main()