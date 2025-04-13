# FAANG Stock Price Prediction

## **Project Overview**
The goal of this project is to predict stock prices for FAANG companies (Facebook, Amazon, Apple, Netflix, Google) using historical stock data. The project follows a structured workflow encompassing data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment. The **MLflow** library was utilized to track experiments, compare models, register the best-performing model, and deploy it to a Streamlit application for real-time predictions.

---
## **Project Structure**
    FAANG/
    │
    ├── app.py                  # Streamlit application for deployment
    ├── faang.ipynb             # Jupyter notebook for experimentation and model development
    ├── FAANG - FAANG.csv       # Historical stock data for FAANG companies
    ├── readme.md               # Project README file
    │
    ├── mlartifacts/            # Directory for storing MLflow artifacts
    │
    ├── mlruns/                 # Directory for MLflow experiment tracking
    │
    └── myenv/                  # Virtual environment or dependency management folder
    

## **Workflow Steps**
### 1. **Data Preprocessing**
Preprocessing steps were crucial for cleaning and preparing the data. Key steps included:

#### Handling Missing Values
- Checked for missing values and handled them appropriately (e.g., imputation or removal).

#### Outlier Removal
- Applied **IQR Clipping** to remove extreme outliers, ensuring robust model performance.

#### Scaling
- Standardized numeric features (`Open`, `High`, `Low`, `Volume`,`Day_of_Week`,`Ticker_AMZN`,`Ticker_GOOGL`,`Ticker_NFLX`) using **StandardScaler**.

#### Log Transformation
- Transformed the target variable (`Close Price`) into `Log_Close` to address skewness and stabilize variance.

---
### 2. **Exploratory Data Analysis (EDA)**
EDA was performed to gain insights into the data and identify patterns and relationships. Key steps included:

#### Descriptive Statistics
- Summarized data with metrics such as mean, median, standard deviation, and percentiles.
- Used boxplots to identify outliers in numerical columns such as `Open`, `High`, `Low`, `Close`, `Volume`, and `Adj Close`.

#### Correlation Analysis
- Generated heatmaps to visualize correlations among features.
- Highlighted features most correlated with the target variable (`Close Price`).

#### Distribution and Trends
- Analyzed variable distributions to check for skewness and normality.
- Visualized time-series trends for stock prices (`Close`, `Open`, `High`, `Low`) over time.

#### Stock Ticker Analysis
- Bar plots were used to analyze the distribution of different FAANG stocks (Amazon, Google, Netflix, etc.) in the dataset.

---

### 3. **Feature Engineering**
Feature engineering aimed to enhance model performance by creating and selecting meaningful predictors. Key steps included:

#### Feature Selection
- Selected predictors based on domain knowledge and correlation analysis:
  - `Open`, `High`, `Low`, `Volume`, `Day_of_Week`, `Ticker_AMZN`, `Ticker_GOOGL`, `Ticker_NFLX`.

#### Feature Creation
- Extracted `Day_of_Week` from the `Date` column to capture weekly trends.
- Used one-hot encoding to represent stock tickers (`Ticker_AMZN`, `Ticker_GOOGL`, `Ticker_NFLX`).

#### Correlation Insights
- Analyzed relationships between features to identify strong predictors for the target variable.

---

### 4. **Model Training and Evaluation**
Several machine learning models were trained, evaluated, and compared using metrics like R-squared (R²) and Mean Squared Error (MSE). The models included:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest Regressor


#### Model Performance Metrics
- **R² (Coefficient of Determination)**: Measures the proportion of variance explained by the model.
- **MSE (Mean Squared Error)**: Measures the average squared difference between predictions and actual values.
- **Cross-Validation**: Used 5-fold cross-validation to evaluate model stability and performance.

---

### 5. **MLflow Integration**
MLflow was utilized for experiment tracking, model comparison, and deployment.

#### Experiment Tracking
- Logged metrics such as R², MSE, Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
- Tracked hyperparameters and performance for each model.

#### Model Registration
- Registered the best-performing model (Random Forest Regressor) in the MLflow Model Registry.

#### Model Loading
- Loaded the registered model using MLflow for real-time predictions in a Streamlit application.

---

### 6. **Deployment**
The final Random Forest model was deployed using Streamlit for real-time stock price predictions. The application allows users to input feature values and receive predictions for `Close Price`.

---

## **Libraries Used**
- **NumPy**: Numerical computations.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning model development.
- **Plotly**: Interactive visualizations.
- **MLflow**: Experiment tracking and model deployment.
- **Streamlit**: Web application framework for deployment.

---

## **Results**
- The Random Forest Regressor outperformed other models with the highest R² and lowest MSE on the test set. R^2: 0.968
MSE: 0.073
- The deployed application successfully predicts `Close Price` using the trained model, demonstrating the end-to-end workflow.

---

## **Usage**
1. Clone the repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Input feature values in the web interface to predict the stock `Close Price`.

---
© 2025 MuraliKrishnaGR. All rights reserved.

## **Future Work**
- Include Facebook (Meta) and Apple in Analysis: Extend the analysis by incorporating data from Facebook (Meta) and     Apple to cover all FAANG companies comprehensively.
- Experiment with deep learning models (e.g., LSTMs) for sequential data.
