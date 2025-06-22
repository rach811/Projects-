# Game Monetization Analysis Dashboard
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import streamlit as st

# -------------------------
# Dashboard Setup
# -------------------------
st.set_page_config(page_title="Game Monetization Analysis", layout="wide")
st.title("📱 Game Monetization & Success Analysis")

# -------------------------
# Data Loading and Preparation
# -------------------------
@st.cache_data
def load_data():
    file_path = "Downloads/Copy of appstore_games.xlsx"
    df = pd.read_excel(file_path)
    
    # Clean and engineer features
    df['Has In-App Purchases'] = df['In-app Purchases'].notna()
    df['Is Free'] = df['Price'] == 0.0
    df['User Rating Count'] = df['User Rating Count'].fillna(0)
    df['Average User Rating'] = df['Average User Rating'].fillna(0)
    df['Year'] = df['Original Release Date'].dt.year
    
    # Handle genres
    df['Primary Genre'] = df['Primary Genre'].fillna('Unknown')
    top_genres = df['Primary Genre'].value_counts().nlargest(10).index
    df['Top Genre'] = df['Primary Genre'].apply(lambda x: x if x in top_genres else 'Other')
    
    # Define success (top 25% by rating count)
    rating_threshold = df['User Rating Count'].quantile(0.75)
    df['Successful'] = df['User Rating Count'] >= rating_threshold
    
    # Add min IAP price feature
    def get_min_iap_price(iap):
        if pd.isna(iap) or iap == '':
            return 0.0
        try:
            prices = [float(p.strip()) for p in str(iap).split(',')]
            return min(prices)
        except:
            return 0.0
    
    df['Min In-app Purchase Price'] = df['In-app Purchases'].apply(get_min_iap_price)
    
    return df

df = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filter Games")
selected_genre = st.sidebar.multiselect(
    "Select Genre(s)", 
    options=df['Top Genre'].unique(), 
    default=list(df['Top Genre'].unique())
)

is_free_filter = st.sidebar.radio("Free or Paid?", ["All", "Free", "Paid"])
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(int(df['Year'].min()), int(df['Year'].max()))
)

# Apply filters
filtered_df = df[
    (df['Top Genre'].isin(selected_genre)) & 
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1])
]

if is_free_filter == "Free":
    filtered_df = filtered_df[filtered_df['Is Free']]
elif is_free_filter == "Paid":
    filtered_df = filtered_df[~filtered_df['Is Free']]

# -------------------------
# Key Metrics Overview
# -------------------------
st.header("📊 Key Metrics Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Games", len(filtered_df))
col2.metric("Free Games", f"{filtered_df['Is Free'].mean():.1%}")
col3.metric("Games with IAP", f"{filtered_df['Has In-App Purchases'].mean():.1%}")
col4.metric("Successful Games", f"{filtered_df['Successful'].mean():.1%}")

# -------------------------
# EDA Visualizations
# -------------------------
st.header("🔍 Exploratory Data Analysis")

# Tab layout for EDA
eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
    "Monetization Impact", 
    "Genre Analysis", 
    "Trends Over Time", 
    "Game Attributes"
])

with eda_tab1:
    st.subheader("Monetization Impact on Ratings")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: In-App Purchases vs Ratings
    sns.boxplot(
        data=filtered_df[filtered_df['User Rating Count'] > 0], 
        x='Has In-App Purchases', 
        y='User Rating Count',
        ax=ax1
    )
    ax1.set_yscale('log')
    ax1.set_title("User Ratings by In-App Purchase Availability")
    ax1.set_xlabel("Has In-App Purchases")
    ax1.set_ylabel("User Rating Count (log scale)")
    
    # Plot 2: Free vs Paid Games
    sns.boxplot(
        data=filtered_df[filtered_df['User Rating Count'] > 0], 
        x='Is Free', 
        y='User Rating Count',
        ax=ax2
    )
    ax2.set_yscale('log')
    ax2.set_title("User Ratings: Free vs Paid Games")
    ax2.set_xlabel("Is the Game Free?")
    ax2.set_ylabel("User Rating Count (log scale)")
    
    st.pyplot(fig)
    plt.clf()

with eda_tab2:
    st.subheader("Genre Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top genres overall
        top_genres = filtered_df['Top Genre'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        top_genres.plot(kind='barh')
        plt.title("Top Genres by Count")
        plt.xlabel("Number of Games")
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())
        plt.clf()
    
    with col2:
        # Top genres with IAP
        genre_df = filtered_df[filtered_df['Has In-App Purchases']].copy()
        if not genre_df.empty:
            top_genre_iap = genre_df['Top Genre'].value_counts().head(10)
            plt.figure(figsize=(10, 6))
            top_genre_iap.plot(kind='barh', color='orange')
            plt.title("Top Genres with In-App Purchases")
            plt.xlabel("Number of Games")
            plt.gca().invert_yaxis()
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            st.warning("No games with In-App Purchases in current filters")

with eda_tab3:
    st.subheader("Trends Over Time")
    
    # Prepare time series data
    time_df = filtered_df.groupby('Year').agg({
        'Has In-App Purchases': 'mean',
        'Is Free': 'mean',
        'Successful': 'mean',
        'ID': 'count'
    }).rename(columns={'ID': 'Game Count'})
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Monetization trends
    time_df[['Has In-App Purchases', 'Is Free']].plot(marker='o', ax=ax1)
    ax1.set_title("Monetization Trends Over Time")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Proportion")
    ax1.grid(True)
    ax1.legend(["IAP Adoption", "Free Games"])
    
    # Plot 2: Success rate and game count
    ax2b = ax2.twinx()
    time_df['Successful'].plot(marker='o', color='green', ax=ax2)
    time_df['Game Count'].plot(marker='s', color='purple', ax=ax2b)
    ax2.set_title("Success Rate and Game Releases Over Time")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Success Rate", color='green')
    ax2b.set_ylabel("Number of Games", color='purple')
    ax2.grid(True)
    
    st.pyplot(fig)
    plt.clf()

with eda_tab4:
    st.subheader("Game Attributes Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Size vs Rating
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=filtered_df.dropna(), 
            x='Size', 
            y='Average User Rating',
            alpha=0.5
        )
        plt.title("Game Size vs. Average Rating")
        plt.xlabel("Game Size (bytes)")
        plt.ylabel("Average Rating")
        st.pyplot(plt.gcf())
        plt.clf()
    
    with col2:
        # Price distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(
            filtered_df[filtered_df['Price'] > 0]['Price'],
            bins=20,
            kde=True
        )
        plt.title("Distribution of Game Prices (Paid Games Only)")
        plt.xlabel("Price ($)")
        st.pyplot(plt.gcf())
        plt.clf()

# -------------------------
# Predictive Modeling Section
# -------------------------
st.header("🤖 Predictive Modeling Insights")

model_tab1, model_tab2 = st.tabs(["Success Prediction", "Description Analysis"])

with model_tab1:
    st.subheader("Predicting Game Success")
    
    # Prepare features
    features = ['Price', 'Min In-app Purchase Price', 'Size', 'Has In-App Purchases', 'Is Free', 'Top Genre']
    model_df = df[features + ['Successful']].dropna()
    model_df = pd.get_dummies(model_df, columns=['Top Genre'], drop_first=True)
    
    # Train model
    X = model_df.drop('Successful', axis=1)
    y = model_df['Successful']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Show results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Classification Report")
        st.code(classification_report(y_test, y_pred))
    
    with col2:
        st.markdown("#### Feature Importances")
        importances = pd.Series(clf.feature_importances_, index=X.columns)
        plt.figure(figsize=(8, 6))
        importances.nlargest(10).plot(kind='barh')
        plt.title("Top Features Predicting Success")
        st.pyplot(plt.gcf())
        plt.clf()

with model_tab2:
    st.subheader("Game Description Analysis")
    
    df['Description'] = df['Description'].fillna('')
    successful_text = " ".join(df[df['Successful']]['Description'])
    unsuccessful_text = " ".join(df[~df['Successful']]['Description'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Successful Games - Common Words")
        wc1 = WordCloud(width=800, height=400, background_color='white').generate(successful_text)
        st.image(wc1.to_array(), use_column_width=True)
    
    with col2:
        st.markdown("#### Unsuccessful Games - Common Words")
        wc2 = WordCloud(width=800, height=400, background_color='white').generate(unsuccessful_text)
        st.image(wc2.to_array(), use_column_width=True)

# -------------------------
# Conclusion
# -------------------------
st.markdown("---")
st.header("🎯 Key Takeaways")
st.markdown("""
1. **Monetization Impact**: Games with in-app purchases tend to have higher rating counts, 
   suggesting better engagement or visibility.
2. **Free vs Paid**: Free games dominate the market and receive more ratings on average.
3. **Genre Trends**: Certain genres are more likely to implement in-app purchases than others.
4. **Success Predictors**: Price, in-app purchases, and game size are important factors in predicting success.
5. **Descriptions Matter**: Successful games use different language in their descriptions compared to unsuccessful ones.
""")

st.markdown("---")
st.markdown("Created as part of a game monetization analysis project. 💼")
