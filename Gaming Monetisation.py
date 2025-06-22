# Game Monetization Analysis with Machine Learning and NLP

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import numpy as np
import streamlit as st

# Load data
file_path = "Downloads/Copy of appstore_games.xlsx"
df = pd.read_excel(file_path)

#view data 
df.head(), df.info()
print(df.columns)

# Clean and prepare data
df_clean = df.copy()
df_clean['Has In-App Purchases'] = df_clean['In-app Purchases'].notna()
df_clean['Is Free'] = df_clean['Price'] == 0.0
df['User Rating Count'] = df['User Rating Count'].fillna(0)
df['Average User Rating'] = df['Average User Rating'].fillna(0)
df['Year'] = df['Original Release Date'].dt.year

# Fill missing rating counts with 0 for analysis
df_clean['User Rating Count'] = df_clean['User Rating Count'].fillna(0)

# Simplify genres
df['Primary Genre'] = df['Primary Genre'].fillna('Unknown')
top_genres = df['Primary Genre'].value_counts().nlargest(10).index
df['Top Genre'] = df['Primary Genre'].apply(lambda x: x if x in top_genres else 'Other')

# Binary target: was this game relatively successful (top 25%)?
rating_threshold = df['User Rating Count'].quantile(0.75)
df['Successful'] = df['User Rating Count'] >= rating_threshold

# -------------------------
# EDA Plots 
#In-App Purchases vs. Success (Ratings)
sns.boxplot(data=df_clean[df_clean['User Rating Count'] > 0], x='Has In-App Purchases', y='User Rating Count')
plt.yscale('log')
plt.title("User Ratings by In-App Purchase Availability")
plt.xlabel("Has In-App Purchases")
plt.ylabel("User Rating Count (log scale)")
plt.show()

#Free vs Paid Games: Which Gets More Ratings?
sns.boxplot(data=df_clean[df_clean['User Rating Count'] > 0], x='Is Free', y='User Rating Count')
plt.yscale('log')
plt.title("User Ratings: Free vs Paid Games")
plt.xlabel("Is the Game Free?")
plt.ylabel("User Rating Count (log scale)")
plt.show()

#Genre Breakdown of Monetisation
genre_df = df_clean[df_clean['Has In-App Purchases']].copy()
top_genres = genre_df['Primary Genre'].value_counts().head(10)

plt.yscale('linear')  # reset any log scale on Y-axis
top_genres.plot(kind='barh')
plt.title("Top Genres by Count")
plt.xlabel("Number of Games")
plt.tight_layout()
plt.show()

#Trends Over Time: In-App Purchase Strategy
df_clean['Year'] = df_clean['Original Release Date'].dt.year
iap_by_year = df_clean.groupby('Year')['Has In-App Purchases'].mean()

iap_by_year.plot(marker='o')
plt.title("Adoption of In-App Purchases Over Time")
plt.xlabel("Year")
plt.ylabel("Proportion with IAP")
plt.grid(True)
plt.show()

#Does Game Size or Language Support Correlate with Ratings?
# Correlation check
correlation_data = df_clean[['Average User Rating', 'Size']].dropna()
sns.scatterplot(data=correlation_data, x='Size', y='Average User Rating')
plt.title("Game Size vs. Average Rating")
plt.xlabel("Game Size (bytes)")
plt.ylabel("Average Rating")
plt.show()

# -------------------------
# Feature 1: Predicting Game Success
# -------------------------
  
def get_min_iap_price(iap):
    if pd.isna(iap) or iap == '':
        return 0.0
    try:
        prices = [float(p.strip()) for p in str(iap).split(',')]
        return min(prices)
    except:
        return 0.0  # fallback in case something unexpected slips through
df['Min In-app Purchase Price'] = df['In-app Purchases'].apply(get_min_iap_price)


# Final cleaned feature list
features = ['Price', 'Min In-app Purchase Price', 'Size', 'Top Genre']

# Select features and target
model_df = df[features + ['Successful']].dropna()

# Convert 'Top Genre' into dummy variables
model_df = pd.get_dummies(model_df, columns=['Top Genre'], drop_first=True)

# Separate features and target
X = model_df.drop('Successful', axis=1)
y = model_df['Successful']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred))
print(df['In-app Purchases'].unique()[:20])

# Feature Importance
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importances - Predicting Game Success")
plt.show()

# -------------------------
# Feature 2: NLP on Descriptions
# -------------------------
# Fill missing descriptions
df['Description'] = df['Description'].fillna('')

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_text = vectorizer.fit_transform(df['Description'])

# Compare word clouds for successful vs unsuccessful games
successful_text = " ".join(df[df['Successful']]['Description'])
unsuccessful_text = " ".join(df[~df['Successful']]['Description'])

# Word clouds
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
wc1 = WordCloud(width=800, height=400, background_color='white').generate(successful_text)
wc2 = WordCloud(width=800, height=400, background_color='white').generate(unsuccessful_text)

axs[0].imshow(wc1, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Successful Games - Keywords')

axs[1].imshow(wc2, interpolation='bilinear')
axs[1].axis('off')
axs[1].set_title('Unsuccessful Games - Keywords')

plt.tight_layout()
plt.show()

# -------------------------
# Conclusion
# - Random forest identified in-app purchases and being free as key features.
# - NLP revealed distinct vocabulary differences in game descriptions.
# - These insights could guide marketing and monetization strategy.

