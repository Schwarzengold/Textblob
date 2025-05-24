import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

df = pd.read_csv('product-reviews-english.csv')

def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'Позитивний'
    elif polarity < -0.1:
        return 'Негативний'
    else:
        return 'Нейтральний'

df['sentiment'] = df['Review Title'].apply(get_sentiment)
counts = df['sentiment'].value_counts().reindex(['Позитивний', 'Нейтральний', 'Негативний'], fill_value=0)
plt.figure(figsize=(7, 5))
colors = ['green', 'blue', 'red']
counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Кількість позитивних, нейтральних і негативних відгуків')
plt.xlabel('Тип відгуку')
plt.ylabel('Кількість')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(counts):
    plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()
