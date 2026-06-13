import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# Load and Parse Conversations

print("=" * 60)
print("TASK 2.1: Customer Support Bot Analysis")
print("=" * 60)

with open('CustomerSupportSample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total conversations: {len(data)}")

# Extract conversations and categories
conversations = []
categories = []
handoff_detected = []

for item in data:
    conv_text = item.get('conversation', '')
    
    # Convert to string if it's not
    if not isinstance(conv_text, str):
        conv_text = str(conv_text) if conv_text is not None else ''
    
    category = item.get('category', 'Unknown')
    
    conversations.append(conv_text)
    categories.append(category)
    
    # Detect handoff (bot -> manual agent)
    if isinstance(conv_text, str) and len(conv_text) > 0:
        if re.search(r'(مندوب خدمه|خدمة العملاء|تحويل|كلم خدمه|اتواصل مع خدمه|اتصل ب|ضروووووري|ضروري)', conv_text, re.IGNORECASE):
            handoff_detected.append(1)
        else:
            handoff_detected.append(0)
    else:
        handoff_detected.append(0)

# Create DataFrame
df_conversations = pd.DataFrame({
    'conversation_id': range(len(conversations)),
    'conversation_text': conversations,
    'provided_category': categories,
    'handoff_detected': handoff_detected
})

# Remove empty conversations
df_conversations = df_conversations[df_conversations['conversation_text'].str.len() > 10]
df_conversations = df_conversations.reset_index(drop=True)

print(f"Valid conversations after cleaning: {len(df_conversations)}")
print(f"Handoff detected in: {df_conversations['handoff_detected'].sum()} conversations")

# Clustering using TF-IDF + KMeans

print("\n" + "=" * 60)
print("Clustering conversations by main reason")
print("=" * 60)

# Clean conversation text for clustering
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove agent/bot markers and keep user messages mostly
    text = re.sub(r'Agent:.*?(?=User:|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'User:', '', text)
    # Keep Arabic letters and spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text.lower().strip()

df_conversations['clean_text'] = df_conversations['conversation_text'].apply(clean_text)

# Remove empty cleaned texts
df_conversations = df_conversations[df_conversations['clean_text'].str.len() > 5]
df_conversations = df_conversations.reset_index(drop=True)

print(f"Conversations for clustering: {len(df_conversations)}")

if len(df_conversations) < 10:
    print("Not enough conversations for clustering!")
    exit()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=100, stop_words=None, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df_conversations['clean_text'])

# Use 5 clusters
n_clusters = min(5, len(df_conversations))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_conversations['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Get top keywords per cluster
def get_top_keywords(cluster_id, n_words=5):
    cluster_indices = df_conversations[df_conversations['cluster'] == cluster_id].index.tolist()
    if not cluster_indices:
        return []
    cluster_texts = df_conversations.loc[cluster_indices, 'clean_text'].tolist()
    all_text = ' '.join(cluster_texts)
    words = re.findall(r'[\u0600-\u06FF]{3,}', all_text)
    word_counts = Counter(words).most_common(n_words)
    return [w for w, c in word_counts]

cluster_labels = {}
for i in range(n_clusters):
    keywords = get_top_keywords(i)
    keywords_str = ' '.join(keywords)
    
    if any(k in keywords_str for k in ['شحن', 'توصيل', 'استلام', 'اوردر', 'طلب']):
        cluster_labels[i] = 'Shipping/Delivery Issue'
    elif any(k in keywords_str for k in ['حساب', 'اكونت', 'تسجيل', 'دخول']):
        cluster_labels[i] = 'Account Problem'
    elif any(k in keywords_str for k in ['دفع', 'فاتورة', 'سعر', 'فلوس']):
        cluster_labels[i] = 'Payment/Billing'
    elif any(k in keywords_str for k in ['كلم', 'خدمه', 'عملاء', 'تحويل', 'مندوب']):
        cluster_labels[i] = 'Request Human Agent'
    else:
        cluster_labels[i] = 'General Inquiry'

df_conversations['cluster_label'] = df_conversations['cluster'].map(cluster_labels)

# Display Results

print(f"\nNumber of clusters: {n_clusters}")
print("\nCluster Distribution:")
cluster_counts = df_conversations['cluster_label'].value_counts()
for label, count in cluster_counts.items():
    print(f"  {label}: {count} conversations ({count/len(df_conversations)*100:.1f}%)")

print(f"\nHandoff detection per cluster:")
for label in cluster_counts.index:
    cluster_handoff = df_conversations[df_conversations['cluster_label'] == label]['handoff_detected'].sum()
    cluster_total = len(df_conversations[df_conversations['cluster_label'] == label])
    if cluster_total > 0:
        print(f"  {label}: {cluster_handoff}/{cluster_total} handoffs ({cluster_handoff/cluster_total*100:.1f}%)")

# Save Results

df_conversations.to_csv('conversation_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\n Saved: conversation_analysis.csv")

"""
   saved the results in conversation_analysis.csv to be clair for revision
"""

# Save summary JSON
summary = {
    "total_conversations": len(df_conversations),
    "handoff_count": int(df_conversations['handoff_detected'].sum()),
    "clusters": []
}

for label, count in cluster_counts.items():
    cluster_data = df_conversations[df_conversations['cluster_label'] == label]
    summary["clusters"].append({
        "cluster_name": label,
        "count": int(count),
        "percentage": round(count/len(df_conversations)*100, 1),
        "handoff_count": int(cluster_data['handoff_detected'].sum()),
        "sample_conversation": cluster_data['conversation_text'].iloc[0][:300] if len(cluster_data) > 0 else ""
    })

with open('conversation_clusters.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(" Saved: conversation_clusters.json")

# Bonus: Sentiment Analysis (Simple) - Bonus: Sentiment & Urgency Analysis

print("\n" + "=" * 60)
print("BONUS: Sentiment & Urgency Analysis")
print("=" * 60)

def simple_sentiment(text):
    if not isinstance(text, str):
        return 'Neutral', 'Low'
    
    positive_words = ['شكرا', 'تمام', 'ممتاز', 'رائع', 'حلو', 'زي الفل']
    negative_words = ['مشكلة', 'خطأ', 'غلط', 'ضيق', 'زعلان', 'متأخر', 'ضروووري', 'عايز', 'محتاج']
    urgency_words = ['ضروري', 'عاجل', 'بسرعة', 'اسرع', 'ضروووري', 'حالاً']

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    urgency = sum(1 for w in urgency_words if w in text_lower)
    
    if neg_count > pos_count:
        sentiment = 'Negative'
    elif pos_count > neg_count:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    
    if urgency > 0:
        urgency_level = 'High' if urgency > 1 else 'Medium'
    else:
        urgency_level = 'Low'
    
    return sentiment, urgency_level

df_conversations[['sentiment', 'urgency']] = df_conversations['conversation_text'].apply(
    lambda x: pd.Series(simple_sentiment(x))
)

print("\nSentiment Distribution:")
sentiment_counts = df_conversations['sentiment'].value_counts()
for sent, count in sentiment_counts.items():
    print(f"  {sent}: {count} conversations ({count/len(df_conversations)*100:.1f}%)")

print("\nUrgency Distribution:")
urgency_counts = df_conversations['urgency'].value_counts()
for urg, count in urgency_counts.items():
    print(f"  {urg}: {count} conversations ({count/len(df_conversations)*100:.1f}%)")

# Update CSV with sentiment
df_conversations.to_csv('conversation_analysis_with_sentiment.csv', index=False, encoding='utf-8-sig')
print("\n Saved: conversation_analysis_with_sentiment.csv")

print("\n" + "=" * 60)
print("All is DONE!")
print("=" * 60)