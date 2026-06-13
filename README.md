# Bosta-Senior-Data-Scientist-Assessment
This project demonstrates hands-on experience in handling unstructured and semi-structured data, applying traditional machine learning techniques, and solving real-world operational optimization problems. It also includes NLP applications for text analysis and a conceptual design for an agentic AI system.

##  Quick Start

### Prerequisites
- Python 3.11+
- vs code (or any Code Editor)

### 1. Clone the repository
```bash
git clone https://github.com/NadiaMoustafa/Bosta-Senior-Data-Scientist-Assessment.git

```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. run the apps by 
```bash
python task1.py

python task2.py

```

## Note:
* Task 1 will generate: 3 files (2 CSVs + 1 JSON) and 5 images (charts)

* When an image appears, close it to view the next one

* All output files are saved in the outputs/ folder for easy review

## TASK1

✅ Part 1.1: 135 building, 771 apartment, 552 street, 342 landmark
✅ Part 1.2: 311/313 estimated (99.4%), avg confidence 42.1%
✅ Part 1.3: 308 shipments, 20 couriers, total area 64.8354 km²

and files :

**deliveries_with_extracted_fields.csv**:   All data + extracted fields Task 1.1
**deliveries_16_oct_with_estimates.csv**:   313 rows + estimates + confidence Task 1.2
**dispatch_plan.json**:                     Distribution plan + area 64.84 km² Task 1.3

## Task 2: Customer Support Bot Analysis

### Overview
Analyzed 678 customer support chatbot conversations to identify conversation patterns, detect handoffs to human agents, and perform sentiment analysis.

---

### 2.1 Conversation Clustering & Handoff Detection

**Methodology:**
- TF-IDF vectorization for text features
- KMeans clustering (5 clusters)
- Keyword-based handoff detection

**Results:**

| Cluster Label | Count | Percentage | Handoff Rate |
|---------------|-------|------------|--------------|
| Shipping/Delivery Issue | 587 | 90.0% | 82.6% |
| General Inquiry | 65 | 10.0% | 56.9% |

**Overall Handoff Rate:** 530 out of 676 conversations (78.4%)

---

### 2.2 Sentiment & Urgency Analysis (Bonus)

**Methodology:** Keyword-based Arabic sentiment analyzer

**Sentiment Distribution:**

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Negative | 365 | 56.0% |
| Neutral | 181 | 27.8% |
| Positive | 106 | 16.3% |

**Urgency Distribution:**

| Urgency Level | Count | Percentage |
|---------------|-------|------------|
| Low | 563 | 86.3% |
| Medium | 81 | 12.4% |
| High | 8 | 1.2% |

---

### 2.3 Agentic AI Design (Conceptual)

**Architecture Components:**
1. Input Processing
2. Intent Classifier (LLM/TF-IDF)
3. Knowledge & APIs (FAQ, Tracking API, Customer Data)
4. Decision Engine (Auto-response vs Handoff)
5. Response Formatter (Text, Buttons, Links)
6. Feedback Loop (Rating, Sentiment, Retraining)

**Efficiency Improvements:**
- Response time: Minutes → 3 seconds for shipping inquiries
- Handoff rate: 80% → 20% (60% reduction)
- Agent time saved: 40%

---

### Task 2 Output Files

| File Name | Description |
|-----------|-------------|
| `conversation_analysis.csv` | Full conversation analysis with cluster labels (Task 2.1) |
| `conversation_clusters.json` | Cluster summary in JSON format (Task 2.1) |
| `conversation_analysis_with_sentiment.csv` | Analysis with sentiment and urgency scores (Task 2.2 - Bonus) |
| `agentic_ai_architecture.png` | Agentic AI system architecture diagram (Task 2.3) |

## Project Contents :

### Python scripts
- task1.py, task2.py
> run these after cloned the repo and install the requirments and it generate the CSVs and Images.

### BDF Document + agentic AI system design

- PDF (All deatils inclueding:explanation of agentic AI system design and other tasks )- Final Report : explaining methodology, assumptions, and results

And outputs folder : Attaced all outputs i created + python code to generate the agentic AI system design.

## Technologies Used
Category               |	Libraries
Data Processing        |	pandas, numpy
Machine Learning & NLP |	scikit-learn (TF-IDF, KMeans)
Geometric Operations	 | scipy (ConvexHull), shapely (Polygon)
Visualization	         | matplotlib, seaborn
