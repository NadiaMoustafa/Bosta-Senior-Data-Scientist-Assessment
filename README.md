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
pip install -r requirements.txt

### 4. run the apps by 

NLP (regex, TF-IDF)، Optimization (KMeans + convex hull)، Confidence scores


Files saved:
File Size Purpose
deliveries_with_extracted_fields.csv: All data + extracted fields Task 1.1
deliveries_16_oct_with_estimates.csv :313 rows + estimates + confidence Task 1.2
dispatch_plan.json: Distribution plan + area 64.84 km² Task 1.3


✅ Part 1.1: 135 building, 771 apartment, 552 street, 342 landmark
✅ Part 1.2: 311/313 estimated (99.4%), avg confidence 42.1%
✅ Part 1.3: 308 shipments, 20 couriers, total area 64.8354 km²

when you run file one task1.py

it creates 
3 file (2 csvs and 1 json)
5 iamges as charts
note when appears to you the image you must close it after review it to show another one



What's the file name and what does it contain?

conversation_analysis.csv: All conversations + cluster + handoff to see the complete result.

conversation_clusters.json: Summary of clustering in JSON for automation.

conversation_analysis_with_sentiment.csv: Same as above + sentiment + urgency for the bonus.

Assumptions made in Task 1:
1. All deliveries are within Nasr City, Cairo
2. Missing coordinates (16-10) are estimated using historical matching
3. Operational area is calculated using convex hull of delivery points
4. Courier assignment is based on location clustering (KMeans)
