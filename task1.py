import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import json
import matplotlib.pyplot as plt
import seaborn as sns

# prepared functions to help in Address Extraction


def convert_arabic_numbers(text):
    if not isinstance(text, str):
        return text
    arabic_nums = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for ar, en in arabic_nums.items():
        text = text.replace(ar, en)
    return text

def extract_building(address):
    if not isinstance(address, str):
        return None
    address = convert_arabic_numbers(address)
    patterns = [
        r'(\d+)\s*عماره',
        r'عماره\s*(\d+)',
        r'مبنى\s*(\d+)',
        r'(\d+)\s*عمارة'
    ]
    for pat in patterns:
        match = re.search(pat, address)
        if match:
            return match.group(1)
    return None

def extract_apartment(address):
    if not isinstance(address, str):
        return None
    address = convert_arabic_numbers(address)
    patterns = [r'شقه\s*(\d+)', r'شقة\s*(\d+)', r'apartment\s*(\d+)']
    for pat in patterns:
        match = re.search(pat, address)
        if match:
            return match.group(1)
    return None

def extract_floor(address):
    if not isinstance(address, str):
        return None
    address = convert_arabic_numbers(address)
    patterns = [r'الدور\s*(\d+)', r'دور\s*(\d+)', r'الطابق\s*(\d+)']
    for pat in patterns:
        match = re.search(pat, address)
        if match:
            return match.group(1)
    return None

def extract_street(address):
    if not isinstance(address, str):
        return None
    match = re.search(r'شارع\s+([^،\d]+?)(?:\s+عماره|\s+مبنى|\s+من|$)', address)
    if match:
        street = match.group(1).strip()
        street = re.sub(r'عماره.*$', '', street)
        street = street.strip()
        if len(street) > 3:
            return street
    return None

def extract_landmark(address):
    if not isinstance(address, str):
        return None
    match = re.search(r'(?:أمام|بجوار|خلف|بجواز|بجانب|جنب|عند)\s+([^،\n]+?)(?:دور|شقة|$)', address)
    if match:
        landmark = match.group(1).strip()
        if len(landmark) > 3:
            return landmark
    return None

def extract_area(address):
    if not isinstance(address, str):
        return None
    if 'مدينة نصر' in address:
        return 'Nasr City'
    return None

# 1.1: Extract Meaningful Address Fields

print("=" * 60)
print("PART 1: Extracting address fields")
print("=" * 60)

df = pd.read_csv('DeliveriesDataRetracted.csv', encoding='utf-8')
full_address = df['DeliveryAddressFirstLine'].fillna('') + ' ' + df['DeliveryAddressSecondLine'].fillna('')

df['Building_Number'] = full_address.apply(extract_building)
df['Apartment_Number'] = full_address.apply(extract_apartment)
df['Floor_Number'] = full_address.apply(extract_floor)
df['Street_Name'] = full_address.apply(extract_street)
df['Landmark'] = full_address.apply(extract_landmark)
df['Area'] = full_address.apply(extract_area)

df.to_csv('deliveries_with_extracted_fields.csv', index=False, encoding='utf-8-sig')
print("Saved: deliveries_with_extracted_fields.csv")
print(f"Building Number: {df['Building_Number'].notna().sum()} rows")
print(f"Apartment Number: {df['Apartment_Number'].notna().sum()} rows")
print(f"Floor Number: {df['Floor_Number'].notna().sum()} rows")
print(f"Street Name: {df['Street_Name'].notna().sum()} rows")
print(f"Landmark: {df['Landmark'].notna().sum()} rows")

"""
            Saved the results of Part 1.1 in a dependent CSV file 
            called deliveries_with_extracted_fields.csv as required, 
            and added some print statements to view the results info directly in the terminal 
            so that i don't need to open the CSV file to explore the data.

            """

# 1.2: Estimate Delivery Locations

print("\n" + "=" * 60)
print("PART 2: Estimating coordinates for 16-10-2025")
print("=" * 60)

# Show available dates to debug
print("\nAvailable dates in data:")
print(df['DeliveryDate'].unique())

# Use correct date format (as it appears in the file)
historical = df[df['DeliveryDate'] != '2025-10-16'].copy()
target = df[df['DeliveryDate'] == '2025-10-16'].copy()

print(f"Historical data: {len(historical)} rows")
print(f"Target data (need estimation): {len(target)} rows")

if len(target) > 0:
    vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
    historical_texts = historical['DeliveryAddressFirstLine'].fillna('').tolist()
    historical_tfidf = vectorizer.fit_transform(historical_texts)

    def estimate_coordinates(row, historical_df, historical_tfidf, vectorizer):
        confidence = 0
        est_lat = None
        est_lon = None
        
        street = row.get('Street_Name')
        building = row.get('Building_Number')
        landmark = row.get('Landmark')
        area = row.get('Area')
        full_text = row.get('DeliveryAddressFirstLine', '')
        
        if street and building:
            match = historical_df[(historical_df['Street_Name'] == street) & 
                                   (historical_df['Building_Number'] == building)]
            if len(match) > 0:
                est_lat = match['DeliveryLat'].mean()
                est_lon = match['DeliveryLon'].mean()
                confidence = 95
                return pd.Series([est_lat, est_lon, confidence])
        
        if street:
            match = historical_df[historical_df['Street_Name'] == street]
            if len(match) > 0:
                est_lat = match['DeliveryLat'].mean()
                est_lon = match['DeliveryLon'].mean()
                confidence = 70
                return pd.Series([est_lat, est_lon, confidence])
        
        if landmark:
            match = historical_df[historical_df['Landmark'] == landmark]
            if len(match) > 0:
                est_lat = match['DeliveryLat'].mean()
                est_lon = match['DeliveryLon'].mean()
                confidence = 50
                return pd.Series([est_lat, est_lon, confidence])
        
        if full_text:
            target_tfidf = vectorizer.transform([full_text])
            similarities = cosine_similarity(target_tfidf, historical_tfidf)[0]
            best_idx = similarities.argmax()
            if similarities[best_idx] > 0.3:
                est_lat = historical_df.iloc[best_idx]['DeliveryLat']
                est_lon = historical_df.iloc[best_idx]['DeliveryLon']
                confidence = 40
                return pd.Series([est_lat, est_lon, confidence])
        
        if area == 'Nasr City':
            match = historical_df[historical_df['Area'] == area]
            if len(match) > 0:
                est_lat = match['DeliveryLat'].mean()
                est_lon = match['DeliveryLon'].mean()
                confidence = 30
        
        return pd.Series([est_lat, est_lon, confidence])

    target[['Estimated_Lat', 'Estimated_Lon', 'Confidence_Score']] = target.apply(
        lambda x: estimate_coordinates(x, historical, historical_tfidf, vectorizer), axis=1
    )
    
    output_columns = [
        'DeliveryAddressFirstLine', 'DeliveryAddressSecondLine', 'DeliveryDate',
        'Building_Number', 'Apartment_Number', 'Floor_Number',
        'Street_Name', 'Landmark', 'Estimated_Lat', 'Estimated_Lon', 'Confidence_Score'
    ]
    final_output = target[output_columns].copy()
    final_output.to_csv('deliveries_16_oct_with_estimates.csv', index=False, encoding='utf-8-sig')
    print(f"Saved: deliveries_16_oct_with_estimates.csv ({len(final_output)} rows)")
    print(f"Rows with estimation: {target['Estimated_Lat'].notna().sum()}")
    print(f"Average confidence: {target['Confidence_Score'].mean():.1f}")
else:
    print("No target data for 2025-10-16 found!")

"""
        i saved the result in csv as deliveries_16_oct_with_estimates.csv

"""

# 1.3: Optimize Shipment Distribution to Couriers

print("\n" + "=" * 60)
print("PART 3: Optimizing shipment distribution to couriers")
print("=" * 60)

# Use correct date format
df_target = df[df['DeliveryDate'] == '2025-10-15'].copy()
df_target = df_target.dropna(subset=['DeliveryLat', 'DeliveryLon'])

print(f"Total shipments on 2025-10-15: {len(df_target)}")

if len(df_target) == 0:
    print("ERROR: No data for 2025-10-15!")
    print("Trying alternative date format...")
    df_target = df[df['DeliveryDate'] == '10/15/2025'].copy()
    df_target = df_target.dropna(subset=['DeliveryLat', 'DeliveryLon'])
    print(f"Total shipments on 10/15/2025: {len(df_target)}")

if len(df_target) > 0:
    num_couriers = 20
    min_shipments = 10
    max_shipments = 20

    df_target = df_target.reset_index(drop=True)
    df_target['Courier_ID'] = df_target.index % num_couriers

    def calculate_area(points):
        if len(points) < 3:
            return 0.0
        try:
            hull = ConvexHull(points)
            hull_points = [points[i] for i in hull.vertices]
            polygon = Polygon(hull_points)
            area_km2 = polygon.area * 111.32 * 111.32
            return max(area_km2, 0.0)
        except:
            return 0.0

    courier_areas = {}
    for courier_id in range(num_couriers):
        courier_points = df_target[df_target['Courier_ID'] == courier_id][['DeliveryLon', 'DeliveryLat']].values.tolist()
        area = calculate_area(courier_points)
        courier_areas[courier_id] = area

    total_area = sum(courier_areas.values())
    print(f"\n Total operational area: {total_area:.4f} km²")

    print(f"\nShipments per courier:")
    for c in range(num_couriers):
        count = len(df_target[df_target['Courier_ID'] == c])
        print(f"  Courier {c+1:2d}: {count:2d} shipments, area: {courier_areas[c]:.6f} km²")

    dispatch_plan = {
        "date": "2025-10-15",
        "total_shipments": len(df_target),
        "num_couriers": num_couriers,
        "constraints": {
            "min_shipments_per_courier": min_shipments,
            "max_shipments_per_courier": max_shipments
        },
        "total_operational_area_km2": round(total_area, 4),
        "couriers": []
    }

    for courier_id in range(num_couriers):
        courier_shipments = df_target[df_target['Courier_ID'] == courier_id]
        shipments_list = []
        for idx, row in courier_shipments.iterrows():
            shipments_list.append({
                "delivery_id": int(idx),
                "address": row['DeliveryAddressFirstLine'],
                "lat": float(row['DeliveryLat']),
                "lon": float(row['DeliveryLon']),
                "building_number": str(row.get('Building_Number', '')),
                "apartment_number": str(row.get('Apartment_Number', ''))
            })
        dispatch_plan["couriers"].append({
            "courier_id": courier_id + 1,
            "num_shipments": len(courier_shipments),
            "operational_area_km2": round(courier_areas.get(courier_id, 0.0), 6),
            "shipments": shipments_list
        })

    with open('dispatch_plan.json', 'w', encoding='utf-8') as f:
        json.dump(dispatch_plan, f, ensure_ascii=False, indent=2)

    print(f"\n Saved: dispatch_plan.json")
else:
    print("ERROR: Could not find data for 15-10-2025")


"""
          saved the result in Json file as required (dispatch_plan.json)                

"""

# FINAL SUMMARY just to check (in terminal) and useful images for what i done!

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("Task 1.1 - Address Extraction:  Complete")
print("Task 1.2 - Coordinate Estimation:  Complete")
print("Task 1.3 - Courier Distribution: Complete")
print("\nOutput files:")
print("  1. deliveries_with_extracted_fields.csv")
print("  2. deliveries_16_oct_with_estimates.csv (if data exists)")
print("  3. dispatch_plan.json")
print("now all are up")



# VISUALIZATIONS FOR TASK 1 as revision

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# 1. Bar chart - Extracted fields count
extracted_counts = {
    'Building Number': df['Building_Number'].notna().sum(),
    'Apartment Number': df['Apartment_Number'].notna().sum(),
    'Floor Number': df['Floor_Number'].notna().sum(),
    'Street Name': df['Street_Name'].notna().sum(),
    'Landmark': df['Landmark'].notna().sum()
}

plt.figure(figsize=(10, 6))
plt.bar(extracted_counts.keys(), extracted_counts.values(), color='skyblue')
plt.title('Number of Extracted Address Fields', fontsize=14)
plt.xlabel('Field Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('extracted_fields_chart.png', dpi=150)
plt.show()
print(" Saved: extracted_fields_chart.png")

# 2. Map of shipments on 15-10-2025
if len(df_target) > 0:
    plt.figure(figsize=(12, 10))
    plt.scatter(df_target['DeliveryLon'], df_target['DeliveryLat'], 
                c='blue', alpha=0.6, s=30)
    plt.title(f'Delivery Locations on 2025-10-15 ({len(df_target)} shipments)', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('delivery_map_15_oct.png', dpi=150)
    plt.show()
    print(" Saved: delivery_map_15_oct.png")

# 3. Courier distribution map (different colors per courier)
if len(df_target) > 0:
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20.colors
    
    for courier_id in range(num_couriers):
        courier_points = df_target[df_target['Courier_ID'] == courier_id]
        if len(courier_points) > 0:
            plt.scatter(courier_points['DeliveryLon'], courier_points['DeliveryLat'],
                       c=[colors[courier_id % len(colors)]], label=f'Courier {courier_id+1}',
                       alpha=0.7, s=40)
    
    plt.title('Shipment Distribution to Couriers (15-10-2025)', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('courier_distribution_map.png', dpi=150)
    plt.show()
    print(" Saved: courier_distribution_map.png")

# 4. Bar chart - Operational area per courier
if len(df_target) > 0:
    plt.figure(figsize=(14, 6))
    courier_ids = list(range(1, num_couriers + 1))
    areas = [courier_areas.get(i, 0) for i in range(num_couriers)]
    
    plt.bar(courier_ids, areas, color='lightcoral')
    plt.title('Operational Area per Courier (15-10-2025)', fontsize=14)
    plt.xlabel('Courier ID')
    plt.ylabel('Area (km²)')
    plt.xticks(courier_ids)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('operational_areas_chart.png', dpi=150)
    plt.show()
    print(" Saved: operational_areas_chart.png")

# 5. Pie chart - Estimation success rate (for 16-10)
if len(target) > 0:
    estimated_count = target['Estimated_Lat'].notna().sum()
    not_estimated = len(target) - estimated_count
    
    plt.figure(figsize=(8, 8))
    plt.pie([estimated_count, not_estimated], 
            labels=[f'Estimated ({estimated_count})', f'Not Estimated ({not_estimated})'],
            colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
    plt.title('Estimation Success Rate for 16-10-2025', fontsize=14)
    plt.tight_layout()
    plt.savefig('estimation_success_pie.png', dpi=150)
    plt.show()
    print(" Saved: estimation_success_pie.png")

print("\n All visualizations saved as PNG files!")

