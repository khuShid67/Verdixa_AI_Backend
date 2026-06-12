import pandas as pd

df = pd.read_excel("plant_disease_recommendations_detailed.xlsx")
df = df.fillna("")

def normalize(text):
    return str(text).strip().lower()

recommendations = {}

for _, row in df.iterrows():
    disease = normalize(row["Disease"])

    recommendations[disease] = {
        "plant": str(row["Plant"]),
        "status": str(row["Status"]),
        "severity": str(row["Severity"]),
        "description": str(row["Description"]),

        # ✅ FIX: convert to list
        "symptoms": [
            s.strip()
            for s in str(row["Symptoms"]).split(",")
            if s.strip()
        ],

        "organic_treatment": str(row["Organic Treatment"]),
        "chemical_treatment": str(row["Chemical Treatment"]),
        "prevention": str(row["Prevention"]),
    }


def get_recommendation(disease_name):
    key = normalize(disease_name)

    return recommendations.get(key, {
        "plant": "",
        "status": "Unknown",
        "severity": "Unknown",
        "description": "No recommendation available",

        "symptoms": [],
        "organic_treatment": "Not available",
        "chemical_treatment": "Not available",
        "prevention": "Not available"
    })