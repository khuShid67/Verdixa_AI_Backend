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
        "symptoms": str(row["Symptoms"]),
        "organic_treatment": str(row["Organic Treatment"]),
        "chemical_treatment": str(row["Chemical Treatment"]),
        "prevention": str(row["Prevention"]),
    }

def get_recommendation(disease_name):
    return recommendations.get(
        normalize(disease_name),
        {"description": "No recommendation available."}
    )