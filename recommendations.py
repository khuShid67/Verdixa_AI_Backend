import pandas as pd

df = pd.read_excel("plant_disease_recommendations_detailed.xlsx")
df = df.fillna("")
recommendations = {}

for _, row in df.iterrows():

    recommendations[row["Disease"]] = {
        "plant": str(row["Plant"]) if pd.notna(row["Plant"]) else "",
        "status": str(row["Status"]) if pd.notna(row["Status"]) else "",
        "severity": str(row["Severity"]) if pd.notna(row["Severity"]) else "",
        "description": str(row["Description"]) if pd.notna(row["Description"]) else "",
        "symptoms": str(row["Symptoms"]) if pd.notna(row["Symptoms"]) else "",
        "organic_treatment": str(row["Organic Treatment"]) if pd.notna(row["Organic Treatment"]) else "",
        "chemical_treatment": str(row["Chemical Treatment"]) if pd.notna(row["Chemical Treatment"]) else "",
        "prevention": str(row["Prevention"]) if pd.notna(row["Prevention"]) else "",
    }


def get_recommendation(disease_name):

    return recommendations.get(
        disease_name,
        {
            "description": "No recommendation available."
        }
    )