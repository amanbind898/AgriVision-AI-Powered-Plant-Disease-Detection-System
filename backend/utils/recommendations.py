from typing import Dict, List

# Disease treatment database
DISEASE_TREATMENTS = {
    "Apple scab": {
        "fungicides": ["Captan", "Mancozeb", "Myclobutanil"],
        "precautions": [
            "Remove and destroy infected leaves",
            "Prune trees to improve air circulation",
            "Apply fungicide in early spring before bud break",
            "Avoid overhead watering"
        ],
        "organic_options": ["Neem oil", "Copper-based fungicides", "Sulfur sprays"]
    },
    "Black rot": {
        "fungicides": ["Captan", "Thiophanate-methyl", "Ziram"],
        "precautions": [
            "Remove mummified fruits and infected wood",
            "Prune during dry weather",
            "Apply fungicide at pink bud stage",
            "Maintain proper spacing between plants"
        ],
        "organic_options": ["Bordeaux mixture", "Copper fungicides"]
    },
    "Cedar apple rust": {
        "fungicides": ["Myclobutanil", "Propiconazole", "Triadimefon"],
        "precautions": [
            "Remove nearby cedar trees if possible",
            "Apply fungicide from pink to petal fall",
            "Plant resistant varieties",
            "Rake and destroy fallen leaves"
        ],
        "organic_options": ["Sulfur-based fungicides"]
    },
    "Powdery mildew": {
        "fungicides": ["Sulfur", "Potassium bicarbonate", "Myclobutanil"],
        "precautions": [
            "Improve air circulation around plants",
            "Avoid overhead watering",
            "Remove infected plant parts",
            "Apply fungicide at first sign of disease"
        ],
        "organic_options": ["Baking soda solution", "Neem oil", "Milk spray (1:9 ratio)"]
    },
    "Cercospora leaf spot": {
        "fungicides": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
        "precautions": [
            "Rotate crops annually",
            "Remove crop debris after harvest",
            "Avoid working in wet fields",
            "Use disease-free seeds"
        ],
        "organic_options": ["Copper-based fungicides"]
    },
    "Common rust": {
        "fungicides": ["Azoxystrobin", "Propiconazole", "Trifloxystrobin"],
        "precautions": [
            "Plant resistant hybrids",
            "Apply fungicide at first sign of disease",
            "Scout fields regularly",
            "Remove volunteer corn plants"
        ],
        "organic_options": ["Sulfur sprays"]
    },
    "Northern Leaf Blight": {
        "fungicides": ["Azoxystrobin", "Pyraclostrobin", "Propiconazole"],
        "precautions": [
            "Use resistant hybrids",
            "Rotate with non-host crops",
            "Bury crop residue",
            "Apply fungicide at tasseling stage"
        ],
        "organic_options": ["Copper fungicides"]
    },
    "Bacterial spot": {
        "fungicides": ["Copper-based bactericides", "Streptomycin"],
        "precautions": [
            "Use disease-free seeds and transplants",
            "Avoid overhead irrigation",
            "Remove and destroy infected plants",
            "Rotate crops for 2-3 years",
            "Disinfect tools between plants"
        ],
        "organic_options": ["Copper hydroxide", "Copper sulfate"]
    },
    "Early blight": {
        "fungicides": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
        "precautions": [
            "Rotate crops (3-4 year rotation)",
            "Mulch around plants",
            "Water at base of plants",
            "Remove lower leaves as they yellow",
            "Space plants properly for air circulation"
        ],
        "organic_options": ["Copper fungicides", "Bacillus subtilis"]
    },
    "Late blight": {
        "fungicides": ["Chlorothalonil", "Mancozeb", "Cymoxanil"],
        "precautions": [
            "Use certified disease-free seed potatoes",
            "Destroy volunteer plants",
            "Apply fungicide preventively in wet weather",
            "Remove and destroy infected plants immediately",
            "Avoid overhead irrigation"
        ],
        "organic_options": ["Copper-based fungicides", "Bordeaux mixture"]
    },
    "Leaf Mold": {
        "fungicides": ["Chlorothalonil", "Mancozeb", "Copper fungicides"],
        "precautions": [
            "Improve greenhouse ventilation",
            "Reduce humidity below 85%",
            "Space plants for air circulation",
            "Remove infected leaves",
            "Avoid overhead watering"
        ],
        "organic_options": ["Sulfur sprays", "Potassium bicarbonate"]
    },
    "Septoria leaf spot": {
        "fungicides": ["Chlorothalonil", "Mancozeb", "Copper fungicides"],
        "precautions": [
            "Remove infected leaves immediately",
            "Mulch to prevent soil splash",
            "Stake and prune for air circulation",
            "Rotate crops for 3 years",
            "Water at base of plants"
        ],
        "organic_options": ["Copper-based fungicides", "Neem oil"]
    },
    "Spider mites": {
        "fungicides": ["Abamectin", "Bifenazate", "Spiromesifen"],
        "precautions": [
            "Spray plants with water to dislodge mites",
            "Remove heavily infested leaves",
            "Maintain adequate moisture",
            "Introduce predatory mites",
            "Avoid excessive nitrogen fertilization"
        ],
        "organic_options": ["Neem oil", "Insecticidal soap", "Horticultural oil"]
    },
    "Target Spot": {
        "fungicides": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
        "precautions": [
            "Remove crop debris",
            "Rotate crops",
            "Improve air circulation",
            "Apply fungicide preventively",
            "Avoid overhead irrigation"
        ],
        "organic_options": ["Copper fungicides"]
    },
    "Tomato Yellow Leaf Curl Virus": {
        "fungicides": ["No fungicide effective - viral disease"],
        "precautions": [
            "Control whitefly vectors with insecticides",
            "Use reflective mulches",
            "Remove and destroy infected plants",
            "Plant resistant varieties",
            "Use insect-proof screens in greenhouses",
            "Remove weeds that harbor whiteflies"
        ],
        "organic_options": ["Neem oil for whitefly control", "Yellow sticky traps"]
    },
    "Tomato mosaic virus": {
        "fungicides": ["No fungicide effective - viral disease"],
        "precautions": [
            "Use virus-free seeds",
            "Disinfect tools with 10% bleach solution",
            "Wash hands before handling plants",
            "Remove and destroy infected plants",
            "Control aphid vectors",
            "Avoid tobacco use near plants"
        ],
        "organic_options": ["Insecticidal soap for aphid control"]
    },
    "Leaf scorch": {
        "fungicides": ["Captan", "Chlorothalonil"],
        "precautions": [
            "Remove and destroy infected leaves",
            "Improve air circulation",
            "Avoid overhead watering",
            "Mulch to prevent soil splash",
            "Plant in well-drained soil"
        ],
        "organic_options": ["Copper fungicides", "Neem oil"]
    },
    "Leaf blight": {
        "fungicides": ["Mancozeb", "Chlorothalonil", "Copper fungicides"],
        "precautions": [
            "Remove infected plant material",
            "Improve air circulation",
            "Water at base of plants",
            "Apply fungicide preventively",
            "Rotate crops"
        ],
        "organic_options": ["Copper-based fungicides", "Bordeaux mixture"]
    },
    "Esca": {
        "fungicides": ["No effective fungicide - trunk disease"],
        "precautions": [
            "Prune during dry weather",
            "Protect pruning wounds",
            "Remove and destroy infected wood",
            "Avoid water stress",
            "Maintain vine vigor through proper nutrition"
        ],
        "organic_options": ["Wound protectants", "Proper pruning techniques"]
    },
    "Haunglongbing": {
        "fungicides": ["No cure available - bacterial disease"],
        "precautions": [
            "Remove and destroy infected trees",
            "Control Asian citrus psyllid vectors",
            "Use certified disease-free nursery stock",
            "Apply systemic insecticides",
            "Monitor trees regularly for symptoms"
        ],
        "organic_options": ["Neem oil for psyllid control", "Kaolin clay"]
    }
}

def get_treatment_recommendations(plant_name: str, disease_name: str, is_healthy: bool) -> Dict:
    """
    Get treatment recommendations for a plant disease
    
    Args:
        plant_name: Name of the plant
        disease_name: Name of the disease
        is_healthy: Whether the plant is healthy
    
    Returns:
        Dictionary with treatment recommendations
    """
    if is_healthy:
        return {
            "status": "healthy",
            "message": f"Your {plant_name} appears healthy! Keep up the good care.",
            "preventive_measures": [
                "Maintain proper watering schedule",
                "Ensure adequate sunlight",
                "Use balanced fertilizer",
                "Monitor regularly for early signs of disease",
                "Maintain good air circulation",
                "Practice crop rotation"
            ]
        }
    
    # Find matching disease in database
    treatment = None
    for disease_key, treatment_data in DISEASE_TREATMENTS.items():
        if disease_key.lower() in disease_name.lower():
            treatment = treatment_data
            break
    
    if not treatment:
        # Generic recommendations if specific disease not found
        return {
            "status": "diseased",
            "message": f"Disease detected in {plant_name}: {disease_name}",
            "fungicides": ["Consult local agricultural extension for specific recommendations"],
            "precautions": [
                "Remove and destroy infected plant parts",
                "Improve air circulation",
                "Avoid overhead watering",
                "Apply appropriate fungicide",
                "Monitor plant regularly"
            ],
            "organic_options": ["Neem oil", "Copper-based fungicides"],
            "note": "Consult with a local agricultural expert for specific treatment"
        }
    
    return {
        "status": "diseased",
        "message": f"Disease detected in {plant_name}: {disease_name}",
        "fungicides": treatment["fungicides"],
        "precautions": treatment["precautions"],
        "organic_options": treatment["organic_options"],
        "note": "Always follow label instructions when applying any treatment"
    }

def get_all_diseases() -> List[str]:
    """Get list of all diseases in the database"""
    return list(DISEASE_TREATMENTS.keys())
