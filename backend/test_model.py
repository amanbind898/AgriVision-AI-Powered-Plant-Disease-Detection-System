"""
Test the plant disease model with test images
"""
import sys
import os
from pathlib import Path
from PIL import Image
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.disease_model_pytorch import DiseaseModel

# Expected labels based on filename
EXPECTED_LABELS = {
    'AppleCedarRust': 'Apple___Cedar_apple_rust',
    'AppleScab': 'Apple___Apple_scab',
    'CornCommonRust': 'Corn_(maize)___Common_rust_',
    'PotatoEarlyBlight': 'Potato___Early_blight',
    'PotatoHealthy': 'Potato___healthy',
    'TomatoEarlyBlight': 'Tomato___Early_blight',
    'TomatoHealthy': 'Tomato___healthy',
    'TomatoYellowCurlVirus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
}

def get_expected_label(filename):
    """Extract expected label from filename"""
    for key, value in EXPECTED_LABELS.items():
        if filename.startswith(key):
            return value
    return None

def test_model():
    """Test model on all test images"""
    print("=" * 80)
    print("TESTING PLANT DISEASE MODEL")
    print("=" * 80)
    
    # Initialize model
    print("\n📦 Loading model...")
    model = DiseaseModel()
    
    # Get test images
    test_dir = Path("../ml-training/data/test")
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    test_images = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    print(f"✅ Found {len(test_images)} test images\n")
    
    # Test each image
    results = []
    correct = 0
    total = 0
    
    for img_path in sorted(test_images):
        filename = img_path.name
        expected = get_expected_label(filename)
        
        if expected is None:
            print(f"⚠️  Skipping {filename} - unknown expected label")
            continue
        
        # Load and predict
        image = Image.open(img_path)
        prediction = model.predict(image)
        
        # Check if correct
        is_correct = prediction['predicted_class'] == expected
        correct += is_correct
        total += 1
        
        # Format expected label
        exp_plant, exp_disease = model.format_disease_name(expected)
        
        # Print result
        status = "✅" if is_correct else "❌"
        print(f"{status} {filename}")
        print(f"   Expected: {exp_plant} - {exp_disease}")
        print(f"   Predicted: {prediction['plant_name']} - {prediction['disease_name']}")
        print(f"   Confidence: {prediction['confidence']:.2f}%")
        
        if not is_correct:
            print(f"   Top 5 predictions:")
            for i, pred in enumerate(prediction['top_5_predictions'][:3], 1):
                print(f"      {i}. {pred['plant_name']} - {pred['disease_name']} ({pred['confidence']:.2f}%)")
        
        print()
        
        # Store result
        results.append({
            'filename': filename,
            'expected': expected,
            'predicted': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'correct': is_correct
        })
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Total Images: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Group by disease type
    print("\n📊 Results by Disease Type:")
    disease_stats = {}
    for result in results:
        expected = result['expected']
        if expected not in disease_stats:
            disease_stats[expected] = {'correct': 0, 'total': 0}
        disease_stats[expected]['total'] += 1
        if result['correct']:
            disease_stats[expected]['correct'] += 1
    
    for disease, stats in sorted(disease_stats.items()):
        plant, disease_name = model.format_disease_name(disease)
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {plant} - {disease_name}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Save results to JSON
    output_file = "test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    test_model()
