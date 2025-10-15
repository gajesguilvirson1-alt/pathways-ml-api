"""
Flask API for Strand and Track Recommendation System
Provides REST endpoints for ML-based recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from strand_recommender import StrandRecommendationSystem

app = Flask(__name__)
CORS(app)

# Global recommender instance
recommender = None

def initialize_system():
    """Initialize the recommendation system"""
    global recommender
    
    try:
        # Check if models exist
        if os.path.exists('models/track_model.pkl'):
            print("Loading existing models...")
            recommender = StrandRecommendationSystem()
            recommender.load_models('models')
        else:
            print("Training new models...")
            # Use student data file
            data_file = 'data/student_data.csv'
            if not os.path.exists(data_file):
                print(f"Error: Data file not found at {data_file}")
                print("Please ensure student_data.csv exists in the data directory.")
                return False
            
            # Train new models
            recommender = StrandRecommendationSystem(data_file)
            X, y_track, y_strand = recommender.load_data()
            recommender.train_models(X, y_track, y_strand)
            recommender.save_models('models')
        
        print("Recommendation system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_ready': recommender is not None
    })

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get strand and track recommendations for a student"""
    try:
        if recommender is None:
            return jsonify({'error': 'Recommendation system not initialized'}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['assessment_scores', 'interests', 'hobbies', 'work_preferences', 'demographics']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert student data to feature vector
        features = convert_student_data_to_features(data)
        
        # Get recommendations with improved consistency
        top_n = data.get('top_n', 3)
        add_noise = data.get('add_noise', False)  # Disabled by default for consistency
        consistency_mode = data.get('consistency_mode', True)  # Enable consistency mode by default
        recommendations = recommender.predict_recommendations(features, top_n=top_n, add_noise=add_noise, consistency_mode=consistency_mode)
        
        # Format response
        response = {
            'student_id': data.get('student_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from trained models"""
    try:
        if recommender is None:
            return jsonify({'error': 'Recommendation system not initialized'}), 500
        
        importance = recommender.get_feature_importance()
        
        # Convert to JSON-serializable format
        response = {
            'track_importance': importance['track_importance'].head(20).to_dict('records'),
            'strand_importance': importance['strand_importance'].head(20).to_dict('records'),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_models():
    """Retrain models with new data"""
    try:
        global recommender
        
        # Check if new data file is provided
        data = request.get_json()
        data_file = data.get('data_file', 'data/student_data.csv')
        
        if not os.path.exists(data_file):
            return jsonify({'error': f'Data file not found: {data_file}'}), 400
        
        # Retrain models
        recommender = StrandRecommendationSystem(data_file)
        X, y_track, y_strand = recommender.load_data()
        results = recommender.train_models(X, y_track, y_strand)
        recommender.save_models('models')
        
        response = {
            'message': 'Models retrained successfully',
            'track_accuracy': results['track_accuracy'],
            'strand_accuracy': results['strand_accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_student_data_to_features(student_data):
    """Convert student data to feature vector"""
    if recommender is None or recommender.feature_names is None:
        raise ValueError("Recommender not properly initialized")
    
    # Initialize feature vector
    features = np.zeros(len(recommender.feature_names))
    
    # Assessment scores
    assessment_scores = student_data.get('assessment_scores', {})
    for skill, score in assessment_scores.items():
        feature_name = f'assessment_{skill}'
        if feature_name in recommender.feature_names:
            idx = recommender.feature_names.index(feature_name)
            features[idx] = float(score)
    
    # Interests
    interests = student_data.get('interests', [])
    for interest in interests:
        feature_name = f'interest_{interest}'
        if feature_name in recommender.feature_names:
            idx = recommender.feature_names.index(feature_name)
            features[idx] = 1
    
    # Hobbies
    hobbies = student_data.get('hobbies', [])
    for hobby in hobbies:
        feature_name = f'hobby_{hobby}'
        if feature_name in recommender.feature_names:
            idx = recommender.feature_names.index(feature_name)
            features[idx] = 1
    
    # Work preferences
    work_prefs = student_data.get('work_preferences', [])
    for pref in work_prefs:
        feature_name = f'work_pref_{pref}'
        if feature_name in recommender.feature_names:
            idx = recommender.feature_names.index(feature_name)
            features[idx] = 1
    
    # Demographics
    demographics = student_data.get('demographics', {})
    
    # Age
    if 'age' in demographics:
        idx = recommender.feature_names.index('age')
        features[idx] = float(demographics['age'])
    
    # Gender
    gender = demographics.get('gender', '').lower()
    if gender == 'male' and 'gender_male' in recommender.feature_names:
        idx = recommender.feature_names.index('gender_male')
        features[idx] = 1
    elif gender == 'female' and 'gender_female' in recommender.feature_names:
        idx = recommender.feature_names.index('gender_female')
        features[idx] = 1
    
    # GPA
    if 'gpa' in demographics:
        idx = recommender.feature_names.index('gpa')
        features[idx] = float(demographics['gpa'])
    
    # Family income
    income = demographics.get('family_income', '').lower()
    if income == 'low' and 'income_low' in recommender.feature_names:
        idx = recommender.feature_names.index('income_low')
        features[idx] = 1
    elif income == 'middle' and 'income_middle' in recommender.feature_names:
        idx = recommender.feature_names.index('income_middle')
        features[idx] = 1
    elif income == 'high' and 'income_high' in recommender.feature_names:
        idx = recommender.feature_names.index('income_high')
        features[idx] = 1
    
    return features

@app.route('/sample-request', methods=['GET'])
def get_sample_request():
    """Get a sample request format for testing"""
    sample = {
        "student_id": "STU_12345",
        "assessment_scores": {
            "mathematics": 85,
            "science": 88,
            "communication": 75,
            "critical_thinking": 80,
            "problem_solving": 87,
            "creativity": 70,
            "leadership": 65,
            "technical_knowledge": 82,
            "research": 78,
            "analysis": 83
        },
        "interests": [
            "technology",
            "research",
            "innovation",
            "analysis"
        ],
        "hobbies": [
            "coding",
            "gaming",
            "science_experiments",
            "reading"
        ],
        "work_preferences": [
            "analytical",
            "independent",
            "challenging",
            "innovative"
        ],
        "demographics": {
            "age": 16,
            "gender": "male",
            "gpa": 3.5,
            "family_income": "middle"
        },
        "top_n": 3
    }
    
    return jsonify({
        "sample_request": sample,
        "endpoint": "/recommend",
        "method": "POST",
        "description": "Send this JSON structure to get recommendations"
    })

if __name__ == '__main__':
    print("Starting Strand Recommendation API...")
    
    # Initialize the system
    if initialize_system():
        print("System ready! Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize system. Exiting...")
        exit(1)