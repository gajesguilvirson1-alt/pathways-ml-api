"""
Machine Learning Strand and Track Recommendation System
Uses scikit-learn to recommend educational strands and tracks for students
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StrandRecommendationSystem:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.track_model = None
        self.strand_model = None
        self.feature_scaler = StandardScaler()
        self.track_encoder = LabelEncoder()
        self.strand_encoder = LabelEncoder()
        self.feature_names = None
        self.track_strand_mapping = {
            'Academic': ['STEM – Science Technology Engineering and Mathematics', 'ABM – Accountancy Business and Management', 'HUMSS – Humanities and Social Sciences', 'GAS – General Academic Strand'],
            'Technical-Vocational (TechPro)': ['Information and Communication Technology (ICT)', 'Industrial Arts (IA)', 'Home Economics (HE)', 'Agri-Fishery Arts (AFA)']
        }
        
    def load_data(self, data_path=None):
        """Load and preprocess the dataset"""
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("Data path not provided")
            
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Process the data
        processed_data = self._preprocess_data(df)
        return processed_data
    
    def _preprocess_data(self, df):
        """Preprocess the raw dataset"""
        print("Preprocessing data...")
        
        # Create feature matrix
        features = []
        
        # Assessment scores
        assessment_cols = [col for col in df.columns if col.startswith('assessment_')]
        for col in assessment_cols:
            features.append(df[col].fillna(0).values)
        
        # Process interests, hobbies, work preferences
        all_interests = set()
        all_hobbies = set()
        all_work_prefs = set()
        
        for _, row in df.iterrows():
            if pd.notna(row['interests']):
                all_interests.update(row['interests'].split(','))
            if pd.notna(row['hobbies']):
                all_hobbies.update(row['hobbies'].split(','))
            if pd.notna(row['work_preferences']):
                all_work_prefs.update(row['work_preferences'].split(','))
        
        # Create binary features for interests
        for interest in sorted(all_interests):
            interest_feature = df['interests'].fillna('').str.contains(interest, case=False).astype(int)
            features.append(interest_feature.values)
        
        # Create binary features for hobbies
        for hobby in sorted(all_hobbies):
            hobby_feature = df['hobbies'].fillna('').str.contains(hobby, case=False).astype(int)
            features.append(hobby_feature.values)
        
        # Create binary features for work preferences
        for pref in sorted(all_work_prefs):
            pref_feature = df['work_preferences'].fillna('').str.contains(pref, case=False).astype(int)
            features.append(pref_feature.values)
        
        # Add demographic features
        features.append(df['age'].fillna(df['age'].mean()).values)
        features.append((df['gender'] == 'Male').astype(int).values)
        features.append((df['gender'] == 'Female').astype(int).values)
        features.append(df['gpa'].fillna(df['gpa'].mean()).values)
        
        # Add family income features
        features.append((df['family_income'] == 'Low').astype(int).values)
        features.append((df['family_income'] == 'Middle').astype(int).values)
        features.append((df['family_income'] == 'High').astype(int).values)
        
        # Combine all features
        X = np.column_stack(features)
        
        # Create feature names
        feature_names = []
        feature_names.extend(assessment_cols)
        feature_names.extend([f"interest_{interest}" for interest in sorted(all_interests)])
        feature_names.extend([f"hobby_{hobby}" for hobby in sorted(all_hobbies)])
        feature_names.extend([f"work_pref_{pref}" for pref in sorted(all_work_prefs)])
        feature_names.extend(['age', 'gender_male', 'gender_female', 'gpa'])
        feature_names.extend(['income_low', 'income_middle', 'income_high'])
        
        self.feature_names = feature_names
        
        # Get labels
        y_track = df['target_track'].values
        y_strand = df['target_strand'].values
        
        return X, y_track, y_strand
    
    def train_models(self, X, y_track, y_strand, test_size=0.2, random_state=42):
        """Train both track and strand recommendation models"""
        print("Training models...")
        
        # Split the data
        X_train, X_test, y_track_train, y_track_test, y_strand_train, y_strand_test = train_test_split(
            X, y_track, y_strand, test_size=test_size, random_state=random_state, stratify=y_track
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Encode labels
        y_track_train_encoded = self.track_encoder.fit_transform(y_track_train)
        y_track_test_encoded = self.track_encoder.transform(y_track_test)
        y_strand_train_encoded = self.strand_encoder.fit_transform(y_strand_train)
        y_strand_test_encoded = self.strand_encoder.transform(y_strand_test)
        
        # Train Track Recommendation Model
        print("Training track recommendation model...")
        track_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        track_rf = RandomForestClassifier(random_state=random_state)
        track_grid = GridSearchCV(track_rf, track_params, cv=5, scoring='accuracy', n_jobs=-1)
        track_grid.fit(X_train_scaled, y_track_train_encoded)
        
        self.track_model = track_grid.best_estimator_
        
        # Train Strand Recommendation Model
        print("Training strand recommendation model...")
        strand_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        strand_rf = RandomForestClassifier(random_state=random_state)
        strand_grid = GridSearchCV(strand_rf, strand_params, cv=5, scoring='accuracy', n_jobs=-1)
        strand_grid.fit(X_train_scaled, y_strand_train_encoded)
        
        self.strand_model = strand_grid.best_estimator_
        
        # Evaluate models
        print("\nModel Evaluation:")
        
        # Track model evaluation
        track_pred = self.track_model.predict(X_test_scaled)
        track_accuracy = accuracy_score(y_track_test_encoded, track_pred)
        print(f"Track Recommendation Accuracy: {track_accuracy:.4f}")
        
        # Strand model evaluation
        strand_pred = self.strand_model.predict(X_test_scaled)
        strand_accuracy = accuracy_score(y_strand_test_encoded, strand_pred)
        print(f"Strand Recommendation Accuracy: {strand_accuracy:.4f}")
        
        # Detailed classification reports
        print("\nTrack Classification Report:")
        print(classification_report(y_track_test_encoded, track_pred, 
                                  target_names=self.track_encoder.classes_))
        
        print("\nStrand Classification Report:")
        print(classification_report(y_strand_test_encoded, strand_pred, 
                                  target_names=self.strand_encoder.classes_))
        
        return {
            'track_accuracy': track_accuracy,
            'strand_accuracy': strand_accuracy,
            'track_model': self.track_model,
            'strand_model': self.strand_model
        }
    
    def predict_recommendations(self, student_features, top_n=3, add_noise=False, consistency_mode=True):
        """Predict track and strand recommendations for a student"""
        if self.track_model is None or self.strand_model is None:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        # Ensure student_features is 2D
        if len(student_features.shape) == 1:
            student_features = student_features.reshape(1, -1)
        
        # Scale features
        student_features_scaled = self.feature_scaler.transform(student_features)
        
        # Only add noise if explicitly requested and not in consistency mode
        if add_noise and not consistency_mode:
            noise_scale = 0.01  # Reduced noise for better consistency
            noise = np.random.normal(0, noise_scale, student_features_scaled.shape)
            student_features_scaled = student_features_scaled + noise
        
        # Get track predictions with probabilities
        track_probs = self.track_model.predict_proba(student_features_scaled)[0]
        track_classes = self.track_encoder.classes_
        
        # Get strand predictions with probabilities
        strand_probs = self.strand_model.predict_proba(student_features_scaled)[0]
        strand_classes = self.strand_encoder.classes_
        
        # Only add probability noise if explicitly requested and not in consistency mode
        if add_noise and not consistency_mode:
            prob_noise_scale = 0.005  # Much smaller noise
            track_noise = np.random.normal(0, prob_noise_scale, len(track_probs))
            strand_noise = np.random.normal(0, prob_noise_scale, len(strand_probs))
            
            # Apply noise and ensure probabilities remain valid
            track_probs = np.maximum(0, track_probs + track_noise)
            strand_probs = np.maximum(0, strand_probs + strand_noise)
            
            # Renormalize to ensure they sum to 1
            track_probs = track_probs / np.sum(track_probs)
            strand_probs = strand_probs / np.sum(strand_probs)
        
        # Create strand-to-track mapping for reverse lookup
        strand_to_track = {}
        for track, strands in self.track_strand_mapping.items():
            for strand in strands:
                strand_to_track[strand] = track
        
        # Get top strand recommendations directly (this ensures we get exactly top_n recommendations)
        strand_indices = np.argsort(strand_probs)[::-1][:top_n]
        
        recommendations = []
        
        for i, strand_idx in enumerate(strand_indices):
            strand_name = strand_classes[strand_idx]
            strand_confidence = strand_probs[strand_idx]
            
            # Find the track for this strand
            track_name = strand_to_track.get(strand_name, 'Unknown Track')
            
            # Get track confidence
            if track_name in track_classes:
                track_idx = np.where(track_classes == track_name)[0][0]
                track_confidence = track_probs[track_idx]
            else:
                track_confidence = 0.5  # Default confidence for unknown tracks
            
            # Calculate overall score (weighted combination)
            overall_score = float(strand_confidence * 0.7 + track_confidence * 0.3)
            
            # Generate reasoning based on student features
            reasoning = self._generate_reasoning(student_features[0], strand_name, strand_confidence)
            
            recommendations.append({
                'track': track_name,
                'track_confidence': float(track_confidence),
                'recommended_strands': [{
                    'strand': strand_name,
                    'confidence': float(strand_confidence)
                }],
                'overall_score': overall_score,
                'rank': i + 1,
                'reasoning': reasoning
            })
        
        # Sort by overall score (should already be sorted, but just to be sure)
        recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return recommendations
    
    def _generate_reasoning(self, student_features, strand_name, confidence):
        """Generate human-readable reasoning for a recommendation"""
        reasoning = []
        
        # Analyze assessment scores
        strong_skills = []
        assessment_features = [f for f in self.feature_names if f.startswith('assessment_')]
        
        for feature in assessment_features:
            idx = self.feature_names.index(feature)
            score = student_features[idx]
            if score >= 75:
                skill_name = feature.replace('assessment_', '').replace('_', ' ').title()
                strong_skills.append(skill_name)
        
        if strong_skills:
            if len(strong_skills) > 3:
                skills_text = ', '.join(strong_skills[:3])
            else:
                skills_text = ', '.join(strong_skills)
            reasoning.append(f"Strong in {skills_text}")
        
        # Analyze interests
        active_interests = []
        interest_features = [f for f in self.feature_names if f.startswith('interest_')]
        
        for feature in interest_features:
            idx = self.feature_names.index(feature)
            if student_features[idx] == 1:
                interest_name = feature.replace('interest_', '').replace('_', ' ').title()
                active_interests.append(interest_name)
        
        if active_interests:
            if len(active_interests) > 2:
                interests_text = ', '.join(active_interests[:2])
            else:
                interests_text = ', '.join(active_interests)
            reasoning.append(f"Interested in {interests_text}")
        
        # Analyze hobbies
        active_hobbies = []
        hobby_features = [f for f in self.feature_names if f.startswith('hobby_')]
        
        for feature in hobby_features:
            idx = self.feature_names.index(feature)
            if student_features[idx] == 1:
                hobby_name = feature.replace('hobby_', '').replace('_', ' ').title()
                active_hobbies.append(hobby_name)
        
        if active_hobbies:
            if len(active_hobbies) > 2:
                hobbies_text = ', '.join(active_hobbies[:2])
            else:
                hobbies_text = ', '.join(active_hobbies)
            reasoning.append(f"Enjoys {hobbies_text}")
        
        # Add strand-specific reasoning
        strand_lower = strand_name.lower()
        if 'stem' in strand_lower:
            reasoning.append("Analytical and scientific abilities")
        elif 'abm' in strand_lower:
            reasoning.append("Business acumen and leadership potential")
        elif 'humss' in strand_lower:
            reasoning.append("Communication and social awareness")
        elif 'gas' in strand_lower:
            reasoning.append("Diverse skill set and adaptability")
        elif 'ict' in strand_lower or 'information' in strand_lower:
            reasoning.append("Technical aptitude and problem-solving")
        elif 'industrial' in strand_lower:
            reasoning.append("Practical skills and hands-on abilities")
        elif 'home economics' in strand_lower:
            reasoning.append("Creativity and attention to detail")
        elif 'agri' in strand_lower:
            reasoning.append("Environmental awareness and patience")
        
        # If no reasoning was generated, add a default
        if not reasoning:
            reasoning.append("Profile matches strand requirements")
        
        return reasoning
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        if self.track_model is None or self.strand_model is None:
            raise ValueError("Models not trained yet.")
        
        track_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.track_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        strand_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.strand_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'track_importance': track_importance,
            'strand_importance': strand_importance
        }
    
    def save_models(self, model_dir='models'):
        """Save trained models and preprocessors"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.track_model, os.path.join(model_dir, 'track_model.pkl'))
        joblib.dump(self.strand_model, os.path.join(model_dir, 'strand_model.pkl'))
        
        # Save preprocessors
        joblib.dump(self.feature_scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        joblib.dump(self.track_encoder, os.path.join(model_dir, 'track_encoder.pkl'))
        joblib.dump(self.strand_encoder, os.path.join(model_dir, 'strand_encoder.pkl'))
        
        # Save feature names
        joblib.dump(self.feature_names, os.path.join(model_dir, 'feature_names.pkl'))
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir='models'):
        """Load trained models and preprocessors"""
        self.track_model = joblib.load(os.path.join(model_dir, 'track_model.pkl'))
        self.strand_model = joblib.load(os.path.join(model_dir, 'strand_model.pkl'))
        self.feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
        self.track_encoder = joblib.load(os.path.join(model_dir, 'track_encoder.pkl'))
        self.strand_encoder = joblib.load(os.path.join(model_dir, 'strand_encoder.pkl'))
        self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        
        # Update track-strand mapping based on actual trained model
        self._update_track_strand_mapping()
        
        print(f"Models loaded from {model_dir}")
    
    def _update_track_strand_mapping(self):
        """Update track-strand mapping based on actual trained model classes"""
        track_classes = self.track_encoder.classes_
        strand_classes = self.strand_encoder.classes_
        
        # Create mapping based on actual model classes
        self.track_strand_mapping = {}
        
        # Map Academic track strands
        academic_strands = []
        for strand in strand_classes:
            if any(keyword in strand for keyword in ['STEM', 'ABM', 'HUMSS', 'GAS']):
                academic_strands.append(strand)
        
        # Map Technical-Vocational track strands  
        techvoc_strands = []
        for strand in strand_classes:
            if any(keyword in strand for keyword in ['ICT', 'Industrial Arts', 'Home Economics', 'Agri-Fishery']):
                techvoc_strands.append(strand)
        
        # Use the actual track names from the model
        for track in track_classes:
            if 'Academic' in track:
                self.track_strand_mapping[track] = academic_strands
            elif 'Technical' in track or 'Vocational' in track:
                self.track_strand_mapping[track] = techvoc_strands

def main():
    """Main function to demonstrate the system"""
    
    print("=== Strand and Track Recommendation System ===")
    
    # Use the student data file
    data_file = 'data/student_data.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure student_data.csv exists in the data directory.")
        return
    
    # Initialize and train the system
    recommender = StrandRecommendationSystem(data_file)
    
    # Load and preprocess data
    X, y_track, y_strand = recommender.load_data()
    
    # Train models
    results = recommender.train_models(X, y_track, y_strand)
    
    # Save models
    recommender.save_models()
    
    # Display feature importance
    importance = recommender.get_feature_importance()
    print("\nTop 10 Most Important Features for Track Recommendation:")
    print(importance['track_importance'].head(10))
    
    print("\nTop 10 Most Important Features for Strand Recommendation:")
    print(importance['strand_importance'].head(10))
    
    # Example prediction
    print("\n=== Example Recommendation ===")
    # Create a sample student profile (you would get this from your application)
    sample_features = np.zeros(len(recommender.feature_names))
    
    # Set some example values (this would come from student assessment/survey)
    # High math and science scores (STEM-oriented)
    if 'assessment_mathematics' in recommender.feature_names:
        sample_features[recommender.feature_names.index('assessment_mathematics')] = 85
    if 'assessment_science' in recommender.feature_names:
        sample_features[recommender.feature_names.index('assessment_science')] = 88
    if 'assessment_technology' in recommender.feature_names:
        sample_features[recommender.feature_names.index('assessment_technology')] = 82
    
    # Set demographic info
    sample_features[recommender.feature_names.index('age')] = 16
    sample_features[recommender.feature_names.index('gpa')] = 3.5
    sample_features[recommender.feature_names.index('gender_male')] = 1
    
    # Get recommendations
    recommendations = recommender.predict_recommendations(sample_features)
    
    print("Recommendations for sample student:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Track: {rec['track']} (Confidence: {rec['track_confidence']:.3f})")
        print(f"   Overall Score: {rec['overall_score']:.3f}")
        print("   Recommended Strands:")
        for strand in rec['recommended_strands']:
            print(f"   - {strand['strand']} (Confidence: {strand['confidence']:.3f})")

if __name__ == "__main__":
    main()