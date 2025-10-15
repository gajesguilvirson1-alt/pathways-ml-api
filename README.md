# Machine Learning Strand Recommendation System

A comprehensive machine learning system using Python and scikit-learn to recommend educational strands and tracks for students based on their assessment scores, interests, hobbies, and work preferences.

## Features

- **Synthetic Data Generation**: Creates realistic student profiles for training
- **Multi-Model Approach**: Separate models for track and strand recommendations
- **REST API**: Flask-based API for easy integration
- **Laravel Integration**: PHP service class for seamless Laravel integration
- **Feature Importance Analysis**: Understand which factors influence recommendations
- **Caching**: Built-in caching for improved performance
- **Fallback System**: Graceful degradation when ML API is unavailable

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Laravel App   │───▶│   ML API (Flask) │───▶│  ML Models      │
│                 │    │                  │    │  (scikit-learn) │
│ - Controllers   │    │ - Recommendations│    │ - Track Model   │
│ - Services      │    │ - Feature Import.│    │ - Strand Model  │
│ - Models        │    │ - Health Check   │    │ - Preprocessors │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Tracks and Strands Supported

### Academic Track
- **STEM**: Science, Technology, Engineering, Mathematics
- **ABM**: Accountancy, Business, Management
- **HUMSS**: Humanities and Social Sciences
- **GAS**: General Academic Strand

### Technical-Vocational Track
- **ICT**: Information and Communications Technology
- **Industrial Arts**: Manufacturing and construction skills
- **Home Economics**: Culinary arts, fashion, interior design
- **Agri-Fishery Arts**: Agriculture and fishery management

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Laravel application (for integration)

### Quick Setup

1. **Navigate to the ML system directory**:
   ```bash
   cd c:\xampp\htdocs\pathways\ml_system
   ```

2. **Run the automated setup**:
   ```bash
   python setup.py
   ```

   This will:
   - Install required Python packages
   - Generate synthetic training data
   - Train ML models
   - Start the Flask API server

### Manual Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate synthetic data**:
   ```bash
   python setup.py --action generate-data --samples 2000
   ```

3. **Train models**:
   ```bash
   python setup.py --action train
   ```

4. **Start the API**:
   ```bash
   python setup.py --action start-api
   ```

## API Endpoints

### Health Check
```http
GET /health
```
Returns API status and system readiness.

### Get Recommendations
```http
POST /recommend
Content-Type: application/json

{
  "student_id": "STU_12345",
  "assessment_scores": {
    "mathematics": 85,
    "science": 88,
    "communication": 75,
    "critical_thinking": 80,
    "problem_solving": 87
  },
  "interests": ["technology", "research", "innovation"],
  "hobbies": ["coding", "gaming", "science_experiments"],
  "work_preferences": ["analytical", "independent", "challenging"],
  "demographics": {
    "age": 16,
    "gender": "male",
    "gpa": 3.5,
    "family_income": "middle"
  },
  "top_n": 3
}
```

### Feature Importance
```http
GET /feature-importance
```
Returns the most important features for track and strand predictions.

### Retrain Models
```http
POST /retrain
Content-Type: application/json

{
  "data_file": "data/new_student_data.csv"
}
```

### Sample Request Format
```http
GET /sample-request
```
Returns a sample request format for testing.

## Laravel Integration

### 1. Update Environment Variables

Add to your `.env` file:
```env
ML_API_URL=http://localhost:5000
ML_API_TIMEOUT=30
ML_CACHE_MINUTES=60
```

### 2. Add Routes

Add to your `routes/web.php` or `routes/api.php`:
```php
use App\Http\Controllers\MLRecommendationController;

Route::middleware(['auth'])->group(function () {
    Route::get('/ml/recommendations', [MLRecommendationController::class, 'getRecommendations']);
    Route::get('/ml/status', [MLRecommendationController::class, 'getStatus']);
    Route::post('/ml/clear-cache', [MLRecommendationController::class, 'clearCache']);
    Route::get('/ml/compare', [MLRecommendationController::class, 'compareRecommendations']);
});

// Admin routes
Route::middleware(['auth', 'admin'])->group(function () {
    Route::post('/ml/retrain', [MLRecommendationController::class, 'retrainModels']);
    Route::get('/ml/feature-importance', [MLRecommendationController::class, 'getFeatureImportance']);
});
```

### 3. Use in Your Controllers

```php
use App\Services\MLRecommendationService;

class RecommendationController extends Controller
{
    public function getRecommendations(MLRecommendationService $mlService)
    {
        $user = Auth::user();
        $recommendations = $mlService->getRecommendations($user, 5);
        
        return view('recommendations.index', compact('recommendations'));
    }
}
```

## Data Format

### Student Assessment Scores
The system expects assessment scores for these skill areas:
- mathematics, science, communication, critical_thinking
- problem_solving, creativity, leadership, technical_knowledge
- research, analysis, writing, practical_skills
- analytical_thinking, business_acumen, empathy, adaptability
- flexibility, programming, logical_thinking, technical_skills
- hands_on, craftsmanship, mechanical_aptitude, attention_to_detail
- organization, care_giving, environmental_awareness, patience
- physical_stamina, observation

### Interests Categories
- technology, business, science, arts, social_work
- healthcare, education, engineering, finance, research
- management, creative_arts, sports, environment

### Hobbies Categories
- reading, sports, music, art, technology, science
- cooking, gardening, writing, photography, gaming
- volunteering, traveling, crafting

### Work Preferences
- independent, collaborative, leadership, analytical
- creative, hands_on, helping_others, challenging
- stable, flexible, goal_oriented, innovative

## Model Performance

The system uses Random Forest classifiers with the following typical performance:
- **Track Classification Accuracy**: ~85-90%
- **Strand Classification Accuracy**: ~80-85%

### Feature Importance
Top factors influencing recommendations:
1. Assessment scores in relevant subjects
2. Interest alignment with strand focus areas
3. Work preference compatibility
4. Hobby relevance to strand activities
5. Academic performance (GPA)

## Customization

### Adding New Strands
1. Update `data_generator.py` with new strand definitions
2. Regenerate training data
3. Retrain models
4. Update Laravel integration mappings

### Adjusting Feature Weights
Modify the feature importance by updating the training data generation logic in `data_generator.py`.

### Custom Scoring
Implement custom scoring logic in `strand_recommender.py` by modifying the `predict_recommendations` method.

## Monitoring and Maintenance

### Health Monitoring
```bash
curl http://localhost:5000/health
```

### Performance Monitoring
- Monitor API response times
- Track recommendation accuracy
- Monitor cache hit rates

### Model Retraining
Retrain models periodically with new student data:
```bash
curl -X POST http://localhost:5000/retrain \
  -H "Content-Type: application/json" \
  -d '{"data_file": "data/updated_student_data.csv"}'
```

## Troubleshooting

### Common Issues

1. **API Not Starting**
   - Check Python version (3.8+ required)
   - Verify all dependencies are installed
   - Check port 5000 is available

2. **Low Recommendation Accuracy**
   - Increase training data size
   - Review feature mappings
   - Check data quality

3. **Laravel Integration Issues**
   - Verify ML_API_URL in .env
   - Check API connectivity
   - Review Laravel logs

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Increase Cache Duration**:
   ```env
   ML_CACHE_MINUTES=120
   ```

2. **Optimize Model Parameters**:
   Adjust hyperparameters in `strand_recommender.py`

3. **Use Model Compression**:
   Consider model compression for faster inference

## Development

### Running Tests
```bash
python setup.py --action test
```

### Adding Features
1. Update data generator for new features
2. Modify model training pipeline
3. Update API endpoints
4. Update Laravel integration

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is part of the Pathways educational system and follows the same licensing terms.

## Support

For technical support or questions:
1. Check the troubleshooting section
2. Review Laravel and Python logs
3. Test API endpoints directly
4. Verify data format and model training

## Version History

- **v1.0**: Initial release with basic ML recommendations
- **v1.1**: Added feature importance analysis
- **v1.2**: Improved Laravel integration
- **v1.3**: Added caching and fallback systems