"""
Main Flask API application.
Provides REST endpoints for scam detection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.scam_classifier import ScamClassifier
from src.models.risk_scoring_model import RiskScoringModel
from src.models.anomaly_detection import AnomalyDetector
from src.decision_engine.alert_decision import AlertDecisionEngine
from src.nlp_engine.sentiment_analysis import SentimentAnalyzer
from src.nlp_engine.intent_detection import IntentDetector
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

# Initialize
app = Flask(__name__)
CORS(app)
logger = get_logger(__name__)
config = get_config()

# Load models (global)
classifier = None
risk_model = None
alert_engine = None
anomaly_detector = None


def load_models():
    """Load all trained models."""
    global classifier, risk_model, alert_engine, anomaly_detector
    
    try:
        model_path = Path(config.MODEL_PATH) / 'scam_classifier.pkl'
        if model_path.exists():
            classifier = ScamClassifier.load(str(model_path))
            logger.info("Scam classifier loaded successfully")
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Please train the model first using: python src/training/train_model.py")
            return False
        
        # Initialize risk model
        sentiment_analyzer = SentimentAnalyzer()
        intent_detector = IntentDetector()
        risk_model = RiskScoringModel(
            classifier=classifier,
            sentiment_analyzer=sentiment_analyzer,
            intent_detector=intent_detector
        )
        
        # Initialize alert engine
        alert_engine = AlertDecisionEngine()
        
        # Load anomaly detector if available
        if config.ENABLE_ANOMALY_DETECTION:
            anomaly_path = Path(config.MODEL_PATH) / 'anomaly_detector.pkl'
            if anomaly_path.exists():
                anomaly_detector = AnomalyDetector.load(str(anomaly_path))
                logger.info("Anomaly detector loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': classifier is not None,
        'version': '1.0.0'
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_message():
    """
    Analyze a message for scam detection.
    
    Request body:
        {
            "message": "text to analyze",
            "message_id": "optional ID"
        }
    
    Returns:
        {
            "message_id": "...",
            "prediction": "scam/legitimate",
            "risk_score": 0.85,
            "risk_level": "high",
            "alert": {...},
            "details": {...}
        }
    """
    if not classifier:
        return jsonify({'error': 'Models not loaded. Please train the model first.'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        message_id = data.get('message_id', 'unknown')
        
        # Calculate risk score
        risk_analysis = risk_model.calculate_risk_score(message)
        
        # Generate alert if needed
        alert = alert_engine.generate_alert(message, risk_analysis)
        
        # Anomaly detection
        anomaly_result = None
        if anomaly_detector:
            anomaly_result = anomaly_detector.analyze(message)
        
        result = {
            'message_id': message_id,
            'prediction': risk_analysis['ml_prediction'],
            'risk_score': risk_analysis['risk_score'],
            'risk_level': risk_analysis['risk_level'],
            'confidence': risk_analysis['confidence'],
            'alert': alert,
            'risk_factors': risk_analysis['risk_factors'],
            'details': {
                'ml_score': risk_analysis['ml_score'],
                'intent_score': risk_analysis['intent_score'],
                'sentiment_score': risk_analysis['sentiment_score'],
                'pattern_score': risk_analysis['pattern_score'],
                'primary_intent': risk_analysis['intent_analysis']['primary_intent'],
                'anomaly_detection': anomaly_result
            }
        }
        
        logger.info(f"Analyzed message {message_id}: {risk_analysis['risk_level']} risk")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing message: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple messages.
    
    Request body:
        {
            "messages": ["text1", "text2", ...]
        }
    """
    if not classifier:
        return jsonify({'error': 'Models not loaded'}), 503
    
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        results = []
        for i, message in enumerate(messages):
            risk_analysis = risk_model.calculate_risk_score(message)
            alert = alert_engine.generate_alert(message, risk_analysis)
            
            results.append({
                'message_id': i,
                'message': message[:100] + '...' if len(message) > 100 else message,
                'risk_score': risk_analysis['risk_score'],
                'risk_level': risk_analysis['risk_level'],
                'prediction': risk_analysis['ml_prediction'],
                'has_alert': alert is not None
            })
        
        # Summary statistics
        summary = risk_model.get_summary_statistics(
            [risk_model.calculate_risk_score(m) for m in messages]
        )
        
        return jsonify({
            'results': results,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on a prediction.
    
    Request body:
        {
            "message_id": "...",
            "prediction": "scam/legitimate",
            "actual": "scam/legitimate",
            "feedback": "optional comment"
        }
    """
    try:
        data = request.get_json()
        
        # In a production system, store this in a database
        logger.info(f"Feedback received: {data}")
        
        return jsonify({
            'status': 'success',
            'message': 'Thank you for your feedback'
        })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get system statistics."""
    try:
        stats = {
            'models_loaded': classifier is not None,
            'anomaly_detection_enabled': anomaly_detector is not None,
            'scam_threshold': config.SCAM_THRESHOLD,
            'high_risk_threshold': config.HIGH_RISK_THRESHOLD
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Load models
    if not load_models():
        logger.warning("Running API without trained models. Some endpoints will not work.")
    
    # Run app
    logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )