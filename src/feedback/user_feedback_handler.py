"""
User feedback handler module.
Manages user feedback for continuous model improvement.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserFeedbackHandler:
    """Handle user feedback for model improvement."""
    
    def __init__(self, feedback_file: str = 'data/feedback.jsonl'):
        """
        Initialize feedback handler.
        
        Args:
            feedback_file: Path to store feedback data
        """
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    def record_feedback(self, 
                       message: str,
                       predicted_label: str,
                       actual_label: str,
                       risk_score: float,
                       user_comment: str = None,
                       message_id: str = None) -> Dict:
        """
        Record user feedback.
        
        Args:
            message: Original message
            predicted_label: Model's prediction
            actual_label: User-provided correct label
            risk_score: Model's risk score
            user_comment: Optional user comment
            message_id: Optional message ID
            
        Returns:
            Feedback record
        """
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'message_id': message_id,
            'message': message,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'risk_score': risk_score,
            'is_correct': predicted_label.lower() == actual_label.lower(),
            'user_comment': user_comment
        }
        
        # Append to file
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
        
        logger.info(f"Feedback recorded for message {message_id}")
        
        return feedback
    
    def load_feedback(self, limit: int = None) -> List[Dict]:
        """
        Load feedback records.
        
        Args:
            limit: Maximum number of records to load
            
        Returns:
            List of feedback records
        """
        if not self.feedback_file.exists():
            return []
        
        feedback_records = []
        
        with open(self.feedback_file, 'r') as f:
            for line in f:
                if limit and len(feedback_records) >= limit:
                    break
                feedback_records.append(json.loads(line))
        
        return feedback_records
    
    def get_statistics(self) -> Dict:
        """
        Get feedback statistics.
        
        Returns:
            Dictionary with statistics
        """
        feedback_records = self.load_feedback()
        
        if not feedback_records:
            return {
                'total_feedback': 0,
                'accuracy': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        
        total = len(feedback_records)
        correct = sum(1 for r in feedback_records if r['is_correct'])
        
        false_positives = sum(
            1 for r in feedback_records 
            if r['predicted_label'] == 'scam' and r['actual_label'] == 'legitimate'
        )
        
        false_negatives = sum(
            1 for r in feedback_records 
            if r['predicted_label'] == 'legitimate' and r['actual_label'] == 'scam'
        )
        
        return {
            'total_feedback': total,
            'correct_predictions': correct,
            'accuracy': correct / total if total > 0 else 0,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'error_rate': (total - correct) / total if total > 0 else 0
        }
    
    def get_misclassified_samples(self) -> List[Dict]:
        """
        Get samples that were misclassified.
        
        Returns:
            List of misclassified samples
        """
        feedback_records = self.load_feedback()
        return [r for r in feedback_records if not r['is_correct']]