"""
Alert decision engine.
Determines when and how to alert users based on risk scores.
"""

from typing import Dict, List
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)
config = get_config()


class AlertDecisionEngine:
    """Decision engine for generating alerts."""
    
    def __init__(self, 
                 scam_threshold: float = None,
                 high_risk_threshold: float = None):
        """
        Initialize alert decision engine.
        
        Args:
            scam_threshold: Minimum risk score to trigger alert
            high_risk_threshold: Threshold for high-priority alerts
        """
        self.scam_threshold = scam_threshold or config.SCAM_THRESHOLD
        self.high_risk_threshold = high_risk_threshold or config.HIGH_RISK_THRESHOLD
    
    def should_alert(self, risk_score: float) -> bool:
        """
        Determine if an alert should be generated.
        
        Args:
            risk_score: Risk score between 0 and 1
            
        Returns:
            True if alert should be generated
        """
        return risk_score >= self.scam_threshold
    
    def get_alert_priority(self, risk_score: float) -> str:
        """
        Determine alert priority level.
        
        Args:
            risk_score: Risk score between 0 and 1
            
        Returns:
            Priority: 'low', 'medium', 'high', or 'critical'
        """
        if risk_score < self.scam_threshold:
            return 'none'
        elif risk_score < 0.75:
            return 'medium'
        elif risk_score < self.high_risk_threshold:
            return 'high'
        else:
            return 'critical'
    
    def generate_alert(self, message: str, risk_analysis: Dict) -> Dict:
        """
        Generate alert with details.
        
        Args:
            message: Original message text
            risk_analysis: Risk analysis from RiskScoringModel
            
        Returns:
            Alert dictionary
        """
        risk_score = risk_analysis['risk_score']
        
        if not self.should_alert(risk_score):
            return None
        
        priority = self.get_alert_priority(risk_score)
        
        alert = {
            'alert_id': self._generate_alert_id(),
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'risk_score': risk_score,
            'risk_level': risk_analysis['risk_level'],
            'message': message,
            'title': self._generate_alert_title(risk_analysis),
            'description': self._generate_alert_description(risk_analysis),
            'risk_factors': risk_analysis['risk_factors'],
            'recommended_actions': self._get_recommended_actions(risk_analysis),
            'details': {
                'ml_prediction': risk_analysis['ml_prediction'],
                'confidence': risk_analysis['confidence'],
                'primary_intent': risk_analysis['intent_analysis']['primary_intent']
            }
        }
        
        logger.info(f"Alert generated: {alert['alert_id']} - Priority: {priority}")
        
        return alert
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return f"ALERT-{uuid.uuid4().hex[:8]}"
    
    def _generate_alert_title(self, risk_analysis: Dict) -> str:
        """Generate alert title based on risk analysis."""
        risk_level = risk_analysis['risk_level']
        primary_intent = risk_analysis['intent_analysis']['primary_intent']
        
        titles = {
            'low': "Possible Scam Detected",
            'medium': "Scam Alert: Review Message",
            'high': "High-Risk Scam Detected",
            'critical': "CRITICAL: Dangerous Scam Detected"
        }
        
        return titles.get(risk_level, "Scam Alert")
    
    def _generate_alert_description(self, risk_analysis: Dict) -> str:
        """Generate alert description."""
        risk_factors = risk_analysis['risk_factors']
        
        if not risk_factors:
            return "This message has been flagged as a potential scam based on our analysis."
        
        description = "This message has been identified as a potential scam. "
        description += f"We detected {len(risk_factors)} risk factors:\n"
        
        for factor in risk_factors[:3]:  # Show top 3 factors
            description += f"• {factor}\n"
        
        return description
    
    def _get_recommended_actions(self, risk_analysis: Dict) -> List[str]:
        """Get recommended actions based on risk analysis."""
        actions = []
        risk_level = risk_analysis['risk_level']
        
        # Base recommendations
        actions.append("Do not respond to this message")
        actions.append("Do not click any links in the message")
        
        if risk_analysis['intent_analysis'].get('has_financial_request'):
            actions.append("Do not provide any payment information")
        
        if risk_analysis['intent_analysis'].get('requests_personal_info'):
            actions.append("Do not share personal or sensitive information")
        
        if risk_level in ['high', 'critical']:
            actions.append("Delete this message immediately")
            actions.append("Report this message to authorities")
            actions.append("Block the sender")
        else:
            actions.append("Mark this message as spam")
        
        if risk_analysis['intent_analysis'].get('impersonates_authority'):
            actions.append("Contact the organization directly through official channels")
        
        return actions
    
    def batch_alerts(self, messages: List[str], risk_analyses: List[Dict]) -> List[Dict]:
        """
        Generate alerts for multiple messages.
        
        Args:
            messages: List of messages
            risk_analyses: List of risk analyses
            
        Returns:
            List of alerts (only for messages exceeding threshold)
        """
        alerts = []
        
        for message, analysis in zip(messages, risk_analyses):
            alert = self.generate_alert(message, analysis)
            if alert:
                alerts.append(alert)
        
        logger.info(f"Generated {len(alerts)} alerts from {len(messages)} messages")
        
        return alerts
    
    def get_alert_statistics(self, alerts: List[Dict]) -> Dict:
        """
        Get statistics about generated alerts.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Statistics dictionary
        """
        if not alerts:
            return {
                'total_alerts': 0,
                'by_priority': {},
                'by_risk_level': {},
                'avg_risk_score': 0
            }
        
        return {
            'total_alerts': len(alerts),
            'by_priority': {
                priority: sum(1 for a in alerts if a['priority'] == priority)
                for priority in ['medium', 'high', 'critical']
            },
            'by_risk_level': {
                level: sum(1 for a in alerts if a['risk_level'] == level)
                for level in ['low', 'medium', 'high', 'critical']
            },
            'avg_risk_score': sum(a['risk_score'] for a in alerts) / len(alerts),
            'critical_count': sum(1 for a in alerts if a['priority'] == 'critical')
        }