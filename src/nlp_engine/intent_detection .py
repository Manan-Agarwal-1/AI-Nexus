"""
Intent detection module.
Detects the intent behind messages to identify potential scams.
"""

import re
from typing import Dict, List, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IntentDetector:
    """Detect intent and purpose of messages."""
    
    def __init__(self):
        """Initialize intent detector with patterns."""
        
        # Define intent patterns
        self.intent_patterns = {
            'financial_request': [
                r'\b(?:send|transfer|pay|wire)\s+(?:money|cash|funds)',
                r'\b(?:bank|credit card|account)\s+(?:details|information|number)',
                r'\bpayment\s+(?:required|needed|due)',
                r'\b(?:deposit|fee|charge)\s+\$?\d+',
            ],
            'urgency_pressure': [
                r'\b(?:urgent|immediately|asap|now|hurry|quick)',
                r'\b(?:limited|expires|deadline|final|last chance)',
                r'\b(?:act now|don\'t wait|time sensitive)',
                r'\bwithin\s+\d+\s+(?:hours|minutes|days)',
            ],
            'prize_reward': [
                r'\b(?:won|winner|congratulations|selected)',
                r'\b(?:prize|reward|gift|bonus|cash)',
                r'\b(?:claim|collect|redeem)',
                r'\bfree\s+(?:iphone|money|cash|trip|cruise)',
            ],
            'account_security': [
                r'\b(?:account|password)\s+(?:suspended|locked|compromised)',
                r'\b(?:verify|confirm|update)\s+(?:identity|information|account)',
                r'\b(?:security|suspicious)\s+(?:alert|activity|breach)',
                r'\bunauthorized\s+(?:access|transaction|activity)',
            ],
            'authority_impersonation': [
                r'\b(?:irs|social security|government|federal)',
                r'\b(?:bank|paypal|amazon|apple|microsoft)\s+(?:alert|security)',
                r'\b(?:law enforcement|police|fbi|customs)',
                r'\b(?:tax|debt|fine|penalty)\s+(?:owed|due|payment)',
            ],
            'personal_info_request': [
                r'\b(?:ssn|social security number)',
                r'\b(?:date of birth|dob|birthday)',
                r'\b(?:provide|share|send)\s+(?:your|personal)\s+(?:information|details)',
                r'\b(?:password|pin|security code)',
            ],
            'link_click': [
                r'click\s+(?:here|link|below)',
                r'visit\s+(?:our|this)\s+(?:website|link|page)',
                r'go to\s+http',
                r'\b(?:http|https|www)\.',
            ],
            'threatening': [
                r'\b(?:arrest|jail|lawsuit|legal action)',
                r'\byour\s+(?:account|service)\s+will\s+be\s+(?:closed|suspended|terminated)',
                r'\b(?:consequences|penalties|charges)',
                r'\b(?:lose|miss out|forfeit)',
            ],
        }
        
        # Compile patterns
        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) 
                    for pattern in patterns]
            for intent, patterns in self.intent_patterns.items()
        }
    
    def detect_intents(self, text: str) -> Dict[str, int]:
        """
        Detect all intents in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping intent to match count
        """
        intent_scores = {}
        
        for intent, patterns in self.compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
            intent_scores[intent] = matches
        
        return intent_scores
    
    def get_primary_intent(self, text: str) -> Tuple[str, int]:
        """
        Get the primary intent of the text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (primary_intent, score)
        """
        intent_scores = self.detect_intents(text)
        
        if not intent_scores or all(score == 0 for score in intent_scores.values()):
            return 'legitimate', 0
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        return primary_intent
    
    def analyze_intent(self, text: str) -> Dict:
        """
        Comprehensive intent analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with intent analysis results
        """
        intent_scores = self.detect_intents(text)
        primary_intent, primary_score = self.get_primary_intent(text)
        
        # Calculate total scam indicators
        scam_intents = [
            'financial_request', 'urgency_pressure', 'prize_reward',
            'account_security', 'authority_impersonation',
            'personal_info_request', 'threatening'
        ]
        
        total_scam_indicators = sum(
            intent_scores.get(intent, 0) for intent in scam_intents
        )
        
        # Detect multiple red flags
        multiple_red_flags = sum(1 for score in intent_scores.values() if score > 0) >= 2
        
        return {
            'intent_scores': intent_scores,
            'primary_intent': primary_intent,
            'primary_score': primary_score,
            'total_scam_indicators': total_scam_indicators,
            'multiple_red_flags': multiple_red_flags,
            'has_financial_request': intent_scores.get('financial_request', 0) > 0,
            'has_urgency': intent_scores.get('urgency_pressure', 0) > 0,
            'has_prize_claim': intent_scores.get('prize_reward', 0) > 0,
            'has_security_alert': intent_scores.get('account_security', 0) > 0,
            'impersonates_authority': intent_scores.get('authority_impersonation', 0) > 0,
            'requests_personal_info': intent_scores.get('personal_info_request', 0) > 0,
            'has_threat': intent_scores.get('threatening', 0) > 0,
            'has_link': intent_scores.get('link_click', 0) > 0,
        }
    
    def is_likely_scam(self, text: str, threshold: int = 2) -> bool:
        """
        Determine if text is likely a scam based on intent.
        
        Args:
            text: Input text
            threshold: Minimum number of scam indicators
            
        Returns:
            True if likely scam
        """
        analysis = self.analyze_intent(text)
        return analysis['total_scam_indicators'] >= threshold
    
    def get_risk_factors(self, text: str) -> List[str]:
        """
        Get list of identified risk factors.
        
        Args:
            text: Input text
            
        Returns:
            List of risk factor descriptions
        """
        analysis = self.analyze_intent(text)
        risk_factors = []
        
        if analysis['has_financial_request']:
            risk_factors.append("Requests financial information or payment")
        
        if analysis['has_urgency']:
            risk_factors.append("Creates sense of urgency or pressure")
        
        if analysis['has_prize_claim']:
            risk_factors.append("Offers unrealistic prizes or rewards")
        
        if analysis['has_security_alert']:
            risk_factors.append("Claims account security issue")
        
        if analysis['impersonates_authority']:
            risk_factors.append("Impersonates authority or legitimate organization")
        
        if analysis['requests_personal_info']:
            risk_factors.append("Requests sensitive personal information")
        
        if analysis['has_threat']:
            risk_factors.append("Contains threats or intimidation")
        
        if analysis['has_link']:
            risk_factors.append("Contains suspicious links")
        
        if analysis['multiple_red_flags']:
            risk_factors.append("Multiple red flags detected")
        
        return risk_factors


def detect_intent(text: str) -> Dict:
    """
    Convenience function for intent detection.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with intent analysis
    """
    detector = IntentDetector()
    return detector.analyze_intent(text)