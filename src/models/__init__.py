"""Models package for AI Nexus.

This package avoids importing heavy ML dependencies at package import time.
Modules in `src/models/` should be imported directly (for example:
`from src.models.scam_classifier import ScamClassifier`) to keep imports lazy.
"""

# Export only the simple fallback model names so other modules can import
# them specifically via their module path when needed.
__all__ = [
    'SimpleScamClassifier',
    'SimpleRiskScoringModel',
    'SimpleAnomalyDetector'
]