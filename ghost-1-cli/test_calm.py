import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO)

from core.calm_decoder import CALMDecoder, DecodingConfig

def test_calm():
    print("Testing CALM Decoder Logic...")
    
    config = DecodingConfig(latent_dim=128, beta=1.0)
    decoder = CALMDecoder(config)
    
    # 1. Test Energy Score
    # Case A: Prediction matches candidates perfectly (Low Energy? Wait, formula)
    # S = Diversity - 2*Fidelity
    # If pred == candidate, Fidelity dist is 0. S = Diversity. 
    # High score = High confidence? 
    # Logic in code: score = term_diversity - term_fidelity.
    # Confidence = sigmoid(score).
    # If perfect match (fidelity=0), score = diversity (positive). High confidence.
    # If bad match (fidelity large), score = small - large (negative). Low confidence.
    
    z_pred = np.zeros((128,))
    z_candidates_good = np.random.normal(0, 0.1, (5, 128)) # Close to 0
    z_candidates_bad = np.random.normal(10, 1, (5, 128))   # Far from 0
    
    score_good = decoder.energy_score(z_pred, z_candidates_good)
    score_bad = decoder.energy_score(z_pred, z_candidates_bad)
    
    print(f"Score Good (close): {score_good:.4f}")
    print(f"Score Bad (far): {score_bad:.4f}")
    
    if score_good > score_bad:
        print("✓ Energy Score Logic Correct (Closer = Higher Score)")
    else:
        print("✗ Energy Score Logic Incorrect")
        
    # 2. Test Decode Vector Stub
    print("\nTesting decode_vector stub...")
    hidden_state = np.zeros((128,))
    tokens, conf = decoder.decode_vector(hidden_state)
    print(f"Draft Tokens: {tokens}")
    print(f"Confidence: {conf:.4f}")

if __name__ == "__main__":
    test_calm()
