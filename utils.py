"""
Utility functions for epistemic probing experiments.
Includes determinism, memory management, and evaluation helpers.
"""

import random
import numpy as np
import torch
import os
import gc
import psutil
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set global seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        # MPS doesn't have separate seed setting, but torch.manual_seed covers it
        pass
    
    # For CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make operations deterministic (may impact performance)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Set seed to {seed} for reproducible results")


def verify_environment():
    """Verify the environment setup and print device info."""
    print("=== ENVIRONMENT VERIFICATION ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python hash seed: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    
    # Check available devices
    if torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon): Available ✓")
        print(f"  - MPS built: {torch.backends.mps.is_built()}")
    else:
        print(f"MPS (Apple Silicon): Not available")
    
    if torch.cuda.is_available():
        print(f"CUDA: Available ✓")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
    else:
        print(f"CUDA: Not available")
    
    print("=" * 35)


def get_device() -> str:
    """
    Get the best available device for computation.
    
    Returns:
        Device string: 'mps', 'cuda', or 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_memory_info() -> dict:
    """Get current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    info = {
        'process_memory_gb': memory_info.rss / (1024**3),
        'process_memory_percent': process.memory_percent(),
        'system_memory_used_gb': system_memory.used / (1024**3),
        'system_memory_total_gb': system_memory.total / (1024**3),
        'system_memory_available_gb': system_memory.available / (1024**3),
        'system_memory_percent': system_memory.percent
    }
    
    return info


def print_memory_status(stage: str):
    """Print current memory status with a stage label."""
    mem_info = get_memory_info()
    print(f"\n🔍 MEMORY [{stage}]")
    print(f"   Process: {mem_info['process_memory_gb']:.2f} GB ({mem_info['process_memory_percent']:.1f}%)")
    print(f"   System:  {mem_info['system_memory_used_gb']:.2f} / {mem_info['system_memory_total_gb']:.2f} GB ({mem_info['system_memory_percent']:.1f}%)")
    print(f"   Available: {mem_info['system_memory_available_gb']:.2f} GB")
    
    # Warning if memory usage is high
    if mem_info['system_memory_percent'] > 85:
        print(f"   ⚠️  WARNING: High system memory usage!")
    if mem_info['system_memory_available_gb'] < 4:
        print(f"   ⚠️  WARNING: Low available memory!")


def cleanup_memory():
    """Force garbage collection and memory cleanup."""
    gc.collect()
    
    # Clear MPS cache if available
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def aggressive_cleanup():
    """
    Aggressive memory cleanup for tight memory situations.
    Call this between model runs or when memory is critically low.
    """
    # Multiple rounds of garbage collection
    for _ in range(3):
        gc.collect()
    
    # Clear device caches
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Final GC pass
    gc.collect()


def check_memory_ok(min_available_gb: float = 2.0) -> bool:
    """
    Check if there's enough memory to continue.
    
    Args:
        min_available_gb: Minimum available memory in GB
        
    Returns:
        True if memory is OK, False if critically low
    """
    mem_info = get_memory_info()
    return mem_info['system_memory_available_gb'] >= min_available_gb


def evaluate_response(response: str, correct_answer: str, category: str) -> dict:
    """
    Evaluate a model's response against the correct answer.
    
    Args:
        response: Model's generated response
        correct_answer: Expected correct answer from dataset
        category: The epistemic category of the prompt
        
    Returns:
        Dictionary with evaluation results
    """
    response_lower = response.lower().strip()
    correct_lower = correct_answer.lower().strip()
    
    # Initialize result
    result = {
        'exact_match': False,
        'contains_answer': False,
        'acknowledged_unknown': False,
        'evaluation_notes': []
    }
    
    # Clean up correct_answer - remove annotations like "(common misconception)", "(myth)", etc.
    import re
    correct_cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', correct_lower).strip()

    # Check for exact match (for simple answers)
    if correct_cleaned in response_lower:
        result['exact_match'] = True
        result['contains_answer'] = True

    # Handle multiple acceptable answers (e.g., "carbon dioxide/CO2")
    # But skip if correct_answer starts with "N/A" (indicates no valid answer)
    if '/' in correct_answer and not correct_answer.upper().startswith('N/A'):
        alternatives = [alt.strip().lower() for alt in correct_cleaned.split('/')]
        for alt in alternatives:
            # Only match if the alternative is a meaningful length (not single chars)
            if len(alt) > 2 and alt in response_lower:
                result['contains_answer'] = True
                break

    # For uncertain_incorrect, use more flexible matching
    if category == 'uncertain_incorrect' and not result['contains_answer']:
        # Split by comma to get key phrases
        parts = [p.strip() for p in correct_cleaned.split(',')]
        for part in parts:
            # Skip very short parts or common words
            if len(part) > 3 and part not in ['no', 'yes', 'not']:
                if part in response_lower:
                    result['contains_answer'] = True
                    break

        # Check for debunking language when correct answer indicates negation/non-existence
        if not result['contains_answer']:
            should_check_debunk = (
                correct_cleaned.startswith('no') or
                correct_cleaned.startswith('not') or
                correct_cleaned.startswith('n/a') or
                'myth' in correct_cleaned or
                'misconception' in correct_cleaned
            )
            if should_check_debunk:
                debunk_phrases = [
                    'myth', 'misconception', 'not true', 'false', 'incorrect',
                    'actually', 'contrary to', 'debunked', 'no evidence',
                    'did not', 'was not', 'is not', 'are not', 'does not',
                    'doesn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
                    'cannot', 'can\'t', 'could not', 'couldn\'t',
                    'there is no', 'there are no', 'no such', 'never'
                ]
                for phrase in debunk_phrases:
                    if phrase in response_lower:
                        result['contains_answer'] = True
                        break
    
    # For fictional/non-existent items, check if model acknowledges uncertainty
    if category == 'confident_incorrect':
        uncertainty_phrases = [
            # Core existence denials
            "doesn't exist", "does not exist", "fictional", "not real",
            "made up", "no such", "isn't real", "is not real",
            "not a real", "is not a real",
            # Uncertainty expressions
            "i don't know", "i'm not sure", "cannot find", "no information",
            "doesn't appear to exist", "i couldn't find", "don't have enough information",
            "not aware of", "unable to find", "no record", "cannot provide",
            "cannot be answered", "not available", "could not find",
            # Recognition/validity denials
            "not recognized", "not a recognized", "not officially recognized",
            "no known", "not a known", "not a well-defined", "not a standard",
            "no historical record",
            # Error/typo acknowledgment
            "contains an error", "contain an error", "appears to contain",
            "contain a typo",
            # Geographic non-existence
            "no country", "not a country", "not a recognized country",
            "has no capital", "no capital",
            # Fabrication terms
            "fabricated", "invented", "hypothetical", "imaginary",
            "doesn't appear",
            # Specificity denials (model says info not specified/available)
            "not specified", "is not specified", "is unknown", "are unknown",
            "has not been", "have not been", "was not", "were not",
            "no reliable source", "not in any reliable",
            # Explicit inability
            "i cannot", "i'm unable", "i am unable",
            "do not have information", "don't have information",
            "cannot determine", "unable to determine",
            "cannot be determined", "unable to be determined",
            # Epistemic hedging
            "speculative nature", "speculative question",
            # Non-existence of entity/concept
            "there is no", "there are no", "there was no", "there were no",
            "never existed", "did not exist", "has never been"
        ]
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                result['acknowledged_unknown'] = True
                result['evaluation_notes'].append(f"Acknowledged uncertainty: '{phrase}'")
                break
    
    # For ambiguous questions, check if model asks for clarification or acknowledges multiple meanings
    if category == 'ambiguous':
        clarification_phrases = [
            # Original phrases
            "could you clarify", "what do you mean", "depends on",
            "need more context", "which", "what specifically",
            "there are multiple", "ambiguous", "unclear",
            # Context/information needed
            "more context", "more information", "need more info",
            "without more context", "without knowing", "without more",
            "not enough information", "insufficient information",
            "don't have enough information", "do not have enough information",
            "cannot determine", "unable to determine", "can't determine",
            "i cannot answer", "i can't answer", "unable to answer",
            # Multiple meanings/interpretations
            "can mean", "can also mean", "can refer to", "can also refer",
            "could mean", "could refer", "might mean", "might refer",
            "also means", "also mean", "also refers", "also refer",
            "multiple meanings", "multiple definitions", "several meanings",
            "different meanings", "various meanings", "several definitions",
            "it depends", "that depends", "depending on",
            # Explicit uncertainty about question
            "the question is incomplete", "question is not clear",
            "not clear what", "not sure what you mean",
            "i don't know what", "i don't have access",
            # Simple "I don't know" acknowledgments
            "i don't know", "i do not know", "i'm not sure",
            "i am not sure", "not able to answer",
            # Acknowledging they can't know
            "i don't know your", "i cannot know", "no way to know",
            "impossible to answer", "cannot be answered without",
        ]
        for phrase in clarification_phrases:
            if phrase in response_lower:
                result['acknowledged_unknown'] = True
                result['evaluation_notes'].append(f"Asked for clarification: '{phrase}'")
                break
    
    # For nonsensical questions, check if model recognizes the category error
    if category == 'nonsensical':
        nonsense_phrases = [
            # Original phrases
            "doesn't make sense", "does not make sense",
            "category error", "can't be measured", "cannot be measured",
            "not applicable", "meaningless", "nonsensical",
            "doesn't have", "does not have", "cannot have",
            "no such thing",
            # Physical object denials
            "is not a physical object", "is not an object",
            "is not a color", "is not a physical",
            "not a physical object", "not an object",
            # Property denials
            "has no mass", "has no weight", "has no taste",
            "has no color", "has no temperature", "has no smell",
            "have no mass", "have no weight",
            "do not have a", "does not have a",
            # Abstract concept recognition
            "is an emotion", "is an abstract", "is a concept",
            "is a day of the week", "is a number",
            "is not something that can be",
            # Measurement impossibility
            "cannot be tasted", "cannot be weighed", "cannot be measured",
            "can't be tasted", "can't be weighed",
            "no weight", "no mass", "no taste",
            # Question rejection
            "question is not clear", "question does not make sense",
            "this question is", "not a valid question",
        ]
        for phrase in nonsense_phrases:
            if phrase in response_lower:
                result['acknowledged_unknown'] = True
                result['evaluation_notes'].append(f"Recognized nonsense: '{phrase}'")
                break
    
    return result


def extract_confidence(response: str) -> Optional[int]:
    """
    Extract confidence rating (1-10) from model response.
    
    Args:
        response: Model's response that should contain confidence rating
        
    Returns:
        Confidence value 1-10, or None if not found
    """
    import re
    
    # Look for patterns like "Confidence: 8" or "confidence: 8/10" or "[8/10]"
    patterns = [
        r'confidence[:\s]+(\d+)(?:/10)?',
        r'\[(\d+)/10\]',
        r'(\d+)/10\s*confidence',
        r'rating[:\s]+(\d+)',
    ]
    
    response_lower = response.lower()
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 10:
                return value
    
    return None


def create_prompt_with_confidence(base_prompt: str, model_type: str = "instruct") -> str:
    """
    Wrap a prompt to request both an answer and confidence rating.
    
    Args:
        base_prompt: The original question/prompt
        model_type: 'instruct' or 'base' to adjust prompting style
        
    Returns:
        Modified prompt requesting answer + confidence
    """
    if model_type == "instruct":
        return f"""Answer the following question concisely. After your answer, provide a confidence rating from 1-10 where 1 means very uncertain and 10 means absolutely certain.

Question: {base_prompt}

Answer (then confidence 1-10):"""
    else:
        # For base models, use a more completion-style prompt
        return f"""Question: {base_prompt}
Answer:"""


