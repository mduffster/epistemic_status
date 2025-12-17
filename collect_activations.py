#!/usr/bin/env python3
"""
Epistemic Probing Data Collection Script

Collects model responses and internal activations for epistemic state analysis.
Runs prompts through base/instruct models and saves:
- Model responses
- Confidence ratings (for instruct models)
- Hidden state activations (all layers)
- Attention patterns (all layers, all heads)
- Logits and entropy for the predicted next token
- Token positions: first, middle, last

Output structure:
    activations/
    ├── llama_base/
    │   ├── batch_00.npz      # Activations shape: (batch_size, n_layers, hidden_dim)
    │   ├── batch_01.npz
    │   └── metadata.json     # prompts, labels, logits, entropy, responses
    ├── llama_instruct/
    └── ...

Usage:
    python collect_activations.py --family qwen --variant instruct
    python collect_activations.py --family llama --variant base --start_idx 100
"""

import os
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from utils import (
    set_seed, verify_environment, get_device,
    print_memory_status, cleanup_memory, aggressive_cleanup,
    check_memory_ok, evaluate_response, extract_confidence,
    create_prompt_with_confidence
)
from model_config import get_model_config, DEFAULTS


class EpistemicDataCollector:
    """Collects activations and responses for epistemic probing."""
    
    def __init__(
        self,
        model_config: dict,
        output_dir: Path,
        device: str = "auto",
        batch_size: int = 50
    ):
        self.config = model_config
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device if device != "auto" else get_device()
        
        # Create output directories with new structure
        # activations/{family}_{variant}/
        self.model_dir_name = f"{self.config['family']}_{self.config['variant']}"
        self.activations_dir = self.output_dir / "activations" / self.model_dir_name
        self.activations_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        
        # Batch accumulators
        self.reset_batch_accumulators()
        
        # Track all results for final metadata
        self.all_metadata = []
        self.current_batch_idx = 0
        
    def reset_batch_accumulators(self):
        """Reset accumulators for a new batch."""
        # Activations: will be (batch_size, n_layers, hidden_dim) per position
        self.batch_resid_first = []
        self.batch_resid_middle = []
        self.batch_resid_last = []
        
        # MLP outputs: (batch_size, n_layers, mlp_hidden_dim) per position
        self.batch_mlp_first = []
        self.batch_mlp_middle = []
        self.batch_mlp_last = []
        
        # Metadata for this batch
        self.batch_prompts = []
        self.batch_categories = []
        self.batch_correct_answers = []
        self.batch_notes = []
        self.batch_responses = []
        self.batch_confidences = []
        self.batch_evaluations = []
        self.batch_logits = []  # Top-k logits for last token
        self.batch_entropy = []  # Entropy of logit distribution
        self.batch_prompt_indices = []
        
    def load_model(self):
        """Load the model using TransformerLens."""
        print(f"\n🔄 Loading model: {self.config['model_id']}")
        print(f"   Device: {self.device}")
        print_memory_status("BEFORE MODEL LOAD")
        
        # For MPS, we need float32
        dtype = torch.float32 if self.device == "mps" else torch.float16
        
        self.model = HookedTransformer.from_pretrained(
            self.config['model_id'],
            device=self.device,
            dtype=dtype
        )
        
        print(f"✓ Model loaded successfully")
        print(f"   Layers: {self.model.cfg.n_layers}")
        print(f"   Hidden size: {self.model.cfg.d_model}")
        print(f"   Attention heads: {self.model.cfg.n_heads}")
        print_memory_status("AFTER MODEL LOAD")
        
    def unload_model(self):
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        aggressive_cleanup()
        print_memory_status("AFTER MODEL UNLOAD")
        
    def get_token_positions(self, seq_len: int) -> Dict[str, int]:
        """
        Calculate token positions to extract activations from.
        
        Args:
            seq_len: Total sequence length
            
        Returns:
            Dictionary mapping position names to indices
        """
        return {
            "first": 1,  # Skip BOS token if present, take first content token
            "middle": seq_len // 2,
            "last": seq_len - 1  # Last token (will be prompt's last token)
        }
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute entropy of the logit distribution.
        
        Args:
            logits: Logits tensor of shape (vocab_size,)
            
        Returns:
            Entropy value (higher = more uncertain)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy
    
    def extract_activations_for_batch(
        self,
        cache: dict,
        positions: Dict[str, int],
        n_layers: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract activations at specified token positions for batching.
        
        Returns activations organized for stacking into batch arrays.
        Shape per item: (n_layers, hidden_dim) for resid/mlp
        """
        extracted = {
            'resid_first': [],
            'resid_middle': [],
            'resid_last': [],
            'mlp_first': [],
            'mlp_middle': [],
            'mlp_last': [],
        }
        
        pos_map = {'first': positions['first'], 'middle': positions['middle'], 'last': positions['last']}
        
        # Extract per-layer activations
        for layer in range(n_layers):
            for pos_name, pos_idx in pos_map.items():
                # Residual stream: (hidden_dim,)
                resid_key = f"blocks.{layer}.hook_resid_post"
                if resid_key in cache:
                    resid = cache[resid_key][0, pos_idx, :].cpu().numpy()
                    extracted[f'resid_{pos_name}'].append(resid)
                
                # MLP output: (mlp_hidden_dim,)
                mlp_key = f"blocks.{layer}.mlp.hook_post"
                if mlp_key in cache:
                    mlp = cache[mlp_key][0, pos_idx, :].cpu().numpy()
                    extracted[f'mlp_{pos_name}'].append(mlp)
        
        # Stack layers: (n_layers, hidden_dim)
        for key in extracted:
            if extracted[key]:
                extracted[key] = np.stack(extracted[key], axis=0)
            else:
                extracted[key] = None
        
        return extracted
    
    def process_prompt(self, prompt_data: dict, prompt_idx: int) -> dict:
        """
        Process a single prompt: generate response and extract activations.
        Stores activations in batch accumulators.
        """
        prompt = prompt_data['prompt']
        category = prompt_data['category']
        correct_answer = prompt_data['correct_answer']
        notes = prompt_data['notes']
        
        # Determine if this is an instruct model
        is_instruct = self.config['variant'] == 'instruct'
        
        # Create the full prompt (with confidence request for instruct models)
        full_prompt = create_prompt_with_confidence(prompt, "instruct" if is_instruct else "base")
        
        # Tokenize
        tokens = self.model.to_tokens(full_prompt)
        prompt_len = tokens.shape[1]
        
        # Get token positions for activation extraction
        positions = self.get_token_positions(prompt_len)
        
        # Define which hooks to capture (residual stream + MLP, no attention)
        hooks_to_capture = []
        for layer in range(self.config['n_layers']):
            hooks_to_capture.extend([
                f"blocks.{layer}.hook_resid_post",
                f"blocks.{layer}.mlp.hook_post",
            ])
        
        # Run forward pass with hooks
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                tokens,
                names_filter=hooks_to_capture,
                return_type="logits"
            )
        
        # Extract logits for last token position (what the model predicts next)
        last_token_logits = logits[0, -1, :]  # (vocab_size,)
        
        # Compute entropy of prediction distribution
        entropy = self.compute_entropy(last_token_logits)
        
        # Get top-k logits (k=10) for storage
        top_k = 10
        top_values, top_indices = torch.topk(last_token_logits, top_k)
        top_logits = {
            'values': top_values.cpu().numpy().tolist(),
            'indices': top_indices.cpu().numpy().tolist(),
            'tokens': [self.model.tokenizer.decode([idx]) for idx in top_indices.cpu().numpy()]
        }
        
        # Extract activations at key positions
        activations = self.extract_activations_for_batch(
            cache, positions, self.config['n_layers']
        )
        
        # Clear cache to free memory
        del cache
        del logits
        cleanup_memory()
        
        # Generate response
        with torch.no_grad():
            response = self.model.generate(
                full_prompt,
                max_new_tokens=DEFAULTS['max_new_tokens'],
                temperature=DEFAULTS['temperature'],
                do_sample=False
            )
        
        # Extract just the generated part (remove the prompt)
        generated_text = response[len(full_prompt):].strip()
        
        # Extract confidence if present
        confidence = extract_confidence(generated_text) if is_instruct else None
        
        # Evaluate the response
        evaluation = evaluate_response(generated_text, correct_answer, category)
        
        # Add to batch accumulators
        self.batch_resid_first.append(activations['resid_first'])
        self.batch_resid_middle.append(activations['resid_middle'])
        self.batch_resid_last.append(activations['resid_last'])
        self.batch_mlp_first.append(activations['mlp_first'])
        self.batch_mlp_middle.append(activations['mlp_middle'])
        self.batch_mlp_last.append(activations['mlp_last'])
        
        # Add metadata
        self.batch_prompts.append(prompt)
        self.batch_categories.append(category)
        self.batch_correct_answers.append(correct_answer)
        self.batch_notes.append(notes)
        self.batch_responses.append(generated_text)
        self.batch_confidences.append(confidence)
        self.batch_evaluations.append(evaluation)
        self.batch_logits.append(top_logits)
        self.batch_entropy.append(entropy)
        self.batch_prompt_indices.append(prompt_idx)
        
        return {
            'prompt_idx': prompt_idx,
            'category': category,
            'response': generated_text,
            'confidence': confidence,
            'entropy': entropy
        }
    
    def save_batch(self):
        """Save current batch to disk and reset accumulators."""
        if not self.batch_prompts:
            return
        
        batch_file = self.activations_dir / f"batch_{self.current_batch_idx:02d}.npz"
        
        # Stack activations into batch arrays
        # Shape: (batch_size, n_layers, hidden_dim) for resid/mlp
        
        save_dict = {}
        
        # Residual stream activations
        if self.batch_resid_first[0] is not None:
            save_dict['resid_first'] = np.stack(self.batch_resid_first, axis=0)
            save_dict['resid_middle'] = np.stack(self.batch_resid_middle, axis=0)
            save_dict['resid_last'] = np.stack(self.batch_resid_last, axis=0)
        
        # MLP activations
        if self.batch_mlp_first[0] is not None:
            save_dict['mlp_first'] = np.stack(self.batch_mlp_first, axis=0)
            save_dict['mlp_middle'] = np.stack(self.batch_mlp_middle, axis=0)
            save_dict['mlp_last'] = np.stack(self.batch_mlp_last, axis=0)
        
        # Save batch metadata within npz for easy access
        save_dict['prompt_indices'] = np.array(self.batch_prompt_indices)
        save_dict['categories'] = np.array(self.batch_categories)
        save_dict['entropy'] = np.array(self.batch_entropy)
        
        np.savez_compressed(batch_file, **save_dict)
        
        # Also add to cumulative metadata
        for i in range(len(self.batch_prompts)):
            self.all_metadata.append({
                'prompt_idx': self.batch_prompt_indices[i],
                'batch_idx': self.current_batch_idx,
                'within_batch_idx': i,
                'prompt': self.batch_prompts[i],
                'category': self.batch_categories[i],
                'correct_answer': self.batch_correct_answers[i],
                'notes': self.batch_notes[i],
                'response': self.batch_responses[i],
                'confidence': self.batch_confidences[i],
                'evaluation': self.batch_evaluations[i],
                'logits': self.batch_logits[i],
                'entropy': self.batch_entropy[i]
            })
        
        print(f"💾 Saved batch_{self.current_batch_idx:02d}.npz ({len(self.batch_prompts)} prompts)")
        
        # Reset for next batch
        self.current_batch_idx += 1
        self.reset_batch_accumulators()
        cleanup_memory()
    
    def save_metadata(self):
        """Save complete metadata.json file."""
        metadata_file = self.activations_dir / "metadata.json"
        
        metadata = {
            'model_config': {
                'family': self.config['family'],
                'variant': self.config['variant'],
                'model_id': self.config['model_id'],
                'n_layers': self.config['n_layers'],
                'hidden_size': self.config['hidden_size']
            },
            'collection_info': {
                'timestamp': datetime.now().isoformat(),
                'total_prompts': len(self.all_metadata),
                'batch_size': self.batch_size,
                'num_batches': self.current_batch_idx
            },
            'activation_shapes': {
                'resid': f'(batch_size, {self.config["n_layers"]}, {self.config["hidden_size"]})',
                'mlp': f'(batch_size, {self.config["n_layers"]}, mlp_hidden_dim)'
            },
            'token_positions': ['first', 'middle', 'last'],
            'prompts': self.all_metadata
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Metadata saved to {metadata_file}")
    
    def load_checkpoint(self) -> Optional[int]:
        """
        Load checkpoint from existing batches.
        
        Returns:
            Index to resume from, or None if no checkpoint
        """
        # Check for existing batch files
        batch_files = sorted(self.activations_dir.glob("batch_*.npz"))
        if not batch_files:
            return None
        
        # Load metadata if exists
        metadata_file = self.activations_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            self.all_metadata = data.get('prompts', [])
        
        # Determine resume point
        last_batch = batch_files[-1]
        self.current_batch_idx = int(last_batch.stem.split('_')[1]) + 1
        
        # Load last batch to see how many prompts processed
        last_data = np.load(last_batch, allow_pickle=True)
        last_prompt_idx = last_data['prompt_indices'][-1]
        resume_idx = int(last_prompt_idx) + 1
        
        print(f"📂 Found {len(batch_files)} existing batches")
        print(f"   Resuming from prompt {resume_idx}")
        
        return resume_idx
    
    def run(
        self,
        dataset: List[dict],
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        resume: bool = True
    ):
        """
        Run data collection on the dataset.
        """
        # Check for existing checkpoint
        if resume:
            checkpoint_idx = self.load_checkpoint()
            if checkpoint_idx is not None:
                start_idx = max(start_idx, checkpoint_idx)
        
        if end_idx is None:
            end_idx = len(dataset)
        
        # Load model
        self.load_model()
        
        print(f"\n🚀 Starting data collection")
        print(f"   Prompts: {start_idx} to {end_idx-1} ({end_idx - start_idx} total)")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Output: {self.activations_dir}")
        
        # Process prompts
        try:
            for idx in tqdm(range(start_idx, end_idx), desc="Processing prompts"):
                # Check memory before each prompt
                if not check_memory_ok(min_available_gb=2.0):
                    print(f"\n⚠️ Low memory at prompt {idx}, saving batch and cleaning up...")
                    self.save_batch()
                    aggressive_cleanup()
                    print_memory_status("AFTER EMERGENCY CLEANUP")
                    
                    if not check_memory_ok(min_available_gb=1.0):
                        print(f"❌ Memory critically low, stopping")
                        break
                
                # Process the prompt
                self.process_prompt(dataset[idx], idx)
                
                # Save batch when full
                if len(self.batch_prompts) >= self.batch_size:
                    self.save_batch()
                    self.save_metadata()  # Update metadata after each batch
                
                # Periodic memory status
                if (idx + 1) % 25 == 0:
                    print_memory_status(f"PROMPT {idx + 1}")
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted")
            if self.batch_prompts:
                self.save_batch()
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if self.batch_prompts:
                self.save_batch()
            raise
        
        finally:
            # Save any remaining prompts in partial batch
            if self.batch_prompts:
                self.save_batch()
            
            # Save final metadata
            self.save_metadata()
            
            # Unload model
            self.unload_model()
    
        print(f"\n✓ Data collection complete!")
        print(f"   Total batches: {self.current_batch_idx}")
        print(f"   Total prompts: {len(self.all_metadata)}")


def load_dataset(csv_path: str) -> List[dict]:
    """Load the epistemic probing dataset."""
    dataset = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    return dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect activations for epistemic probing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Qwen instruct model on full dataset
    python collect_activations.py --family qwen --variant instruct
    
    # Run Llama base model starting from prompt 100
    python collect_activations.py --family llama --variant base --start_idx 100
    
    # Run with custom batch size
    python collect_activations.py --family mistral --variant instruct --batch_size 25
        """
    )
    
    parser.add_argument("--family", required=True,
                       choices=["qwen", "llama", "mistral", "yi"],
                       help="Model family to use")
    parser.add_argument("--variant", required=True,
                       choices=["base", "instruct"],
                       help="Model variant (base or instruct)")
    parser.add_argument("--dataset", default="epistemic_probing_dataset.csv",
                       help="Path to dataset CSV (default: epistemic_probing_dataset.csv)")
    parser.add_argument("--output_dir", default=".",
                       help="Base output directory (default: current directory)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting prompt index (default: 0)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="Ending prompt index, exclusive (default: all)")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Prompts per batch file (default: 50)")
    parser.add_argument("--device", default="auto",
                       choices=["auto", "mps", "cuda", "cpu"],
                       help="Device to use (default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't resume from checkpoint, start fresh")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("EPISTEMIC PROBING DATA COLLECTION")
    print("=" * 60)
    
    # Setup
    set_seed(args.seed)
    verify_environment()
    
    # Get model configuration
    model_config = get_model_config(args.family, args.variant)
    print(f"\nModel: {model_config['model_id']}")
    print(f"Layers: {model_config['n_layers']}")
    print(f"Hidden size: {model_config['hidden_size']}")
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    print(f"Dataset: {len(dataset)} prompts loaded")
    
    # Show category breakdown
    categories = {}
    for row in dataset:
        cat = row['category']
        categories[cat] = categories.get(cat, 0) + 1
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    
    # Initialize collector
    collector = EpistemicDataCollector(
        model_config=model_config,
        output_dir=Path(args.output_dir),
        device=args.device,
        batch_size=args.batch_size
    )
    
    print(f"\nOutput structure:")
    print(f"  {collector.activations_dir}/")
    print(f"  ├── batch_00.npz")
    print(f"  ├── batch_01.npz")
    print(f"  └── metadata.json")
    
    # Run collection
    print("\n" + "=" * 60)
    collector.run(
        dataset=dataset,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resume=not args.no_resume
    )
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
