import torch
import logging

logger = logging.getLogger(__name__)

class GenerationStreamer:
    """Streamer for tracking token generation progress in transformer models."""
    
    def __init__(self, processor):
        """
        Initialize the streamer.
        
        Args:
            processor: HuggingFace processor for decoding tokens
        """
        self.generated_text = ""
        self.last_printed_len = 0
        self.processor = processor
    
    def put(self, token_ids):
        """
        Process new tokens as they're generated.
        
        Args:
            token_ids: New token IDs from the model
        """
        # Ensure token_ids is a tensor and has the right shape
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor([token_ids])
        if len(token_ids.shape) == 1:
            token_ids = token_ids.unsqueeze(0)
        
        # Decode the new tokens
        text = self.processor.batch_decode(token_ids, skip_special_tokens=True)[0]
        self.generated_text = text
        
        if logger.isEnabledFor(logging.DEBUG):
            # Only print if we have new tokens
            if len(text) > self.last_printed_len:
                print(f"\rGenerating: {text}", end="", flush=True)
                self.last_printed_len = len(text)
    
    def end(self):
        """Called when generation is complete."""
        if logger.isEnabledFor(logging.DEBUG):
            print()  # Add newline after generation 