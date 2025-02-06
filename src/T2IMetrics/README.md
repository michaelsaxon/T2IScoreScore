# T2IMetrics

Collection of text-to-image evaluation metrics that can be analyzed using T2IScoreScore.

## Structure

- `qga/`: Question-guided assessment metrics (VQAScore, DSGScore, etc.)
- `correlation/`: Correlation-based metrics (CLIPScore, BLIP similarity, etc.)
- `likelihood/`: Likelihood-based metrics (VQAScore, etc.)
- `models/`: Shared model implementations (CLIP, BLIP, VQA models, etc.)

## Available Metrics

#### Correlation-based metrics

Correlation-based metrics use feature embeddings extracted from the image and text to compute a similarity score.
For example, [CLIPScore](https://github.com/jmhessel/clipscore) is the cosine similarity between the image and text embeddings from CLIP.

#### Question-generation and answering assessment metrics

Question-guided assessment metrics use question generation and answering to compute a score.
For example, [TIFAScore](https://tifa-benchmark.github.io/) first generates a set of requirements from the prompt using an LLM, then checks if each requirement is satisfied using a VQA model.

#### Likelihood-based metrics

> ⚠️ **Warning**: The implementation of likelihood-based metrics is currently under development. The API for likelihood estimation methods in vision-language models may change without notice.


Likelihood-based metrics use the likelihood of tokens describing desired characteristics of the image conditioned on the image from a vision-language model to score the image.
For example, [VQAScore](https://github.com/linzhiqiu/t2v_metrics) uses the token for the answer to questions about the prompt to score the image.

## Models

The `models/` directory provides abstract interfaces for language (LM) and vision-language (VLM) models, allowing metrics to use different backends interchangeably:

### Language Models (LM)
Base class that standardizes text generation:
```python
def generate(self, prompt: str, **kwargs) -> str:
    """Generate text from prompt."""
```

Implementations can wrap:
- Local models (e.g., HuggingFace transformers)
- Cloud APIs (e.g., OpenAI GPT)
- Custom models

### Vision-Language Models (VLM)
Base class for image understanding and visual question answering:
```python
def get_answer(self, question: str, image: Union[str, Path, Image.Image]) -> str:
    """Get answer for a question about an image."""

def get_string_probability(self, prompt: str, target_str: str, image: Union[str, Path, Image.Image]) -> float:
    """Get probability of target string given prompt and image."""
```

Implementations include:
- BLIP/BLIP-2
- LLaVA
- Cloud VQA services

This abstraction allows metrics (especially Likelihood and VQA-based) to be model-agnostic, focusing on the evaluation strategy rather than specific model implementations.


## Implementation structure
```
T2IMetrics
├── T2IMetrics.correlation
│   ├── CLIPScore
│   ├── SiGLIPScore
│   └── ALIGNScore
├── T2IMetrics.qga
│   ├── TIFAScore
│   └── DSGScore
├── T2IMetrics.likelihood
│   └── VQAScore
└── T2IMetrics.models
    ├── T2IMetrics.models.lm
    └── T2IMetrics.models.vlm
```

All metrics can be directly imported from the `T2IMetrics` package using the registry.


## Adding New Metrics

1. Choose appropriate base class:
   - For correlation-based metrics (like CLIPScore): inherit from `CorrelationMetric`
   - For likelihood-based metrics: inherit from `LikelihoodMetric`
   - For VQA-based metrics: inherit from `VQAMetric`
   - For completely new approaches: inherit directly from `T2IMetric`

2. Create your metric class:

```python
from T2IMetrics.base import T2IMetric
# or from T2IMetrics.correlation import CorrelationMetric, etc.

class MyMetric(T2IMetric):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        # Initialize your models, etc.
        
    def calculate_score(self, image: Union[str, Path, Image.Image], prompt: str) -> float:
        """
        Calculate metric score for an image-prompt pair.
        
        Args:
            image: Path to image file or PIL Image object
            prompt: Text prompt to evaluate against
            
        Returns:
            Float score indicating prompt-image alignment
        """
        # Load image if needed
        img = self._load_image(image)
        
        # Implement your scoring logic
        score = ...
        
        return score
```

3. Register metric in `__init__.py`:

```python
from .mymetric import MyMetric

AVAILABLE_METRICS = {
    'mymetric': MyMetric,
    ...
}

The base `T2IMetric` class provides:
- Device management (`self.device`)
- Image loading (`self._load_image()`)
- Standard interface through `calculate_score()`

Specialized base classes provide additional functionality:
- `CorrelationMetric`: Methods for computing embedding similarities
- `LikelihoodMetric`: Methods for computing generation probabilities
- `VQAMetric`: Methods for question-answering evaluation

## Usage

```python
from T2IMetrics import load_metric

metric = load_metric('mymetric', device='cuda')
score = metric.calculate_score(image, text) 
```