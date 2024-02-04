from abc import ABC, abstractmethod

class SimScorer(ABC):
    """
    Abstract base class for similarity scoring methods.

    This abstract class defines the structure for custom scoring methods used to calculate similarity scores between an image and a prompt.
    Any scoring method should inherit from this class and implement the 'calculate_score' method.

    Attributes:
    None

    Methods:
    calculate_score(image, prompt):
    Calculate a similarity score between an image and a prompt using the custom scoring method.

    Parameters:
    image (str): The image file name for scoring. ex. "1-0.jpg".
    prompt (str): The prompt data for scoring.

    Returns:
    float: The calculated similarity score.
    """
    @abstractmethod
    def calculate_score(self, image, prompt):
        """
        Calculate a similarity score between an image and a prompt using the custom scoring method.

        Parameters:
        image (str): The image file name for scoring. ex. "1-0.jpg".
        prompt (str): The prompt data for scoring.

        Returns:
        float: The calculated similarity score.
        """
        pass