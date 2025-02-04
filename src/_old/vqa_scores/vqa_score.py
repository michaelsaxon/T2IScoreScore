from abc import ABC, abstractmethod

class VQAScorer(ABC):
    def __init__(self):
        """
        Initialize the VQA scorer.
        """
        pass

    @abstractmethod
    def get_answer(self, question, image_path):
        """
        Generate an answer for a given question and image.

        Args:
            question (str): The question to be answered.
            image_path (str): Path to the image associated with the question.

        Returns:
            str: The generated answer to the question.
        """
        pass
