import argparse
import subprocess

import pandas as pd
from openai_utils import openai_completion
from query_utils import generate_dsg


class DSG_QuestionGenerator:
    def __init__(self, api_key):
        self.api_key = api_key

    def process_prompts(self, input_file_path, csv_output_path):
        prompts = self.read_prompts_from_file(input_file_path)

        csv_data = {'id': [], 'prompt': [], 'question_id': [], 'question': [], 'choices': [], 'answer': []}

        for i, prompt in enumerate(prompts):
            INPUT_TEXT_PROMPT = prompt
            id2prompts = {
                f'{i}': {
                    'input': INPUT_TEXT_PROMPT,
                }
            }

            id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
                id2prompts,
                generate_fn=openai_completion)

              # Extract and store the questions in a list
            questions = id2question_outputs[f'{i}']['output'].split('\n')
            questions = [q.strip().split('|')[-1].strip() for q in questions if q.strip()]  # Extract the question part

            for j, question in enumerate(questions):
                csv_data['id'].append(f'{i}')
                csv_data['prompt'].append(prompt)
                csv_data['question'].append(question)
                csv_data['question_id'].append(f'{j}')
                csv_data['choices'].append("|".join(["yes", "no"]))
                csv_data['answer'].append("yes")

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)

    def read_prompts_from_file(self, input_file_path):
        with open(input_file_path, 'r') as file:
            prompts = file.readlines()
        return [prompt.strip() for prompt in prompts]


def main():
    parser = argparse.ArgumentParser(description="Process prompts with OpenAI API.")

    parser.add_argument("--api_key", type=str, help="OpenAI API key.")
    parser.add_argument("--input_file", type=str, help="Path to the input file with prompts.")
    parser.add_argument("--output_csv", type=str, help="Path to save the output CSV file.")

    args = parser.parse_args()

    api_key = args.api_key
    input_file_path = args.input_file
    output_csv_path = args.output_csv

    question_creator = DSG_QuestionGenerator(api_key)
    repository_url = "https://github.com/j-min/DSG.git"
    target_directory = "SGD"

    subprocess.run(["git", "clone", repository_url, target_directory])
    subprocess.run(["cd", target_directory])

    question_creator.process_prompts(input_file_path, output_csv_path)

if __name__ == "__main__":
    main()