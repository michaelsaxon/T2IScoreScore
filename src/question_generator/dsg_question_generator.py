import argparse
import openai

import pandas as pd
from DSG.openai_utils import openai_completion
from DSG.query_utils import generate_dsg
from parse_utils import parse_dependency_output


class DSG_QuestionGenerator:

    def process_prompts(self, input_file_path, csv_output_path):
        prompts = self.read_prompts_from_file(input_file_path)

        csv_data = {'id': [], 'prompt': [], 'question_id': [], 'parent_question_id': [],  'question': [], 'choices': [], 'answer': []}

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

            qid2dependency = parse_dependency_output(id2dependency_outputs[f'{i}']['output'])
            for key, values in qid2dependency.items():
              qid2dependency[key] = [value - 1 for value in values]

              # Extract and store the questions in a list
            questions = id2question_outputs[f'{i}']['output'].split('\n')
            questions = [q.strip().split('|')[-1].strip() for q in questions if q.strip()]  # Extract the question part

            for j, question in enumerate(questions):
                csv_data['id'].append(f'{i}')
                csv_data['prompt'].append(prompt)
                csv_data['question'].append(question.replace(',', ''))
                csv_data['question_id'].append(f'{j}')
                parent_id = qid2dependency.get(j+1)
                if parent_id:
                    parent_id_string = ','.join(str(id) for id in parent_id)
                    csv_data['parent_question_id'].append(parent_id_string.replace(',', '-'))
                else:
                    csv_data['parent_question_id'].append('')
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

    parser.add_argument("-a", "--api_key", type=str, help="OpenAI API key.")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input file with prompts.")
    parser.add_argument("-o", "--output_csv", type=str, help="Path to save the output CSV file.")


    args = parser.parse_args()

    openai.api_key = args.api_key
    input_file_path = args.input_file
    output_csv_path = args.output_csv

    question_creator = DSG_QuestionGenerator()

    question_creator.process_prompts(input_file_path, output_csv_path)

if __name__ == "__main__":
    main()
