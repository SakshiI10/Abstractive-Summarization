from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from rouge import Rouge
import os

def calculate_rouge_scores(reference_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary, avg=True)
    rouge_1 = scores['rouge-1']['f']
    rouge_2 = scores['rouge-2']['f']
    rouge_l = scores['rouge-l']['f']
    return rouge_1, rouge_2, rouge_l

def summarize_and_save(dataset_path, output_path, reference_dir, sentences_count=2):
    total_rouge_1, total_rouge_2, total_rouge_l = 0, 0, 0
    total_files = 0

    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt"):
            document_path = os.path.join(dataset_path, filename)
            output_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_summary.txt")

            # Load your document
            parser = PlaintextParser.from_file(document_path, Tokenizer("english"))

            # Use LSA for summarization
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=sentences_count)

            # Save the summary to a new file
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write("Summary:\n")
                generated_summary = "\n".join(str(sentence) for sentence in summary)
                output_file.write(generated_summary)

                # Calculate ROUGE scores for each file in the reference directory
                for reference_file in os.listdir(reference_dir):
                    if reference_file.endswith(".txt"):
                        reference_path = os.path.join(reference_dir, reference_file)
                        with open(reference_path, "r", encoding="utf-8") as reference_file:
                            reference_summary = reference_file.read()
                            rouge_1, rouge_2, rouge_l = calculate_rouge_scores(reference_summary, generated_summary)

                            # Accumulate scores
                            total_rouge_1 += rouge_1
                            total_rouge_2 += rouge_2
                            total_rouge_l += rouge_l
                            total_files += 1

    # Calculate averages
    if total_files > 0:
        avg_rouge_1 = total_rouge_1 / total_files
        avg_rouge_2 = total_rouge_2 / total_files
        avg_rouge_l = total_rouge_l / total_files

        # Print combined ROUGE Scores
        print("\nCombined ROUGE Scores:")
        print(f"Average ROUGE-1 Score: {avg_rouge_1}")
        print(f"Average ROUGE-2 Score: {avg_rouge_2}")
        print(f"Average ROUGE-L Score: {avg_rouge_l}")
    else:
        print("No files processed.")

# Example usage with a dataset directory, output directory, and reference summary directory
dataset_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\test_output\articles"
output_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\Sumy\Summary"
reference_summary_dir = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\test_output\reference"

summarize_and_save(dataset_path, output_path, reference_summary_dir)
