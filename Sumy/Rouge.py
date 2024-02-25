from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from rouge import Rouge
import nltk
import os

nltk.download('punkt')

def summarize_and_save(dataset_path, output_path, rouge_output_file, sentences_count=2):
    rouge = Rouge()
    
    with open(rouge_output_file, "w", encoding="utf-8") as rouge_file:
        for filename in os.listdir(dataset_path):
            if filename.endswith(".txt"):
                document_path = os.path.join(dataset_path, filename)
                output_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_summary.txt")
                ''' 
                print(f"Summarizing {document_path} and saving to {output_file_path}:")
                
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
                    print(f"Summary saved to {output_file_path}\n") '''
                    
                # Calculate ROUGE scores
                reference_summary_path = os.path.join(dataset_path, f"{os.path.splitext(filename)[0]}_gold_summary.txt")
                with open(reference_summary_path, "r", encoding="utf-8") as reference_file:
                    reference_summary = reference_file.read()

                rouge_scores = rouge.get_scores(generated_summary, reference_summary)
                print(f"ROUGE Scores: {rouge_scores}\n")
                    
                # Write ROUGE scores to the output file
                rouge_file.write(f"File: {filename}\n")
                rouge_file.write(f"ROUGE Scores: {rouge_scores}\n\n")

# Example usage with a dataset directory, output directory, and Rouge output file
dataset_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\test_output\articles"
output_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\Sumy\Summary"
rouge_output_file = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\Sumy\rouge_scores.txt"

summarize_and_save(dataset_path, output_path, rouge_output_file)
