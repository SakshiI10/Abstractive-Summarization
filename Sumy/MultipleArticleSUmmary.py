from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt')
import os

def summarize_and_save(dataset_path, output_path, sentences_count=2):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt"):
            document_path = os.path.join(dataset_path, filename)
            output_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_summary.txt")

            print(f"Summarizing {document_path} and saving to {output_file_path}:")
            
            # Load your document
            parser = PlaintextParser.from_file(document_path, Tokenizer("english"))

            # Use LSA for summarization
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=sentences_count)

            # Save the summary to a new file
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write("Summary:\n")
                for sentence in summary:
                    output_file.write(str(sentence) + "\n")
                #print(f"Summary saved to {output_file_path}\n")

# Example usage with a dataset directory and output directory
dataset_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\test_output\articles"
output_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\Sumy\Summary"
summarize_and_save(dataset_path, output_path)
