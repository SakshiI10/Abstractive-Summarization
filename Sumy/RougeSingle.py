from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from rouge import Rouge

def calculate_rouge(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]

def summarize_and_save(input_path, output_path, reference_path, sentences_count=2):
    # Load your document
    parser = PlaintextParser.from_file(input_path, Tokenizer("english"))

    # Use LSA for summarization
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=sentences_count)

    # Save the summary to a new file
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("Summary:\n")
        for sentence in summary:
            output_file.write(str(sentence) + "\n")

    # Load reference summary
    with open(reference_path, "r", encoding="utf-8") as reference_file:
        reference_summary = reference_file.read()

    # Calculate Rouge scores
    rouge_scores = calculate_rouge(open(output_path, "r", encoding="utf-8").read(), reference_summary)

    # Extract and print Rouge-1 scores
    rouge_1_scores = rouge_scores['rouge-1']
    print(f"Rouge-1: {rouge_1_scores['f']}")

    # Extract and print Rouge-2 scores
    rouge_2_scores = rouge_scores['rouge-2']
    print(f"Rouge-2: {rouge_2_scores['f']}")

    # Extract and print Rouge-L scores
    rouge_l_scores = rouge_scores['rouge-l']
    print(f"Rouge-L: {rouge_l_scores['f']}")

# Example usage with a single input, output, and reference file
input_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\Sumy\000000_article.txt"
output_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\Sumy\000000_article_summary.txt"
reference_path = r"D:\Engineering\4_yr\Major Project\Pointer Generator Algorithm\Sumy\000000_reference.txt"
summarize_and_save(input_path, output_path, reference_path)