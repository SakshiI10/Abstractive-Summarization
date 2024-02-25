from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt')

# Load your document
document_path = "000000_article.txt"
parser = PlaintextParser.from_file(document_path, Tokenizer("english"))

# Use LSA for summarization
summarizer = LsaSummarizer()
summary = summarizer(parser.document, sentences_count=2)

# Print the summary
for sentence in summary:
    print(sentence)