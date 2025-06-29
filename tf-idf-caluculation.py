import math
from collections import Counter

class TFIDFCalculator:
    def __init__(self, documents):
        """
        Initialize the TF-IDF calculator with a list of documents

        Args:
            documents (list): List of document strings
        """
        self.documents = documents
        self.num_documents = len(documents)
        self.processed_docs = [doc.lower().split() for doc in documents]

    def calculate_tf(self, term, document_words):
        """
        Calculate Term Frequency (TF) for a term in a document
        TF(t,d) = Number of times term appears in document / Total number of terms in document

        Args:
            term (str): The term to calculate TF for
            document_words (list): List of words in the document

        Returns:
            float: TF value
        """
        term_count = document_words.count(term.lower())
        total_words = len(document_words)
        return term_count / total_words if total_words > 0 else 0

    def calculate_idf(self, term):
        """
        Calculate Inverse Document Frequency (IDF) for a term across all documents
        IDF(t,D) = log(Total number of documents / Number of documents containing the term)

        Args:
            term (str): The term to calculate IDF for

        Returns:
            float: IDF value
        """
        documents_containing_term = sum(1 for doc_words in self.processed_docs
                                        if term.lower() in doc_words)
        if documents_containing_term == 0:
            return 0
        return math.log(self.num_documents / documents_containing_term)

    def calculate_tfidf(self, term, document_index):
        """
        Calculate TF-IDF for a term in a specific document
        TF-IDF(t,d,D) = TF(t,d) * IDF(t,D)

        Args:
            term (str): The term to calculate TF-IDF for
            document_index (int): Index of the document

        Returns:
            float: TF-IDF value
        """
        if document_index >= self.num_documents:
            raise IndexError("Document index out of range")

        tf = self.calculate_tf(term, self.processed_docs[document_index])
        idf = self.calculate_idf(term)
        return tf * idf

    def analyze_term(self, term):
        """
        Analyze a term across all documents and return comprehensive results

        Args:
            term (str): The term to analyze

        Returns:
            dict: Dictionary containing analysis results
        """
        results = {
            'term': term,
            'idf': self.calculate_idf(term),
            'documents': []
        }

        for i, (doc, doc_words) in enumerate(zip(self.documents, self.processed_docs)):
            tf = self.calculate_tf(term, doc_words)
            tfidf = self.calculate_tfidf(term, i)

            doc_result = {
                'index': i + 1,
                'content': doc,
                'word_count': len(doc_words),
                'term_frequency': doc_words.count(term.lower()),
                'tf': tf,
                'tfidf': tfidf
            }
            results['documents'].append(doc_result)

        return results

    def print_analysis(self, term):
        """
        Print detailed analysis of a term across all documents

        Args:
            term (str): The term to analyze
        """
        results = self.analyze_term(term)

        print(f"=== TF-IDF Analysis for '{term}' ===\n")
        print(f"IDF({term}) = {results['idf']:.6f}\n")

        for doc_data in results['documents']:
            print(f"Document {doc_data['index']}:")
            print(f"  Content: \"{doc_data['content']}\"")
            print(f"  Total words: {doc_data['word_count']}")
            print(f"  Term '{term}' appears: {doc_data['term_frequency']} times")
            print(f"  TF({term}) = {doc_data['term_frequency']}/{doc_data['word_count']} = {doc_data['tf']:.6f}")
            print(f"  TF-IDF({term}) = {doc_data['tf']:.6f} × {results['idf']:.6f} = {doc_data['tfidf']:.6f}")
            print()

        # Summary
        print("Summary of TF-IDF values:")
        for doc_data in results['documents']:
            print(f"Document {doc_data['index']}: {doc_data['tfidf']:.6f}")

def main():
    # Define the documents
    documents = [
        "Python is a high-level, interpreted programming language known for its simplicity and readability",
        "Java is a widely used, object-oriented programming language and computing platform first released in 1995",
        "Java and Python are both popular, high-level programming languages, but they have different strengths and weaknesses."
    ]

    # Create TF-IDF calculator
    calculator = TFIDFCalculator(documents)

    # Analyze the term "Python"
    calculator.print_analysis("Python")

    print("\n" + "="*60 + "\n")

    # You can also analyze other terms
    calculator.print_analysis("Java")

    print("\n" + "="*60 + "\n")

    # Example of getting raw results for further processing
    python_results = calculator.analyze_term("Python")
    print("Raw results for 'Python':")
    for doc in python_results['documents']:
        print(f"Doc {doc['index']}: TF-IDF = {doc['tfidf']:.6f}")

if __name__ == "__main__":
    main()
