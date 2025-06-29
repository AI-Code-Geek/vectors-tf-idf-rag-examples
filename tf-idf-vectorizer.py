import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextToVectorConverter:
    def __init__(self, documents):
        """
        Initialize the converter with documents

        Args:
            documents (list): List of document strings
        """
        self.documents = documents
        self.processed_docs = [doc.lower().split() for doc in documents]
        self.vocabulary = self._build_vocabulary()
        self.vocab_size = len(self.vocabulary)

    def _build_vocabulary(self):
        """Build vocabulary from all documents"""
        all_words = set()
        for doc_words in self.processed_docs:
            all_words.update(doc_words)
        return sorted(list(all_words))

    def create_tfidf_vectors_manual(self):
        """
        Create TF-IDF vectors manually using our own implementation

        Returns:
            numpy.ndarray: TF-IDF matrix where each row is a document vector
        """
        vectors = []

        # Calculate IDF for each term in vocabulary
        idf_values = {}
        for term in self.vocabulary:
            docs_containing_term = sum(1 for doc_words in self.processed_docs
                                       if term in doc_words)
            if docs_containing_term > 0:
                idf_values[term] = math.log(len(self.documents) / docs_containing_term)
            else:
                idf_values[term] = 0

        # Create vector for each document
        for doc_words in self.processed_docs:
            vector = []
            total_words = len(doc_words)

            for term in self.vocabulary:
                # Calculate TF
                term_count = doc_words.count(term)
                tf = term_count / total_words if total_words > 0 else 0

                # Calculate TF-IDF
                tfidf = tf * idf_values[term]
                vector.append(tfidf)

            vectors.append(vector)

        return np.array(vectors)

    def create_tfidf_vectors_sklearn(self):
        """
        Create TF-IDF vectors using scikit-learn

        Returns:
            tuple: (TF-IDF matrix, feature names)
        """
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform(self.documents)
        feature_names = vectorizer.get_feature_names_out()

        return tfidf_matrix.toarray(), feature_names

    def create_binary_vectors(self):
        """
        Create binary vectors (1 if term exists, 0 if not)

        Returns:
            numpy.ndarray: Binary matrix
        """
        vectors = []
        for doc_words in self.processed_docs:
            vector = []
            for term in self.vocabulary:
                vector.append(1 if term in doc_words else 0)
            vectors.append(vector)

        return np.array(vectors)

    def create_count_vectors(self):
        """
        Create count vectors (frequency of each term)

        Returns:
            numpy.ndarray: Count matrix
        """
        vectors = []
        for doc_words in self.processed_docs:
            vector = []
            for term in self.vocabulary:
                vector.append(doc_words.count(term))
            vectors.append(vector)

        return np.array(vectors)

    def calculate_cosine_similarity(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors

        Args:
            vector1, vector2: numpy arrays

        Returns:
            float: Cosine similarity score
        """
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def print_vectors(self, vectors, vector_type="TF-IDF", top_n=10):
        """
        Print vectors in a readable format

        Args:
            vectors: numpy array of vectors
            vector_type: string describing the vector type
            top_n: number of top features to show
        """
        print(f"=== {vector_type} Vectors ===\n")

        for i, vector in enumerate(vectors):
            print(f"Document {i+1}: {self.documents[i][:50]}...")
            print(f"Vector shape: {vector.shape}")

            # Show top N non-zero features
            non_zero_indices = np.nonzero(vector)[0]
            if len(non_zero_indices) > 0:
                # Get top N values
                top_indices = np.argsort(vector)[-top_n:][::-1]
                top_indices = [idx for idx in top_indices if vector[idx] > 0]

                print("Top features:")
                for idx in top_indices:
                    if idx < len(self.vocabulary):
                        print(f"  {self.vocabulary[idx]}: {vector[idx]:.6f}")
            else:
                print("  All zeros")
            print()

    def analyze_document_similarity(self, vectors):
        """
        Analyze similarity between documents using cosine similarity

        Args:
            vectors: numpy array of document vectors
        """
        print("=== Document Similarity Analysis ===\n")

        n_docs = len(vectors)
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                similarity = self.calculate_cosine_similarity(vectors[i], vectors[j])
                print(f"Document {i+1} vs Document {j+1}: {similarity:.4f}")
        print()

def create_simple_word_embeddings(documents, embedding_dim=5):
    """
    Create simple word embeddings using random initialization
    (In practice, you'd use pre-trained embeddings like Word2Vec, GloVe, etc.)

    Args:
        documents: list of document strings
        embedding_dim: dimension of embedding vectors

    Returns:
        dict: word to vector mapping
    """
    # Build vocabulary
    all_words = set()
    for doc in documents:
        all_words.update(doc.lower().split())

    # Create random embeddings (normally you'd use pre-trained)
    np.random.seed(42)  # For reproducibility
    embeddings = {}
    for word in all_words:
        embeddings[word] = np.random.randn(embedding_dim)

    return embeddings

def document_to_embedding_vector(document, word_embeddings):
    """
    Convert document to vector by averaging word embeddings

    Args:
        document: string document
        word_embeddings: dict of word to vector mappings

    Returns:
        numpy.ndarray: document embedding vector
    """
    words = document.lower().split()
    valid_embeddings = [word_embeddings[word] for word in words if word in word_embeddings]

    if valid_embeddings:
        return np.mean(valid_embeddings, axis=0)
    else:
        return np.zeros(len(next(iter(word_embeddings.values()))))

def main():
    # Define documents
    documents = [
        "Python is a high-level, interpreted programming language known for its simplicity and readability",
        "Java is a widely used, object-oriented programming language and computing platform first released in 1995",
        "Java and Python are both popular, high-level programming languages, but they have different strengths and weaknesses."
    ]

    # Initialize converter
    converter = TextToVectorConverter(documents)

    print("Vocabulary:", converter.vocabulary[:15], "...")  # Show first 15 words
    print(f"Vocabulary size: {converter.vocab_size}\n")

    # 1. Manual TF-IDF Vectors
    print("1. Creating TF-IDF vectors (manual implementation)")
    tfidf_vectors_manual = converter.create_tfidf_vectors_manual()
    converter.print_vectors(tfidf_vectors_manual, "Manual TF-IDF", top_n=5)

    # 2. Scikit-learn TF-IDF Vectors
    print("2. Creating TF-IDF vectors (scikit-learn)")
    tfidf_vectors_sklearn, feature_names = converter.create_tfidf_vectors_sklearn()
    print(f"Sklearn TF-IDF shape: {tfidf_vectors_sklearn.shape}")
    print(f"Sample features: {feature_names[:10]}")
    print()

    # 3. Binary Vectors
    print("3. Creating Binary vectors")
    binary_vectors = converter.create_binary_vectors()
    converter.print_vectors(binary_vectors, "Binary", top_n=8)

    # 4. Count Vectors
    print("4. Creating Count vectors")
    count_vectors = converter.create_count_vectors()
    converter.print_vectors(count_vectors, "Count", top_n=5)

    # 5. Document Similarity Analysis
    converter.analyze_document_similarity(tfidf_vectors_manual)

    # 6. Simple Word Embeddings Example
    print("5. Word Embeddings Example")
    word_embeddings = create_simple_word_embeddings(documents, embedding_dim=5)

    print("Sample word embeddings:")
    for word in ["python", "java", "programming"]:
        if word in word_embeddings:
            print(f"{word}: {word_embeddings[word]}")
    print()

    # Convert documents to embedding vectors
    embedding_vectors = []
    for doc in documents:
        doc_vector = document_to_embedding_vector(doc, word_embeddings)
        embedding_vectors.append(doc_vector)

    embedding_vectors = np.array(embedding_vectors)
    print("Document embedding vectors shape:", embedding_vectors.shape)

    # Analyze similarity using embeddings
    print("\n=== Embedding-based Document Similarity ===")
    converter.analyze_document_similarity(embedding_vectors)

if __name__ == "__main__":
    main()
