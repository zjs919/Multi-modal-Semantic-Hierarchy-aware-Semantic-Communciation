import numpy as np
import os

class EmbeddingLoader:
    def __init__(self):
        self.entity_embedding = None
        self.relation_embedding = None
        self.load_embeddings()

    def load_embeddings(self):
        current_directory = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        
        entity_embedding_path = os.path.join(current_directory, '../Embeddings/entity_embedding.npy')
        relation_embedding_path = os.path.join(current_directory, '../Embeddings/relation_embedding.npy')
        
        if os.path.exists(entity_embedding_path) and os.path.exists(relation_embedding_path):
            try:
                self.entity_embedding = np.load(entity_embedding_path)
                self.relation_embedding = np.load(relation_embedding_path)
                print("Embeddings loaded successfully.")
            except Exception as e:
                print(f"Error loading embeddings: {str(e)}")
        else:
            print("Embedding files not found in the current directory.")

