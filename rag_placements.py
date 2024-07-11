import re
import torch
import faiss
import PyPDF2
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

class RAGModel:
    def __init__(self, retriever_model_name, generator_model_name, data_path):
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
        self.retriever_model = AutoModel.from_pretrained(retriever_model_name)
        
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        
        self.passages, self.faiss_index = self.load_and_index_data(data_path)
        
    def load_and_index_data(self, data_path):
        passages = self.load_data(data_path)
        
        embeddings = self.get_embeddings(passages)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return passages, index
    
    def load_data(self, data_path):
        pdf_files = [
            f"{data_path}/placement.pdf",
            f"{data_path}/si.pdf"
        ]
        passages = []

        for k in pdf_files:
            with open(k, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    page_passages = self.split_into_passages(text)
                    passages.extend(page_passages)

        return passages

    def split_into_passages(self, text, max_length=200, overlap=50):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        passages = []
        current_passage = ""

        for j in sentences:
            if len(current_passage) + len(j) > max_length and current_passage:
                passages.append(current_passage.strip())
                current_passage = current_passage[-overlap:] + " " + j
            else:
                current_passage += " " + j

        if current_passage:
            passages.append(current_passage.strip())

        return passages
    
    def get_embeddings(self, texts):
        embeddings = []
        for i in texts:
            inputs = self.retriever_tokenizer(i, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.retriever_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)
    
    def retrieve(self, query, k=5):
        query_embedding = self.get_embeddings([query])
        _, indices = self.faiss_index.search(query_embedding, k)
        return [self.passages[i] for i in indices[0]]
    
    def generate(self, query, retrieved_passages):
        context = " ".join(retrieved_passages)
        inputs = self.generator_tokenizer(f"query: {query} context: {context}", return_tensors="pt", max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = self.generator_model.generate(**inputs, max_length=150)
        
        return self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def answer_query(self, query):
        retrieved_passages = self.retrieve(query)
        answer = self.generate(query, retrieved_passages)
        return answer

def main():
    data_path = "/home/robosobo/ML_code/"
    retriever_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model_name = "google/flan-t5-base"

    print("Initializing RAG model")
    rag_model = RAGModel(retriever_model_name, generator_model_name, data_path)
    print("RAG model initialized")

    while True:
        query = input("\nEnter your query (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            print("Thank you. Goodbye!")
            break

        print("Processing your query...")
        answer = rag_model.answer_query(query)
        print("\nQuery:", query)
        print("Answer:", answer)

if __name__ == "__main__":
    main()