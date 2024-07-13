import re
import faiss
import PyPDF2
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st

class RAG:
    def __init__(self, ret_model_name, gem_key, data_path):
        self.ret_model = SentenceTransformer(ret_model_name)
        
        genai.configure(api_key=gem_key)
        
        self.gem_model = genai  # Assign the module itself to access its methods
        
        self.psg, self.idx = self.load_idx_data(data_path)
    
    def load_idx_data(self, data_path):
        psg = self.load_data(data_path)
        
        emb = self.get_emb(psg)
        dim = emb.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(emb)
        
        return psg, idx
    
    def load_data(self, data_path):
        pdf_files = [
            f"{data_path}/placements.pdf",
            f"{data_path}/si.pdf"
        ]
        psg = []
        for k in pdf_files:
            with open(k, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    page_psg = self.split_psg(text)
                    psg.extend(page_psg)
        return psg

    def split_psg(self, text, max_len=200, overlap=50):
        text = re.sub(r'\s+', ' ', text).strip()
        sent = re.split(r'(?<=[.!?])\s+', text)
        
        psg = []
        cur_psg = ""
        for j in sent:
            if len(cur_psg) + len(j) > max_len and cur_psg:
                psg.append(cur_psg.strip())
                cur_psg = cur_psg[-overlap:] + " " + j
            else:
                cur_psg += " " + j
        if cur_psg:
            psg.append(cur_psg.strip())
        return psg
    
    def get_emb(self, texts):
        return self.ret_model.encode(texts)
    
    def retrieve(self, query, k=5):
        q_emb = self.get_emb([query])
        _, indices = self.idx.search(q_emb, k)
        return [self.psg[i] for i in indices[0]]
    
    def gen(self, query, ret_psg):
        ctx = " ".join(ret_psg)
        prompt = f"Query: {query}\nContext: {ctx}\nBased on the above context, please answer the query."
        
        response = self.gem_model.generate_text(prompt=prompt)
        return response.result  # Adjust based on the actual response structure
    
    def ans_query(self, query):
        ret_psg = self.retrieve(query)
        ans = self.gen(query, ret_psg)
        return ans

def main():
    st.title("RAG Model Query Answering")
    
    data_path = "/home/robosobo/ML_code/Datasets"
    ret_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    gem_key = "AIzaSyDL2cwR__51RddxvF8bG0oh_AK4uKwl-uM"
  
    st.write("Initializing RAG model")
    rag = RAG(ret_model_name, gem_key, data_path)
    st.write("RAG model initialized")
    
    query = st.text_input("Enter your query:")
    
    if query:
        st.write("Processing...")
        ans = rag.ans_query(query)
        st.write("### Query:")
        st.write(query)
        st.write("### Answer:")
        st.write(ans)

if __name__ == "__main__":
    main()
