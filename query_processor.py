from langchain_community.vectorstores import FAISS
from llm import LocalGeminiChatLLM
from config import Config

class QueryProcessor:
    def __init__(self, llm: LocalGeminiChatLLM):
        self.llm = llm
    
    def process_query(self, query: str, vector_store: FAISS, document_text: str) -> str:
        """Process a single query against the document using FAISS"""
        
        try:
            # Perform similarity search
            similar_docs = vector_store.similarity_search(
                query, 
                k=Config.RETRIEVAL_K
            )
            
            # Extract relevant context
            if similar_docs:
                context = "\n\n".join([doc.page_content for doc in similar_docs])
            else:
                context = "No relevant context found."
            
            # Create enhanced prompt for insurance/legal domain
            prompt = f"""You are an expert in insurance policy analysis and legal document interpretation. 
Your task is to answer questions about policy documents with high accuracy and provide specific details.

DOCUMENT CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided document context
2. If the answer exists in the document, provide specific details including:
   - Exact conditions, time periods, amounts, percentages
   - Any relevant limitations or exclusions
   - Specific clause references where applicable
3. If the information is not found in the document, clearly state "This information is not available in the provided document"
4. For insurance queries, focus on coverage details, waiting periods, conditions, and exclusions
5. Be precise and factual - avoid speculation or general knowledge
6. Use clear, professional language

ANSWER:"""

            answer = self.llm._call(prompt)
            return answer.strip()
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def process_query_with_scores(self, query: str, vector_store: FAISS, document_text: str) -> dict:
        """Process query and return answer with similarity scores"""
        
        try:
            # Perform similarity search with scores
            similar_docs_with_scores = vector_store.similarity_search_with_score(
                query, 
                k=Config.RETRIEVAL_K
            )
            
            if similar_docs_with_scores:
                # Extract content and scores
                context_parts = []
                scores = []
                for doc, score in similar_docs_with_scores:
                    context_parts.append(doc.page_content)
                    scores.append(float(score))
                
                context = "\n\n".join(context_parts)
                avg_score = sum(scores) / len(scores) if scores else 0
            else:
                context = "No relevant context found."
                scores = []
                avg_score = 0
            
            prompt = f"""You are an expert in insurance policy analysis and legal document interpretation. 
Your task is to answer questions about policy documents with high accuracy and provide specific details.

DOCUMENT CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided document context
2. If the answer exists in the document, provide specific details including:
   - Exact conditions, time periods, amounts, percentages
   - Any relevant limitations or exclusions
   - Specific clause references where applicable
3. If the information is not found in the document, clearly state "This information is not available in the provided document"
4. For insurance queries, focus on coverage details, waiting periods, conditions, and exclusions
5. Be precise and factual - avoid speculation or general knowledge
6. Use clear, professional language

ANSWER:"""

            answer = self.llm._call(prompt)
            
            return {
                "answer": answer.strip(),
                "similarity_scores": scores,
                "average_score": avg_score,
                "num_retrieved_chunks": len(similar_docs_with_scores)
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "similarity_scores": [],
                "average_score": 0,
                "num_retrieved_chunks": 0
            }