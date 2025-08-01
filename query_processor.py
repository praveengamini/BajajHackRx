import chromadb
from llm_service import LocalGeminiChatLLM
from config import Config

class QueryProcessor:
    def __init__(self, llm: LocalGeminiChatLLM):
        self.llm = llm
    
    def process_query(self, query: str, collection: chromadb.Collection, document_text: str) -> str:
        """Process a single query against the document using ChromaDB"""
        
        try:
            # Query the collection for relevant documents
            results = collection.query(
                query_texts=[query],
                n_results=Config.RETRIEVAL_K
            )
            
            # Extract relevant context
            if results['documents'] and len(results['documents']) > 0:
                context = "\n\n".join(results['documents'][0])
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