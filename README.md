# GitHub-Coding-Assessment
# GitHub Coding Assessment Solution
import time
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
import math


# Utility functions for vector similarity (simple cosine similarity)
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def simple_text_to_vector(text: str) -> List[float]:
  
    vec = [0] * 26
    for ch in text.lower():
        if 'a' <= ch <= 'z':
            vec[ord(ch) - ord('a')] += 1
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class MemoryAgent:
    def __init__(self):
 
        self.conversation_memory: List[Dict[str, Any]] = []
        self.knowledge_base: List[Dict[str, Any]] = []
        self.agent_state_memory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def store_conversation(self, message: str, agent: str):
        record = {
            "timestamp": time.time(),
            "message": message,
            "agent": agent
        }
        self.conversation_memory.append(record)
        print(f"[MemoryAgent] Stored conversation message from {agent}")

    def store_knowledge(self, topic: str, content: str, source: str, agent: str, confidence: float):
        record = {
            "timestamp": time.time(),
            "topic": topic,
            "content": content,
            "source": source,
            "agent": agent,
            "confidence": confidence,
            "vector": simple_text_to_vector(content)
        }
        self.knowledge_base.append(record)
        print(f"[MemoryAgent] Stored knowledge on topic '{topic}' from {agent}")

    def store_agent_state(self, agent: str, task: str, result: str):
        record = {
            "timestamp": time.time(),
            "task": task,
            "result": result
        }
        self.agent_state_memory[agent].append(record)
        print(f"[MemoryAgent] Stored state for {agent} on task '{task}'")

    def search_knowledge(self, topic_keywords: List[str], query_text: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
       
        results = []
        query_vec = simple_text_to_vector(query_text)
        for record in self.knowledge_base:
            if any(kw.lower() in record["topic"].lower() for kw in topic_keywords):
                sim = cosine_similarity(query_vec, record["vector"])
                if sim >= threshold:
                    results.append(record)
        print(f"[MemoryAgent] Search found {len(results)} records for keywords {topic_keywords}")
        return results

    def retrieve_conversation(self, topic_keywords: List[str]) -> List[Dict[str, Any]]:
      
        results = []
        for record in self.conversation_memory:
            if any(kw.lower() in record["message"].lower() for kw in topic_keywords):
                results.append(record)
        print(f"[MemoryAgent] Retrieved {len(results)} conversation records for keywords {topic_keywords}")
        return results


class ResearchAgent:
    def __init__(self, memory_agent: MemoryAgent):
       
        self.knowledge_base = {
            "neural networks": "Main types: CNN, RNN, LSTM, GAN, Transformer.",
            "transformer architectures": "Transformers use self-attention, are highly parallelizable.",
            "reinforcement learning": "Recent papers focus on deep RL, policy gradients, exploration.",
            "machine learning optimization techniques": "Gradient descent, Adam, RMSProp, Adagrad."
        }
        self.memory_agent = memory_agent

    def research(self, query: str) -> Dict[str, Any]:
     
        print(f"[ResearchAgent] Researching query: {query}")
        for key in self.knowledge_base:
            if key in query.lower():
                content = self.knowledge_base[key]
                confidence = 0.9  # mock confidence
                self.memory_agent.store_agent_state("ResearchAgent", query, content)
                self.memory_agent.store_knowledge(key, content, "MockKnowledgeBase", "ResearchAgent", confidence)
                return {"content": content, "confidence": confidence}
        
        fallback_content = "No relevant information found."
        self.memory_agent.store_agent_state("ResearchAgent", query, fallback_content)
        return {"content": fallback_content, "confidence": 0.1}


class AnalysisAgent:
    def __init__(self, memory_agent: MemoryAgent):
        self.memory_agent = memory_agent

    def analyze(self, data: str) -> Dict[str, Any]:
        print(f"[AnalysisAgent] Analyzing data: {data}")
     
        words = data.split()
        analysis_result = f"Analysis: The data contains {len(words)} words."
        confidence = 0.85
        self.memory_agent.store_agent_state("AnalysisAgent", "Analyze data", analysis_result)
        self.memory_agent.store_knowledge("analysis", analysis_result, "AnalysisAgent", "AnalysisAgent", confidence)
        return {"content": analysis_result, "confidence": confidence}


class Coordinator:
    def __init__(self):
        self.memory_agent = MemoryAgent()
        self.research_agent = ResearchAgent(self.memory_agent)
        self.analysis_agent = AnalysisAgent(self.memory_agent)
        self.conversation_context: List[Dict[str, Any]] = []

    def log(self, message: str):
        print(f"[Coordinator] {message}")

    def receive_query(self, query: str) -> str:
        self.log(f"Received query: {query}")
        self.memory_agent.store_conversation(query, "User ")

       
        if any(keyword in query.lower() for keyword in ["analyze", "compare", "summarize"]):
       
            self.log("Complex query detected, decomposing tasks.")
            research_result = self.research_agent.research(query)
            analysis_result = self.analysis_agent.analyze(research_result["content"])
            final_answer = f"Research result: {research_result['content']}\nAnalysis result: {analysis_result['content']}"
            self.memory_agent.store_conversation(final_answer, "Coordinator")
            self.memory_agent.store_knowledge("final_answer", final_answer, "Coordinator", "Coordinator", 0.95)
            return final_answer
        elif any(keyword in query.lower() for keyword in ["what did we learn", "earlier", "discuss"]):
           
            keywords = query.lower().split()
            knowledge = self.memory_agent.search_knowledge(keywords, query)
            if knowledge:
                summary = "\n".join([k["content"] for k in knowledge])
                self.memory_agent.store_conversation(summary, "Coordinator")
                return f"Memory recall:\n{summary}"
            else:
                return "No prior knowledge found on that topic."
        else:
       
            research_result = self.research_agent.research(query)
            self.memory_agent.store_conversation(research_result["content"], "Coordinator")
            return research_result["content"]


# Example usage and test scenarios
if __name__ == "__main__":
    coordinator = Coordinator()

    queries = [
        "What are the main types of neural networks?",
        "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
        "What did we discuss about neural networks earlier?",
        "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
        "Compare two machine-learning approaches and recommend which is better for our use case."
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n--- Scenario {i} ---")
        response = coordinator.receive_query(q)
        print(f"Response:\n{response}")
