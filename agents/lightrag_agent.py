import os
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

load_dotenv()

WORKING_DIR = "./radix-docs"

print(WORKING_DIR)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()

    return rag


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG


# Create the Pydantic AI agent
lightrag_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=RAGDeps,
    system_prompt=(
        "You are a knowledgeable assistant designed to answer questions about Radix documentation.\n"
        "Use the retrieve tool to extract pertinent information from the documentation.\n"
        "If the information is unavailable, provide your best general knowledge response."
    ),
    # message_handler=add_messages  # Use langgraph's message handler
)


@lightrag_agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str) -> str:
    """Retrieve relevant documents from knowledge base based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    return await context.deps.lightrag.aquery(
        search_query, param=QueryParam(mode="mix")
    )


async def run_agent(question: str) -> str:
    """Run the RAG agent to answer a question about Radix docs.
    
    Args:
        question: The question to answer.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    lightrag = await initialize_rag()
    deps = RAGDeps(lightrag=lightrag)

    
    
    # Run the agent
    result = await lightrag_agent.run(question, deps=deps)
    
    return result.data


async def main():
    """Main function to enable conversational interaction with the RAG agent."""
    print("Welcome to the Radix Documentation Assistant!")
    print("Type your questions below. Type 'exit', 'quit', or 'bye' to end the conversation.\n")

    # Initialize RAG dependencies
    lightrag = await initialize_rag()
    deps = RAGDeps(lightrag=lightrag)

    # Initialize conversation history
    conversation_history = []

    while True:
        try:
            user_input = input("[You]: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            # Prepend conversation history to the input if needed
            if conversation_history:
                context = "\n".join(conversation_history)
                combined_input = f"{context}\n[You]: {user_input}"
            else:
                combined_input = user_input

            # Run the agent
            result = await lightrag_agent.run(combined_input, deps=deps)

            # Add this round to the conversation memory
            conversation_history.append(f"[You]: {user_input}")
            conversation_history.append(f"[Assistant]: {result.data}")

            print("[Assistant]:", result.data)

        except Exception as e:
            print("An error occurred:", e)


if __name__ == "__main__":
    asyncio.run(main())