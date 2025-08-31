#!/usr/bin/env python3
"""
Simple OpenAI + LangGraph + LangMem Example
Demonstrates basic memory storage and retrieval capabilities
"""

import os
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def setup_agent():
    """Set up an agent with memory capabilities"""
    
    # Initialize OpenAI model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Set up storage with vector search capabilities
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )
    
    # Create memory tools
    memory_tools = [
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ]
    
    # Create agent with memory capabilities
    agent = create_react_agent(
        llm,
        tools=memory_tools,
        store=store,
    )
    
    return agent

def main():
    """Main function to demonstrate memory capabilities"""
    
    print("🤖 Setting up AI Agent with Memory...")
    agent = setup_agent()
    
    print("\n📝 Testing Memory Storage...")
    
    # Test 1: Store a memory
    print("1. Storing a memory about user preferences...")
    response1 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Remember that I prefer dark mode and love coffee."
        }]
    })
    print(f"Response: {response1['messages'][-1].content}")
    
    # Test 2: Retrieve the stored memory
    print("\n2. Retrieving stored memory...")
    response2 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "What are my preferences?"
        }]
    })
    print(f"   Response: {response2['messages'][-1].content}")
    
    # Test 3: Store another memory
    print("\n3. Storing another memory...")
    response3 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Remember that I work as a software developer and use Python."
        }]
    })
    print(f"   Response: {response3['messages'][-1].content}")
    
    # Test 4: Search for specific information
    print("\n4. Searching for specific information...")
    response4 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "What do you know about my work?"
        }]
    })
    print(f"   Response: {response4['messages'][-1].content}")
    
    print("\n✅ Memory demonstration completed!")

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have a valid OpenAI API key and sufficient credits.")
