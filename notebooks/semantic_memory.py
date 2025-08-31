#!/usr/bin/env python3
"""
SEMANTIC MEMORY EXAMPLE - User Preferences & Facts
Stores: "What I know about the user"
"""

import os
from dotenv import load_dotenv
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def setup_semantic_agent():
    """Set up agent with semantic memory capabilities"""
    
    # Initialize OpenAI model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Setup semantic memory store
    store = InMemoryStore(
        index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
    )
    
    # Create namespace for user facts
    namespace = ("user_facts", "user123")
    
    # Create semantic memory tools
    semantic_tools = [
        create_manage_memory_tool(namespace),
        create_search_memory_tool(namespace)
    ]
    
    # Create agent with semantic memory
    agent = create_react_agent(
        llm,
        tools=semantic_tools, 
        store=store
    )
    
    return agent, store

def demonstrate_semantic_memory():
    """Demonstrate semantic memory storage and retrieval"""
    
    print("🧠 Setting up Semantic Memory Agent...")
    agent, store = setup_semantic_agent()
    
    print("\n📝 Testing Semantic Memory Storage...")
    
    # Test 1: Store user preferences and facts
    print("1. Storing user preferences and facts...")
    response1 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Remember that I'm Sarah, I'm vegetarian, and I prefer morning workouts at 6 AM"
        }]
    })
    print(f"Response: {response1['messages'][-1].content}")
    
    # Test 2: Store more user information
    print("\n2. Storing additional user information...")
    response2 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Remember that I work as a software developer and use Python. I love hiking on weekends."
        }]
    })
    print(f"Response: {response2['messages'][-1].content}")
    
    # Test 3: Retrieve stored preferences
    print("\n3. Retrieving workout preferences...")
    response3 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "What workout should I do?"
        }]
    })
    print(f"Response: {response3['messages'][-1].content}")
    
    # Test 4: Search for dietary preferences
    print("\n4. Searching for dietary preferences...")
    response4 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "What do you know about my diet?"
        }]
    })
    print(f"Response: {response4['messages'][-1].content}")
    
    # Test 5: Search for work-related information
    print("\n5. Searching for work information...")
    response5 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "What do you know about my work?"
        }]
    })
    print(f"Response: {response5['messages'][-1].content}")
    
    # Test 6: Test memory search functionality
    print("\n6. Testing memory search functionality...")
    response6 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Search my memories for information about my preferences"
        }]
    })
    print(f"Response: {response6['messages'][-1].content}")
    
    print("\n✅ Semantic Memory demonstration completed!")
    
    # Show what's stored in the namespace
    print("\n🔍 Current stored memories in namespace ('user_facts', 'user123'):")
    try:
        items = store.search(("user_facts", "user123"))
        for item in items:
            print(f"  - {item.value}")
    except Exception as e:
        print(f"  Could not retrieve stored items: {e}")

def test_direct_memory_operations():
    """Test direct memory operations to understand how the tools work"""
    
    print("\n🔧 Testing Direct Memory Operations...")
    agent, store = setup_semantic_agent()
    
    # Test direct memory storage
    print("1. Testing direct memory storage...")
    try:
        # Use the manage memory tool directly
        response = agent.invoke({
            "messages": [{
                "role": "user", 
                "content": "Use the manage memory tool to store: 'Sarah prefers dark mode for all applications'"
            }]
        })
        print(f"Response: {response['messages'][-1].content}")
    except Exception as e:
        print(f"Error in direct memory storage: {e}")
    
    # Test memory search
    print("\n2. Testing memory search...")
    try:
        response = agent.invoke({
            "messages": [{
                "role": "user", 
                "content": "Use the search memory tool to find information about Sarah's preferences"
            }]
        })
        print(f"Response: {response['messages'][-1].content}")
    except Exception as e:
        print(f"Error in memory search: {e}")

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    try:
        demonstrate_semantic_memory()
        test_direct_memory_operations()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have a valid OpenAI API key and sufficient credits.")
