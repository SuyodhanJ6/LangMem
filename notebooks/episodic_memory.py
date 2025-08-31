#!/usr/bin/env python3
"""
EPISODIC MEMORY EXAMPLE - Past Conversations & Experiences  
Stores: "What happened and what worked well"
"""

import os
from dotenv import load_dotenv
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def setup_episodic_agent():
    """Set up agent with episodic memory capabilities"""
    
    # Initialize OpenAI model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Setup episodic memory store
    store = InMemoryStore(
        index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
    )
    
    # Create namespace for episodes
    namespace = ("episodes", "user123")
    
    # Create episodic memory tools
    episodic_tools = [
        create_manage_memory_tool(namespace),
        create_search_memory_tool(namespace)
    ]
    
    # Create agent with episodic memory
    agent = create_react_agent(
        llm,
        tools=episodic_tools, 
        store=store
    )
    
    return agent, store

def demonstrate_episodic_memory():
    """Demonstrate episodic memory storage and retrieval"""
    
    print("📚 Setting up Episodic Memory Agent...")
    agent, store = setup_episodic_agent()
    
    print("\n📝 Testing Episodic Memory Storage...")
    
    # Test 1: Store successful interaction pattern for Python explanations
    print("1. Storing successful Python explanation pattern...")
    response1 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "When John asked about Python loops, providing code examples with explanations worked well"
        }]
    })
    print(f"Response: {response1['messages'][-1].content}")
    
    # Test 2: Store successful interaction pattern for workout plans
    print("\n2. Storing successful workout plan pattern...")
    response2 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Sarah responded positively to step-by-step workout plans with rest days"
        }]
    })
    print(f"Response: {response2['messages'][-1].content}")
    
    # Test 3: Store successful interaction pattern for meeting requests
    print("\n3. Storing successful meeting request pattern...")
    response3 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Meeting requests work better when I offer Zoom/Google Meet options"
        }]
    })
    print(f"Response: {response3['messages'][-1].content}")
    
    # Test 4: Search for Python explanation patterns
    print("\n4. Searching for Python explanation patterns...")
    response4 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Can you explain Python functions?"
        }]
    })
    print(f"Response: {response4['messages'][-1].content}")
    
    # Test 5: Search for workout plan patterns
    print("\n5. Searching for workout plan patterns...")
    response5 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "I need a new workout routine"
        }]
    })
    print(f"Response: {response5['messages'][-1].content}")
    
    # Test 6: Search for meeting request patterns
    print("\n6. Searching for meeting request patterns...")
    response6 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "I need to schedule a meeting"
        }]
    })
    print(f"Response: {response6['messages'][-1].content}")
    
    print("\n✅ Episodic Memory demonstration completed!")
    
    # Show what's stored in the namespace
    print("\n🔍 Current stored episodes in namespace ('episodes', 'user123'):")
    try:
        items = store.search(("episodes", "user123"))
        for item in items:
            print(f"  - {item.value}")
    except Exception as e:
        print(f"  Could not retrieve stored items: {e}")

def simulate_learning_scenario():
    """Simulate a learning scenario where the agent improves over time"""
    
    print("\n🎯 Simulating Learning Scenario...")
    agent, store = setup_episodic_agent()
    
    # Initial interaction - agent doesn't know the best approach
    print("1. Initial interaction (agent learning)...")
    response1 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Explain machine learning to me"
        }]
    })
    print(f"Response: {response1['messages'][-1].content}")
    
    # Store what worked well
    print("\n2. Storing successful approach...")
    response2 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "That explanation with real-world examples was perfect! Store that approach."
        }]
    })
    print(f"Response: {response2['messages'][-1].content}")
    
    # Later, similar request - agent should use learned approach
    print("\n3. Later similar request (agent applying learned approach)...")
    response3 = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Can you explain deep learning?"
        }]
    })
    print(f"Response: {response3['messages'][-1].content}")
    
    print("\n✅ Learning scenario completed!")

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    try:
        demonstrate_episodic_memory()
        simulate_learning_scenario()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have a valid OpenAI API key and sufficient credits.")
