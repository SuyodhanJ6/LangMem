#!/usr/bin/env python3
"""
PROCEDURAL MEMORY EXAMPLE - System Behavior Updates
Stores: "How I should behave and respond"
"""

import os
from dotenv import load_dotenv
from langmem import create_manage_memory_tool, create_search_memory_tool, create_prompt_optimizer
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def setup_procedural_agent():
    """Set up agent with procedural memory capabilities"""
    
    # Initialize OpenAI model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Setup procedural memory store
    store = InMemoryStore(
        index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
    )
    
    # Create namespace for instructions
    namespace = ("instructions", "email_agent")
    
    # Create procedural memory tools
    procedural_tools = [
        create_manage_memory_tool(namespace),
        create_search_memory_tool(namespace)
    ]
    
    return llm, store, procedural_tools

def create_email_prompt_function(store):
    """Create a function that retrieves and applies stored instructions"""
    
    def email_prompt(state):
        try:
            # Get current instructions from store
            item = store.get(("instructions",), key="email_agent")
            if item:
                instructions = item.value["prompt"]
                sys_prompt = {"role": "system", "content": f"Instructions: {instructions}"}
                return [sys_prompt] + state['messages']
            else:
                # Default instructions if none stored
                default_prompt = "Write professional emails."
                sys_prompt = {"role": "system", "content": f"Instructions: {default_prompt}"}
                return [sys_prompt] + state['messages']
        except Exception as e:
            print(f"Error retrieving instructions: {e}")
            # Fallback to default
            default_prompt = "Write professional emails."
            sys_prompt = {"role": "system", "content": f"Instructions: {default_prompt}"}
            return [sys_prompt] + state['messages']
    
    return email_prompt

def draft_email(to: str, subject: str, body: str):
    """Tool for drafting emails"""
    return f"Email drafted successfully to {to} with subject: {subject}"

def demonstrate_procedural_memory():
    """Demonstrate procedural memory for email agent behavior"""
    
    print("⚙️ Setting up Procedural Memory Agent...")
    llm, store, procedural_tools = setup_procedural_agent()
    
    # Store initial system prompt
    print("\n1. Storing initial email agent instructions...")
    store.put(
        ("instructions",), 
        key="email_agent", 
        value={"prompt": "Write professional emails."}
    )
    print("   Initial instructions stored: 'Write professional emails.'")
    
    # Create agent with procedural memory
    email_prompt_func = create_email_prompt_function(store)
    email_agent = create_react_agent(
        llm,
        prompt=email_prompt_func, 
        tools=[draft_email], 
        store=store
    )
    
    print("\n📝 Testing Initial Email Behavior...")
    
    # Test 1: Initial email behavior
    print("2. Testing initial email behavior...")
    response1 = email_agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Draft email to john@company.com about meeting tomorrow at 2pm"
        }]
    })
    print(f"Response: {response1['messages'][-1].content}")
    
    # Test 2: User feedback and behavior update
    print("\n3. User provides feedback for behavior improvement...")
    feedback = "Always sign emails 'Best regards, William' and offer video call options for meetings"
    
    # Use prompt optimizer to improve behavior
    print("4. Using prompt optimizer to improve behavior...")
    optimizer = create_prompt_optimizer(llm)
    
    # Get current prompt
    current_item = store.get(("instructions",), key="email_agent")
    current_prompt = current_item.value["prompt"] if current_item else "Write professional emails."
    
    # Create a simple trajectory for optimization
    conversation_messages = [
        {"role": "user", "content": "Draft email to john@company.com about meeting tomorrow at 2pm"},
        {"role": "assistant", "content": "Subject: Meeting Tomorrow at 2pm\n\nHi John,\n\nI'd like to schedule a meeting tomorrow at 2pm to discuss the project updates.\n\nBest regards,\nWilliam"},
        {"role": "user", "content": "Always sign emails 'Best regards, William' and offer video call options for meetings"}
    ]
    
    # Optimize the prompt
    try:
        optimized_result = optimizer.invoke({
            "prompt": current_prompt, 
            "trajectories": [(conversation_messages, feedback)]
        })
        
        # Extract the optimized prompt
        if hasattr(optimized_result, 'content'):
            optimized_prompt = optimized_result.content
        elif isinstance(optimized_result, dict) and 'content' in optimized_result:
            optimized_prompt = optimized_result['content']
        else:
            # Fallback if optimization doesn't work as expected
            optimized_prompt = f"{current_prompt} Always sign 'Best regards, William' and offer Zoom/Google Meet options for meetings."
        
        print(f"   Original prompt: {current_prompt}")
        print(f"   Optimized prompt: {optimized_prompt}")
        
        # Store updated behavior
        store.put(
            ("instructions",), 
            key="email_agent", 
            value={"prompt": optimized_prompt}
        )
        print("   ✅ Updated instructions stored!")
        
    except Exception as e:
        print(f"   ⚠️ Optimization failed, using fallback: {e}")
        # Fallback optimization
        optimized_prompt = f"{current_prompt} Always sign 'Best regards, William' and offer Zoom/Google Meet options for meetings."
        store.put(
            ("instructions",), 
            key="email_agent", 
            value={"prompt": optimized_prompt}
        )
        print("   ✅ Fallback instructions stored!")
    
    # Test 3: New behavior in action
    print("\n5. Testing updated email behavior...")
    response2 = email_agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Draft email to sarah@company.com about weekly team sync"
        }]
    })
    print(f"Response: {response2['messages'][-1].content}")
    
    # Test 4: Verify behavior consistency
    print("\n6. Verifying behavior consistency...")
    response3 = email_agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Send a quick email to mike@company.com about the project deadline"
        }]
    })
    print(f"Response: {response3['messages'][-1].content}")
    
    print("\n✅ Procedural Memory demonstration completed!")
    
    # Show what's stored in the namespace
    print("\n🔍 Current stored instructions in namespace ('instructions', 'email_agent'):")
    try:
        items = store.search(("instructions",))
        for item in items:
            print(f"  - {item.value}")
    except Exception as e:
        print(f"  Could not retrieve stored items: {e}")

def demonstrate_behavior_evolution():
    """Demonstrate how behavior evolves over multiple iterations"""
    
    print("\n🔄 Demonstrating Behavior Evolution...")
    llm, store, procedural_tools = setup_procedural_agent()
    
    # Start with basic instructions
    store.put(
        ("instructions",), 
        key="email_agent", 
        value={"prompt": "Write professional emails."}
    )
    
    # Simulate multiple feedback iterations
    feedback_iterations = [
        "Always use formal language",
        "Include meeting agenda in meeting emails",
        "Use bullet points for action items",
        "Always confirm receipt of important documents"
    ]
    
    current_prompt = "Write professional emails."
    
    for i, feedback in enumerate(feedback_iterations, 1):
        print(f"\nIteration {i}: {feedback}")
        
        # Update prompt with new feedback
        current_prompt = f"{current_prompt} {feedback}"
        
        # Store updated behavior
        store.put(
            ("instructions",), 
            key="email_agent", 
            value={"prompt": current_prompt}
        )
        
        print(f"   Updated prompt: {current_prompt}")
    
    print("\n✅ Behavior evolution demonstration completed!")

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    try:
        demonstrate_procedural_memory()
        demonstrate_behavior_evolution()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have a valid OpenAI API key and sufficient credits.")
