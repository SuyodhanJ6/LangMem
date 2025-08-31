# LangMem Project

A comprehensive demonstration of OpenAI integration with LangGraph storage and three types of memory management capabilities.

## 🧠 Memory Types Demonstrated

### 1. **Semantic Memory** (`semantic_memory.py`)
- **Purpose**: Stores user preferences, facts, and information
- **Example**: "User Sarah is vegetarian", "Prefers 6 AM workouts"
- **Use Case**: Personalization and user profiling

### 2. **Episodic Memory** (`episodic_memory.py`)
- **Purpose**: Stores successful interaction patterns and experiences
- **Example**: "Code examples worked well for John", "Step-by-step plans worked for Sarah"
- **Use Case**: Learning from past interactions to improve future responses

### 3. **Procedural Memory** (`procedural_memory.py`)
- **Purpose**: Updates core system behavior and response patterns
- **Example**: "Always sign emails with 'Best regards, William'", "Offer video call options"
- **Use Case**: Continuous system improvement and behavior adaptation

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up your OpenAI API key:**
   ```bash
   # Follow instructions in env_setup.txt
   # Or manually create .env file with your API key
   ```

3. **Run any of the memory demonstrations:**
   ```bash
   # Semantic Memory - User preferences & facts
   python semantic_memory.py
   
   # Episodic Memory - Past conversations & patterns
   python episodic_memory.py
   
   # Procedural Memory - System behavior updates
   python procedural_memory.py
   
   # Original simple example
   python hello.py
   ```

## 📁 Script Descriptions

### `semantic_memory.py`
Demonstrates how an agent can:
- Store user preferences (dietary restrictions, workout times)
- Remember personal information (work, hobbies)
- Retrieve specific facts when asked
- Maintain persistent user profiles

### `episodic_memory.py`
Shows how an agent learns from:
- Successful interaction patterns
- User feedback and responses
- What works well for different users
- Continuous improvement over time

### `procedural_memory.py`
Illustrates how an agent can:
- Update its core behavior based on feedback
- Optimize prompts using the `create_prompt_optimizer`
- Evolve response patterns over multiple iterations
- Maintain consistent behavior improvements

## 🔧 Technical Features

- **LangGraph Storage**: Uses `InMemoryStore` with vector search
- **LangMem Tools**: Memory management and search capabilities
- **OpenAI Integration**: GPT-4o-mini for natural language processing
- **Prompt Optimization**: Automatic behavior improvement
- **Namespace Management**: Organized memory storage by type and user

## 📊 Example Outputs

### Semantic Memory
```
🧠 Setting up Semantic Memory Agent...
📝 Testing Semantic Memory Storage...
1. Storing user preferences and facts...
Response: I've stored that you're Sarah, you're vegetarian, and prefer 6 AM morning workouts.
```

### Episodic Memory
```
📚 Setting up Episodic Memory Agent...
📝 Testing Episodic Memory Storage...
1. Storing successful Python explanation pattern...
Response: I've stored that providing code examples with explanations works well for Python questions.
```

### Procedural Memory
```
⚙️ Setting up Procedural Memory Agent...
1. Storing initial email agent instructions...
   Initial instructions stored: 'Write professional emails.'
4. Using prompt optimizer to improve behavior...
   Original prompt: Write professional emails.
   Optimized prompt: Write professional emails. Always sign 'Best regards, William' and offer Zoom/Google Meet options for meetings.
```

## 🎯 Use Cases

- **Customer Service**: Remember user preferences and past interactions
- **Educational AI**: Learn from successful teaching methods
- **Business Automation**: Continuously improve email and communication patterns
- **Personal Assistants**: Build long-term user relationships and preferences

## 📋 Requirements

- Python 3.11+
- OpenAI API key with sufficient credits
- Internet connection for API calls

## 🔗 Dependencies

- `langgraph`: Graph-based agent workflows
- `langchain`: LLM integration framework
- `langchain-openai`: OpenAI model integration
- `langmem`: Memory management tools
- `python-dotenv`: Environment variable management
