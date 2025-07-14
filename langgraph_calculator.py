from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from typing import TypedDict, List
import operator

# Define our state
class State(TypedDict):
    messages: List[HumanMessage | AIMessage]

# Our ONE tool - Calculator
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression and return the result."""
    try:
        # Simple eval for basic math (in production, use a safer math parser)
        result = eval(expression)
        return f"The answer is: {result}"
    except:
        return "Sorry, I couldn't calculate that. Please check your math expression."

# Create LLM with tool binding
llm = ChatGroq(model_name="llama3-70b-8192")
llm_with_tools = llm.bind_tools([calculator])

# Node functions
def chatbot(state: State) -> State:
    """Main chatbot node that decides whether to use tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

def should_continue(state: State) -> str:
    """Decide whether to continue to tools or end."""
    last_message = state["messages"][-1]
    
    # If the LLM called a tool, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, we're done
    return END

# Create the graph
def create_smart_calculator():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode([calculator]))
    
    # Define the flow
    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges(
        "chatbot", 
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "chatbot")
    
    return workflow.compile()

# Run it!
def run_smart_calculator():
    app = create_smart_calculator()
    
    print("🧮 Smart Calculator with LLM")
    print("Ask me math questions and I'll calculate and explain!")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
        
        # Run the workflow
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        
        # Get the final AI response
        final_message = result["messages"][-1]
        print(f"🧮 Calculator: {final_message.content}\n")

if __name__ == "__main__":
   
    run_smart_calculator()