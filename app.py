from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st

if "messages" not in st.session_state:
    st.session_state["messages"] = []


load_dotenv()

EMPLOYEE_CSV_PATH = "employees.csv"
employees_df = pd.read_csv(EMPLOYEE_CSV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY is missing. Please put it inside .env")

print("API key loaded successfully!")

def get_weather(city: str):
    """Get weather for a given city"""
    print("get_weather tool called")
    return {
        "city": city,
        "temp_c": 22,
        "condition": "Sunny"
    }

def connect_to_forcast_cloud_info():
    """Connect to Forecast Cloud Info to get live information from the server"""
    return {
        "status": "error",
        "message": "Unable to connect to Forecast Cloud Info"
    }

def get_employee_info(name: str) -> str:
    """Lookup employee information from the employees CSV."""
    print("\n[get_employee_info] Tool called")

    result = employees_df[employees_df['name'].str.contains(name, case=False, na=False)]

    if result.empty:
        return f"No employee found matching '{name}'."

    row = result.iloc[0]
    return (
        f"Name: {row['name']} | Position: {row['position']} | "
        f"Department: {row['department']} | Location: {row['location']}"
    )


gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
     
)

agent = create_agent(
    model=gemini_model,
    tools=[get_weather, connect_to_forcast_cloud_info,get_employee_info],
    system_prompt="You are a weather assistant agent, answer in 50 words.",
    checkpointer=InMemorySaver()
    
)

# streamlit UI
st.set_page_config(page_title="AI Agent Demo", page_icon="🤖")
st.title("🤖 AI Agent Workshop Demo")
st.write("Try asking about weather, cloud info, or employee data.")

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

thread_id = "workshop_session_1"
user_input = st.chat_input("Your message...")

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # --- Agent call ---
    response = agent.invoke(
        {"messages": st.session_state["messages"]},
        {"configurable": {"thread_id": "workshop_session_1"}}
    )

    last_msg = response["messages"][-1]

    # Extract clean text
    if isinstance(last_msg.content, list):
        clean_text = " ".join(
            chunk.get("text", "") for chunk in last_msg.content if isinstance(chunk, dict)
        )
    else:
        clean_text = last_msg.content

    # Save assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": clean_text})
    st.chat_message("assistant").write(clean_text)