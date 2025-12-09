from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
import pandas as pd
# import streamlit as st

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

thread_id = "workshop_session_1"

print("\n🟦 Weather Assistant Agent — Conversation Started")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Ending conversation.")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": "1"}},
    )

    print("Agent:", response["messages"][-1].content)


