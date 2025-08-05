import tkinter as tk
from tkinter import scrolledtext

import uuid

from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
#from langchain_core.messages import ToolMessage, ToolCall
import requests
from langgraph.config import get_store

#from langgraph.types import Command, interrupt

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import BaseMessage

from dotenv import load_dotenv
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import io
import contextlib

from datetime import datetime

from langgraph.types import interrupt, Command

from langchain_core.messages import HumanMessage
import threading

################################################################

load_dotenv()

llm = init_chat_model(model="anthropic:claude-3-5-haiku-latest")
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

DB_URI = "postgresql://postgres:password123@localhost:5432/postgres?sslmode=disable"

# Create a context manager instance
store_ctx = PostgresStore.from_conn_string(DB_URI)
checkpointer_ctx = PostgresSaver.from_conn_string(DB_URI)

# Enter the context manually
store = store_ctx.__enter__()
checkpointer = checkpointer_ctx.__enter__()

################################################################

## GET TRAVEL TIME AND DISTANCE USING GOOGLE MAPS
@tool
def get_travel_time_and_distance_using_google_maps(origin: str, destination: str, mode: str) -> str:
    """
    Get estimated travel time and distance between two locations using Google Maps.
    mode can be 'driving', 'walking', 'transit', 'Metro'
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    endpoint = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "key": api_key
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "OK":
            return f"Error from Google Maps API: {data['status']}"

        route = data["routes"][0]["legs"][0]
        duration = route["duration"]["text"]
        distance = route["distance"]["text"]
        start_address = route["start_address"]
        end_address = route["end_address"]

        return (
            f"From: {start_address}\n"
            f"To: {end_address}\n"
            f"Distance: {distance}\n"
            f"Estimated Time: {duration}\n"
            f"Mode: {mode.capitalize()}"
        )
    except Exception as e:
        return f"Failed to fetch directions: {e}"

## GET PHONE NUMBER USING GOOGLE PLACES
@tool
def get_phone_number_from_google_places(query: str) -> str:
    """
    Given a place name or description, returns the formatted phone number using Google Places API.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return "Error: GOOGLE_MAPS_API_KEY not set."

    # Step 1: Get place_id
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    search_params = {"query": query, "key": api_key}
    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()
    
    if not search_data.get("results"):
        return f"No results found for: {query}"
    
    place_id = search_data["results"][0]["place_id"]

    # Step 2: Get phone number
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    details_params = {
        "place_id": place_id,
        "fields": "formatted_phone_number",
        "key": api_key
    }
    details_response = requests.get(details_url, params=details_params)
    details_data = details_response.json()
    
    return details_data.get("result", {}).get("formatted_phone_number", f"No phone number found for: {query}")

## SEND EMAIL
@tool
def send_email(query: str) -> str:
    """Send email"""
    # Set up your email parameters
    sender_email = 'skyelectric20@gmail.com'
    receiver_email = 'skyelectric20@gmail.com'
    #subject = coin_name + '- MOON TIME !!!!!!'
    subject = 'AI agent travel plan'
    body = query

    # Create a MIMEMultipart message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    # Attach the body of the email
    message.attach(MIMEText(body, 'plain'))

    # SMTP server configuration (using Gmail)
    smtp_server = 'smtp.gmail.com'
    smtp_port = 465

    # Your app-specific password (or regular password if using less secure apps, but not recommended)
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    # Send the email
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

## STORE USER PREFERENCES PERMANENTLY IN DATABASE
@tool
def update_preferences_in_memory(config: RunnableConfig, query: str) -> str:
    """update personal details and preferences in memory."""

    store = get_store()
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # Directly store the query passed by the model as memory
    print("********** storing memory **********")
    store.put(namespace, str(uuid.uuid4()), {"data": query})

    return "Saved your preferences."

## GET MEETING DETAILS
@tool
def get_google_calendar_events() -> str:
    """Fetches upcoming events from the user's Google Calendar."""
    try:
        response = requests.get("http://localhost:8000/events/next_week")
        response.raise_for_status()
        events = response.json().get("next_week_events", [])

        if not events:
            return "No upcoming events found for the next week."

        def format_datetime(dt_raw):
            if isinstance(dt_raw, dict):
                dt_str = dt_raw.get("dateTime") or dt_raw.get("date")
            else:
                dt_str = dt_raw
            if dt_str:
                return datetime.fromisoformat(dt_str).strftime("%A, %Y-%m-%d %I:%M %p")
            return "Unknown Time"

        formatted_events = []
        for i, e in enumerate(events, 1):
            formatted_events.append(
                f"📅 Event {i}\n"
                f"🕒 Start:    {format_datetime(e['start'])}\n"
                f"🕓 End:      {format_datetime(e['end'])}\n"
                f"📝 Summary:  {e['summary']}\n"
                f"📍 Location: {e['location']}\n"
                "-----------------------------------------"
            )
        return "\n".join(formatted_events)

    except Exception as e:
        return f"❌ Error fetching events: {e}"

## FETCH MY PREFERENCES FROM DATABASE
@tool
def fetch_my_memories(config: RunnableConfig) -> str:
    """Fetch preferences stored in the user's memory store."""
    store = get_store()
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query="*")
    if not memories:
        return "No memories found."

    # Extract values
    memory_texts = [item.value["data"] for item in memories if "data" in item.value]
    return "\n".join(memory_texts) or "No memory data available."

## HUMAN IN THE LOOP - TO REVIEW , EDIT AI RESPONSES
@tool
def manual_changes(query: str) -> str:
    """Pause and request manual input."""
    print("\n\n>>> manual_changes()")
    return interrupt({"query": query})


################################################################

tools = [get_google_calendar_events, fetch_my_memories, update_preferences_in_memory, TavilySearchResults(max_results=3), send_email, get_travel_time_and_distance_using_google_maps, get_phone_number_from_google_places, manual_changes,]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

################################################################

def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore,):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

################################################################

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("call_model", call_model)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "call_model",
    tools_condition,
)
graph_builder.add_edge("tools", "call_model")
graph_builder.add_edge(START, "call_model")
    
graph = graph_builder.compile(
    checkpointer=checkpointer,
    store=store,
)

################################################################

config = {
    "configurable": {
        "thread_id": "10",
        "user_id": "vivek1",
    }
}

################################################################
message_history = []

def send_message():
    print("\n\n>>> send_message()")
    user_input = entry.get()
    if not user_input.strip():
        return
    chat_area.insert(tk.END, f"You: {user_input}\n", "human")
    chat_area.insert(tk.END, f"\n")
    entry.delete(0, tk.END)
    response = graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values")

    response_list = list(response)
    print(response_list)

    def stream_handler():
        for event in response_list:
            #print("\n\n>>> event1: ", event)
            if "__interrupt__" in event:
                print(">>> event.get(__type__) == interrupt")
                query_text = "Please provide input:"
                root.after(0, lambda: open_manual_update_window(query_text))
                return
            if "messages" in event:
                print("\n\n>>> event2: ", event["messages"][-1])
                msg_type = type(event["messages"][-1]).__name__
                print(msg_type)
                print(">>> event.get(__type__) == messages")

                if msg_type != "HumanMessage":
                    print(">>> printing")
                    # Capture the output of pretty_print()
                    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                        event["messages"][-1].pretty_print()
                        output = buf.getvalue()

                    # Insert into chat_area
                    chat_area.insert(tk.END, output + "\n\n")
                    chat_area.see(tk.END)
        print("xxx")
    threading.Thread(target=stream_handler, daemon=True).start()

def resume_with_command(text):
    print("\n\n>>> resume_with_command() and input: ",text)
    response = graph.stream(
        Command(resume={"data": text}), 
        config,
        stream_mode="values",
    )

    response_list = list(response)
    if not response_list:
        print("⚠️ No response received.")
        return

    last_event = response_list[-1]
    print("*** Last Event:", last_event)

    if "messages" in last_event:
        # for msg in last_event["messages"]:
        #     msg_type = type(msg).__name__
        #     print(f"[{msg_type}]\n")

        #     # Optionally skip HumanMessage
        #     if msg_type == "HumanMessage":
        #         continue
            
        #     # Pretty print each message
        #     with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        #         msg.pretty_print()
        #         output = buf.getvalue()

        #     chat_area.insert(tk.END, output + "\n\n", "ai")
        #     chat_area.see(tk.END)
        
        print("\n\n>>> event2: ", last_event["messages"][-1])
        msg_type = type(last_event["messages"][-1]).__name__
        print(msg_type)
        print(">>> event.get(__type__) == messages")

        if msg_type != "HumanMessage":
            print(">>> printing")
            # Capture the output of pretty_print()
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                last_event["messages"][-1].pretty_print()
                output = buf.getvalue()

            # Insert into chat_area
            chat_area.insert(tk.END, output + "\n\n")
            chat_area.see(tk.END)

def open_manual_update_window(query_text):
    print("\n\n>>> open_manual_update_window()")
    popup = tk.Toplevel(root)
    popup.title("Manual Update Required")

    # label = tk.Label(popup, text=query_text)
    # label.pack(pady=10)
    # text_box = tk.Text(popup, height=8, width=60)
    # text_box.pack(padx=10)

    popup.geometry("800x600")  # Optional starting size

    # Configure grid for resizing
    popup.grid_rowconfigure(1, weight=1)   # Text box expands vertically
    popup.grid_columnconfigure(0, weight=1)  # Text box expands horizontally

    label = tk.Label(popup, text=query_text)
    label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

    text_box = tk.Text(popup)
    text_box.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    def submit():
        print(">>> submitted manual text")
        updated_text = text_box.get("1.0", tk.END).strip()
        popup.destroy()
        resume_with_command(updated_text)

    # tk.Button(popup, text="Submit", command=submit).pack(pady=10)

    submit_button = tk.Button(popup, text="Submit", command=submit)
    submit_button.grid(row=2, column=0, pady=(0, 10))

    popup.transient(root)  # Optional: makes popup appear as a child
    popup.lift()           # Bring to front
    # popup.grab_set()     # grabs all input focus, preventing interaction (including copy, scroll, or text selection) with any other window — including the root window — until the popup is closed
    popup.focus_force()

def end_chat():
    thread_id = config["configurable"]["thread_id"]
    try:
        # This will remove all checkpoints associated with this thread_id
        checkpointer.delete_thread(thread_id)
        print(f"✅ Deleted checkpoints for thread_id: {thread_id}")
    except Exception as e:
        print(f"❌ Error deleting checkpoints for thread_id {thread_id}: {e}")

    root.destroy()

################################################################

# Create main window
root = tk.Tk()
root.title("AI Chat")
root.geometry("800x600")  # Initial size

# Configure grid for full expansion
root.grid_rowconfigure(0, weight=1)    # chat area expands vertically
root.grid_columnconfigure(0, weight=1) # chat area expands horizontally

# Chat display area
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_area.tag_config("human", background="lightyellow")
chat_area.tag_config("ai", foreground="blue")
chat_area.tag_config("tool_call", foreground="green")
chat_area.tag_config("tool_response", foreground="darkgreen")
chat_area.tag_config("other", foreground="gray")
chat_area.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

# Initial greeting
chat_area.insert(tk.END, "AI: Hi. How can I help you today?\n\n\n", "ai")

# Entry field
entry = tk.Entry(root)
entry.grid(row=1, column=0, sticky="ew", padx=(10, 0), pady=(0, 10))
root.grid_columnconfigure(0, weight=3)  # Allow entry field to grow

# Send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, sticky="ew", padx=(5, 5), pady=(0, 10))
root.grid_columnconfigure(1, weight=1)

# End Chat button
end_button = tk.Button(root, text="End Chat", command=end_chat, fg="white", bg="red")
end_button.grid(row=1, column=2, sticky="ew", padx=(5, 10), pady=(0, 10))
root.grid_columnconfigure(2, weight=1)

# Run the GUI loop
root.mainloop()