from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

class State(TypedDict):
    n: int

def add(state: State):
    return {"n": state["n"] + 1}

g = StateGraph(State)
g.add_node("add", add)
g.add_edge(START, "add")
app = g.compile()

events = []
for event in app.stream_events({"n": 0}, version="v2"):
    events.append(event["event"])

print("Stream events:", events)
