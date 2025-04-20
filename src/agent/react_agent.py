import os
from typing import Dict, List, Any, Literal, TypedDict, Union, Annotated
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent


# Tool definitions for the agent
@tool
def analyze_lyrics(lyrics_data: List[Dict[str, Any]]) -> str:
    """
    Analyzes the lyrics to identify key themes, emotions, and moments.
    
    Args:
        lyrics_data: List of dictionaries containing timestamp and text for each line.
        
    Returns:
        A summary of the analysis including themes, emotional arc, and key moments.
    """
    return "Analysis of lyrics complete. Found key themes and emotional moments."


@tool
def analyze_audio(audio_data: Dict[str, Any]) -> str:
    """
    Analyzes the audio to identify tempo, mood, and key moments.
    
    Args:
        audio_data: Dictionary containing audio metadata including duration, tempo, etc.
        
    Returns:
        A summary of the audio analysis including mood, tempo changes, and beat markers.
    """
    return "Audio analysis complete. Identified tempo and mood characteristics."


@tool
def search_stock_imagery(query: str) -> str:
    """
    Searches for relevant stock imagery based on the query.
    
    Args:
        query: Search query specifying image requirements.
        
    Returns:
        URLs or references to relevant stock images.
    """
    return f"Found stock images matching query: {query}"


@tool
def generate_image(prompt: str) -> str:
    """
    Generates an image using AI based on the prompt.
    
    Args:
        prompt: Detailed description of the image to generate.
        
    Returns:
        Path or URL to the generated image.
    """
    return f"Generated image based on prompt: {prompt}"


@tool
def create_scene_layout(scene_data: Dict[str, Any]) -> str:
    """
    Creates a visual scene layout based on the provided scene data.
    
    Args:
        scene_data: Dictionary containing scene details, timing, and visual elements.
        
    Returns:
        A structured scene layout specification.
    """
    return f"Created scene layout with {len(scene_data)} elements."


@tool
def synchronize_visuals(visuals: List[Dict[str, Any]], lyrics: List[Dict[str, Any]]) -> str:
    """
    Synchronizes visual elements with lyrics based on timestamps.
    
    Args:
        visuals: List of dictionaries containing visual elements and their timing.
        lyrics: List of dictionaries containing lyrics lines and their timestamps.
        
    Returns:
        A timeline of synchronized visual and audio elements.
    """
    return f"Synchronized {len(visuals)} visual elements with {len(lyrics)} lyric lines."


@tool
def finalize_visualization(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalizes the visualization by compiling all elements into a ready-to-render format.
    
    Args:
        timeline: A timeline of synchronized visual and audio elements.
        
    Returns:
        Complete visualization data ready for rendering.
    """
    return {
        "visualization_type": "video",
        "scenes": [
            {
                "start_time": "00:00:00",
                "end_time": "00:00:05",
                "visual_elements": ["intro_image.jpg"],
                "transitions": ["fade_in"],
                "text_overlay": "Song Title"
            },
            # More scenes would be generated dynamically based on actual content
        ],
        "audio_sync_points": [
            {"time": "00:00:00", "event": "start"},
            {"time": "00:00:05", "event": "first_verse"},
        ],
        "style": {
            "color_palette": ["#3A86FF", "#FF006E", "#FB5607"],
            "typography": {"font": "Montserrat", "main_color": "#FFFFFF"},
            "transitions": ["fade", "dissolve", "slide"],
        },
        "output_format": "mp4",
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0"
        }
    }


# Define input state type
class AgentState(TypedDict):
    """State for the lyrics visualizer agent."""
    lyrics_data: List[Dict[str, Any]]
    audio_data: Union[Dict[str, Any], None]
    user_requirements: str
    messages: List[Union[HumanMessage, AIMessage]]
    
    
# Create the ReAct agent using LangGraph
def create_lyrics_visualizer_agent():
    # Get OpenAI API key
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable not set. Please set your OpenAI API key.")
    
    # Create the LLM
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.7, 
        api_key=api_key
    )
    
    # Define the tools available to the agent
    tools = [
        analyze_lyrics,
        analyze_audio,
        search_stock_imagery,
        generate_image,
        create_scene_layout,
        synchronize_visuals,
        finalize_visualization
    ]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a creative AI assistant specialized in creating visual representations of lyrics or transcripts.
        Your goal is to create a visually appealing and meaningful visualization that captures the essence of the provided lyrics and audio.
        
        Follow these steps:
        1. Analyze the lyrics to understand the themes, emotions, and key moments
        2. If audio data is available, analyze it to identify tempo, mood changes, and important audio cues
        3. Based on user requirements, decide on the most appropriate visualization approach
        4. Create a plan for visual elements that align with the lyrics and audio
        5. Generate or source appropriate visual content
        6. Synchronize visual elements with lyrics based on timestamps
        7. Finalize the visualization into a cohesive, time-synchronized experience
        
        Consider the spirit of both lyrics and sound when creating visuals, paying particular attention to timestamps for synchronization.
        Use the available tools to complete these tasks step by step.
        
        Remember that the final output should:
        - Match the emotional tone of the lyrics/audio
        - Time visual elements precisely with the audio
        - Reflect the specific visualization requirements from the user
        - Form a cohesive visual experience that enhances the meaning of the content
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Create the ReAct agent
    # Using the API that accepts only two positional arguments
    react_agent = create_react_agent(llm, tools)
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add the ReAct agent node to the graph
    workflow.add_node("agent", react_agent)
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Modified condition to reduce the number of iterations
    # Agent will stop after fewer iterations or when it determines it has completed the task
    workflow.add_conditional_edges(
        "agent",
        lambda state: END if len(state.get("messages", [])) > 10 or \
                      (len(state.get("messages", [])) > 0 and \
                       state.get("messages", [])[-1].content.endswith("FINAL ANSWER:")) \
                   else "agent"
    )
    
    # Compile the graph into a runnable agent
    return workflow.compile()


# Example usage (for testing)
if __name__ == "__main__":
    # Sample data for testing
    sample_lyrics_data = [
        {"timestamp": "00:00:05", "text": "Verse 1: Starting on a journey"},
        {"timestamp": "00:00:10", "text": "Through the unknown wilderness"},
        {"timestamp": "00:00:15", "text": "Seeking truth and meaning"},
        {"timestamp": "00:00:20", "text": "In a world of emptiness"},
    ]
    
    sample_audio_data = {
        "duration": 180,  # seconds
        "tempo": 120,  # BPM
        "key": "C Major",
        "sections": [
            {"start": 0, "end": 20, "type": "intro"},
            {"start": 20, "end": 60, "type": "verse"},
            {"start": 60, "end": 90, "type": "chorus"},
        ]
    }
    
    sample_requirements = "Create a nature-themed visualization that reflects the journey metaphor in the lyrics. Use warm colors for the chorus sections."
    
    # Create the agent
    agent = create_lyrics_visualizer_agent()
    
    # Run the agent
    result = agent.invoke({
        "lyrics_data": sample_lyrics_data,
        "audio_data": sample_audio_data,
        "user_requirements": sample_requirements,
        "messages": [HumanMessage(content="Create a visualization for these lyrics and audio.")]
    })
    
    print(json.dumps(result, indent=2))
