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
# Global variables to store data temporarily for tools
GLOBAL_LYRICS_DATA = None
GLOBAL_AUDIO_DATA = None

def set_global_data(lyrics_data=None, audio_data=None):
    """Set global data for tools to use"""
    global GLOBAL_LYRICS_DATA, GLOBAL_AUDIO_DATA
    if lyrics_data is not None:
        GLOBAL_LYRICS_DATA = lyrics_data
    if audio_data is not None:
        GLOBAL_AUDIO_DATA = audio_data

@tool
def analyze_lyrics(lyrics_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyzes the lyrics to identify key themes, emotions, and moments.
    
    Args:
        lyrics_data: List of dictionaries containing timestamp and text for each line.
        
    Returns:
        A dictionary containing the analysis including themes, emotional arc, and key segments.
    """
    global GLOBAL_LYRICS_DATA
    
    # Use global data if not provided as parameter
    if not lyrics_data and GLOBAL_LYRICS_DATA:
        lyrics_data = GLOBAL_LYRICS_DATA
        
    if not lyrics_data:
        return {
            "error": "No lyrics data provided",
            "message": "Please provide lyrics data as a list of dictionaries with timestamp and text fields.",
            "example": [
                {"timestamp": "00:00:05", "text": "First line of lyrics"},
                {"timestamp": "00:00:10", "text": "Second line of lyrics"}
            ]
        }
    
    # Clean the lyrics text by removing timestamp markers
    for line in lyrics_data:
        if "-" in line["text"]:
            # Remove the end timestamp marker at the beginning of the text (e.g., "-00:25 ")
            line["text"] = line["text"].split(" ", 1)[1] if " " in line["text"] else line["text"]
    
    # Group lyrics into meaningful segments (verses, chorus, bridge, etc.)
    segments = []
    current_segment = {"lines": [], "start_time": lyrics_data[0]["timestamp"], "end_time": ""}
    segment_id = 1
    segment_types = ["intro", "verse", "pre-chorus", "chorus", "bridge", "outro"]
    current_segment_type_index = 0
    
    # Group every 4-8 lines as a segment (simplified approach)
    for i, line in enumerate(lyrics_data):
        current_segment["lines"].append(line)
        
        # Create a new segment every 4-8 lines (adjust as needed)
        if len(current_segment["lines"]) >= 4 and (len(current_segment["lines"]) >= 8 or i == len(lyrics_data) - 1 or i % 4 == 3):
            # Calculate segment start and end times
            current_segment["start_time"] = current_segment["lines"][0]["timestamp"]
            current_segment["end_time"] = current_segment["lines"][-1]["timestamp"]
            
            # Assign a segment type (rotating through segment_types)
            current_segment["segment_type"] = segment_types[current_segment_type_index % len(segment_types)]
            current_segment_type_index += 1
            
            # Combine all text in the segment
            current_segment["text"] = " ".join([line["text"] for line in current_segment["lines"]])
            current_segment["segment_id"] = segment_id
            
            # Add segment to the list
            segments.append(current_segment)
            
            # Start a new segment
            if i < len(lyrics_data) - 1:
                segment_id += 1
                current_segment = {"lines": [], "start_time": lyrics_data[i+1]["timestamp"], "end_time": ""}
    
    # Create a summary of the analysis
    analysis = {
        "segments": segments,
        "total_segments": len(segments),
        "summary": f"Analyzed {len(lyrics_data)} lyric lines and identified {len(segments)} segments."
    }
    
    return analysis


@tool
def analyze_audio(audio_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyzes the audio to identify tempo, mood, and key moments.
    
    Args:
        audio_data: Dictionary containing audio metadata including duration, tempo, etc.
        
    Returns:
        A dictionary containing the audio analysis including mood, tempo changes, and beat markers.
    """
    global GLOBAL_AUDIO_DATA
    
    # Use global data if not provided as parameter
    if not audio_data and GLOBAL_AUDIO_DATA:
        audio_data = GLOBAL_AUDIO_DATA
        
    if not audio_data:
        return {
            "error": "No audio data provided",
            "message": "Please provide audio data as a dictionary with duration and metadata fields.",
            "example": {
                "duration": 180.5,
                "metadata": {"tempo": 120, "key": "C Major"}
            }
        }
    
    # Extract key audio metrics
    duration = audio_data.get("duration", 0)
    segments = audio_data.get("segments", [])
    
    # Identify volume peaks (simplified)
    volume_peaks = []
    if segments:
        # Find segments with higher than average volume
        volumes = [segment.get("volume_dBFS", -60) for segment in segments if "volume_dBFS" in segment]
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            volume_peaks = [
                segment for segment in segments 
                if segment.get("volume_dBFS", -60) > avg_volume
            ]
    
    # Create an analysis summary
    analysis = {
        "duration": duration,
        "volume_peaks": volume_peaks,
        "tempo": audio_data.get("metadata", {}).get("tempo", "unknown"),
        "key": audio_data.get("metadata", {}).get("key", "unknown"),
        "summary": f"Audio duration: {duration:.2f} seconds with {len(volume_peaks)} volume peaks identified."
    }
    
    return analysis


# Removed search_stock_imagery tool as requested - using DALL-E instead


@tool
def generate_image(prompt: str) -> str:
    """
    Generates an image using DALL-E based on the prompt.
    
    Args:
        prompt: Detailed description of the image to generate.
        
    Returns:
        URL to the generated image.
    """
    from openai import OpenAI
    import os
    
    # Initialize OpenAI client
    api_key = os.getenv("API_KEY")
    if not api_key:
        return "Error: API_KEY environment variable not set"
        
    client = OpenAI(api_key=api_key)
    
    try:
        # Generate image with DALL-E
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Get the image URL
        image_url = response.data[0].url
        return image_url
        
    except Exception as e:
        return f"Error generating image: {str(e)}"


@tool
def create_scene_layout(scene_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a visual scene layout based on the provided scene data.
    
    Args:
        scene_data: Dictionary containing scene details, timing, and visual elements.
        
    Returns:
        A structured scene layout specification.
    """
    if not scene_data:
        return {"error": "No scene data provided"}
    
    # Extract scene information
    segment_id = scene_data.get("segment_id", 0)
    segment_type = scene_data.get("segment_type", "verse")
    start_time = scene_data.get("start_time", "00:00:00")
    end_time = scene_data.get("end_time", "00:00:00")
    text = scene_data.get("text", "")
    image_url = scene_data.get("image_url", "")
    
    # Determine appropriate transitions based on segment type
    transitions = ["fade_in"]
    if segment_type == "chorus":
        transitions = ["dissolve"]
    elif segment_type == "bridge":
        transitions = ["slide_left"]
    
    # Create a properly formatted scene layout
    scene_layout = {
        "segment_id": segment_id,
        "start_time": start_time,
        "end_time": end_time,
        "segment_type": segment_type,
        "visual_elements": [image_url] if image_url else [],
        "transitions": transitions,
        "text_overlay": text
    }
    
    return scene_layout


@tool
def synchronize_visuals(visuals: List[Dict[str, Any]], lyrics_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synchronizes visual elements with lyrics based on timestamps.
    
    Args:
        visuals: List of dictionaries containing visual elements and their timing.
        lyrics_segments: List of dictionaries containing lyrics segments and their timestamps.
        
    Returns:
        A timeline of synchronized visual and audio elements.
    """
    if not visuals or not lyrics_segments:
        return {"error": "Missing visuals or lyrics segments data"}
    
    # Create a synchronized timeline
    timeline = {
        "segments": [],
        "total_duration": "00:00:00"
    }
    
    # Match visuals to lyrics segments based on segment_id
    for segment in lyrics_segments:
        segment_id = segment.get("segment_id", 0)
        
        # Find matching visual for this segment
        matching_visual = None
        for visual in visuals:
            if visual.get("segment_id", 0) == segment_id:
                matching_visual = visual
                break
        
        # Create a synchronized segment
        sync_segment = {
            "segment_id": segment_id,
            "segment_type": segment.get("segment_type", "verse"),
            "start_time": segment.get("start_time", "00:00:00"),
            "end_time": segment.get("end_time", "00:00:00"),
            "text": segment.get("text", ""),
            "image_url": matching_visual.get("visual_elements", [""])[0] if matching_visual else ""
        }
        
        timeline["segments"].append(sync_segment)
    
    # Set the total duration to the end time of the last segment
    if timeline["segments"]:
        timeline["total_duration"] = timeline["segments"][-1].get("end_time", "00:00:00")
    
    return timeline


@tool
def finalize_visualization(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalizes the visualization by compiling all elements into a ready-to-render format.
    
    Args:
        timeline: A timeline of synchronized visual and audio elements.
        
    Returns:
        Complete visualization data ready for rendering.
    """
    if not timeline or "segments" not in timeline:
        return {
            "error": "Invalid timeline data",
            "visualization_type": "none"
        }
    
    # Create a properly structured visualization data object
    visualization = {
        "visualization_type": "video",
        "scenes": [],
        "audio_sync_points": [],
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
    
    # Add scenes from the timeline
    for segment in timeline.get("segments", []):
        # Determine appropriate transitions based on segment type
        transitions = ["fade_in"]
        if segment.get("segment_type") == "chorus":
            transitions = ["dissolve"]
        elif segment.get("segment_type") == "bridge":
            transitions = ["slide_left"]
        
        scene = {
            "start_time": segment.get("start_time", "00:00:00"),
            "end_time": segment.get("end_time", "00:00:00"),
            "visual_elements": [segment.get("image_url", "")] if segment.get("image_url") else [],
            "transitions": transitions,
            "text_overlay": segment.get("text", "")
        }
        visualization["scenes"].append(scene)
        
        # Add sync point
        visualization["audio_sync_points"].append({
            "time": segment.get("start_time", "00:00:00"),
            "event": f"segment_{segment.get('segment_id', 0)}"
        })
    
    return visualization


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
    
    # Note: We'll set the global data in the agent_with_state function instead
    # Don't attempt to set it here since we don't have access to the data yet
    
    # Define the tools available to the agent
    tools = [
        analyze_lyrics,  # Keep original for documentation
        analyze_audio,   # Keep original for documentation
        generate_image,
        create_scene_layout,
        synchronize_visuals,
        finalize_visualization
    ]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a creative AI assistant specialized in creating visual representations of lyrics or transcripts.
        Your goal is to create a visually appealing and meaningful visualization that captures the essence of the provided lyrics and audio.
        
        IMPORTANT: You already have access to the lyrics_data and audio_data in your state. DO NOT ask the user
        to provide this data again. Instead, use the analyze_lyrics and analyze_audio tools with the data already
        provided to you. The data is automatically passed to these tools.
        
        Follow these steps:
        1. First, call the analyze_lyrics tool to understand the themes, emotions, and key moments
        2. Then, call the analyze_audio tool to identify tempo, mood changes, and important audio cues
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
    # We'll handle tool state injection in a custom node
    react_agent = create_react_agent(llm, tools)
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Set global data when agent is invoked
    def agent_with_state(state):
        # Set the global data for tools to access
        set_global_data(
            lyrics_data=state.get("lyrics_data"),
            audio_data=state.get("audio_data")
        )
        
        # Run the agent with state
        return react_agent.invoke(state)
    
    # Add the agent node to the graph
    workflow.add_node("agent", agent_with_state)
    
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
