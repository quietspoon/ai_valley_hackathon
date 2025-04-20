import os
from typing import Dict, List, Any, Literal, TypedDict, Union, Annotated
from datetime import datetime
import json
import tempfile

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
    
    # Group transcript into meaningful segments (topics, sections, etc.)
    segments = []
    current_segment = {"lines": [], "start_time": lyrics_data[0]["timestamp"], "end_time": ""}
    segment_id = 1
    segment_types = ["introduction", "main_point", "example", "anecdote", "discussion", "conclusion"]
    current_segment_type_index = 0
    
    # Group every 4-8 lines as a segment (simplified approach)
    for i, line in enumerate(lyrics_data):
        current_segment["lines"].append(line)
        
        # Create a new segment every 6-12 lines for podcast content (longer segments for discussion)
        if len(current_segment["lines"]) >= 6 and (len(current_segment["lines"]) >= 12 or i == len(lyrics_data) - 1 or i % 6 == 5):
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


# Tool calling sequence for visualization:
# 1. analyze_lyrics - analyze the lyric content
# 2. analyze_audio - analyze the audio content
# 3. create_scene_layout - create the initial scene layout based on analysis
# 4. synchronize_visuals - synchronize the visuals with lyrics and scene layout
# 5. finalize_visualization - finalize the visualization with the timeline from synchronize_visuals
# 6. stitch_music_video - create a complete music video with audio, visuals, and synchronized lyrics


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
def create_scene_layout(scene_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Creates a visual scene layout based on the provided scene data.
    
    Args:
        scene_data: Dictionary containing scene details, timing, and visual elements.
        
    Returns:
        A structured scene layout specification.
    """
    # Ensure scene_data is a valid dictionary
    if not scene_data or not isinstance(scene_data, dict):
        # Instead of returning an error, create a default scene layout
        scene_data = {
            "segment_id": 1,
            "segment_type": "verse",
            "start_time": "00:00:00",
            "end_time": "00:00:30",
            "text": "Default scene",
            "image_url": ""
        }
    
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
def synchronize_visuals(visuals: List[Dict[str, Any]] = None, lyrics_segments: List[Dict[str, Any]] = None, scene_layout: Union[Dict[str, Any], None] = None) -> Dict[str, Any]:
    """
    Synchronizes visual elements with lyrics based on timestamps.
    
    Args:
        visuals: List of dictionaries containing visual elements and their timing.
        lyrics_segments: List of dictionaries containing lyrics segments and their timestamps.
        scene_layout: Optional scene layout from create_scene_layout to use as a fallback.
        
    Returns:
        A timeline of synchronized visual and audio elements.
    """
    # Note: This function has been updated to work with the generate_image_for_segment function
    # Access global data if available when parameters are None
    global GLOBAL_LYRICS_DATA, GLOBAL_AUDIO_DATA
    
    # Check if we have a scene_layout to work with
    if scene_layout and isinstance(scene_layout, dict):
        # Extract data from scene_layout if it has the required fields
        segment_id = scene_layout.get("segment_id")
        start_time = scene_layout.get("start_time")
        end_time = scene_layout.get("end_time")
        segment_type = scene_layout.get("segment_type")
        visual_elements = scene_layout.get("visual_elements", [])
        text_overlay = scene_layout.get("text_overlay", "")
        
        if segment_id is not None and start_time and end_time:
            # Create a single visual from scene_layout
            if not visuals:
                visuals = [{
                    "segment_id": segment_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "visual_elements": visual_elements
                }]
            
            # Create a single lyrics segment from scene_layout if needed
            if not lyrics_segments:
                lyrics_segments = [{
                    "segment_id": segment_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "segment_type": segment_type,
                    "text": text_overlay
                }]
    
    # Provide default empty lists if parameters are None
    if visuals is None:
        visuals = []
    if lyrics_segments is None:
        # Try to use global lyrics data
        if GLOBAL_LYRICS_DATA:
            # Check if GLOBAL_LYRICS_DATA is a dictionary or a list
            if isinstance(GLOBAL_LYRICS_DATA, dict):
                # Extract segments from global lyrics data if it's a dictionary
                segments_data = GLOBAL_LYRICS_DATA.get("segments", [])
                if segments_data:
                    lyrics_segments = segments_data
            elif isinstance(GLOBAL_LYRICS_DATA, list):
                # If GLOBAL_LYRICS_DATA is already a list, use it directly
                lyrics_segments = GLOBAL_LYRICS_DATA
        else:
            lyrics_segments = []
    
    # If we still don't have any data, create default data based on audio duration
    if not visuals or not lyrics_segments:
        # Create a default segment based on audio duration if available
        if GLOBAL_AUDIO_DATA and "duration" in GLOBAL_AUDIO_DATA:
            duration = GLOBAL_AUDIO_DATA["duration"]
            end_time = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            
            # Create default lyrics segment
            if not lyrics_segments:
                default_segment = {
                    "segment_id": 1,
                    "start_time": "00:00",
                    "end_time": end_time,
                    "segment_type": "verse",
                    "text": "Default visualization"
                }
                lyrics_segments = [default_segment]
            
            # Create default visual
            if not visuals:
                default_visual = {
                    "segment_id": 1,
                    "start_time": "00:00",
                    "end_time": end_time,
                    "visual_elements": []
                }
                visuals = [default_visual]
    
    # Ensure lyrics_segments is a list if we got a single dictionary
    if isinstance(lyrics_segments, dict):
        lyrics_segments = [lyrics_segments]
    
    # Ensure visuals is a list if we got a single dictionary
    if isinstance(visuals, dict):
        visuals = [visuals]
    
    # If we still have missing data, return error
    if not visuals or not lyrics_segments:
        return {"error": "Missing visuals or lyrics segments data", "segments": [], "total_duration": "00:00:00"}
    
    # Create a synchronized timeline
    timeline = {
        "segments": [],
        "total_duration": "00:00:00"
    }
    
    # If we have lyrics_segments as a string, try to parse it
    if isinstance(lyrics_segments, str):
        try:
            lyrics_segments = json.loads(lyrics_segments)
        except json.JSONDecodeError:
            # Couldn't parse it as JSON
            lyrics_segments = []
            
    # Ensure we have at least one segment even if we have no data
    if not lyrics_segments:
        lyrics_segments = [{
            "segment_id": 1,
            "segment_type": "verse",
            "start_time": "00:00:00",
            "end_time": "00:00:30",
            "text": "Default segment"
        }]
    
    # Process all lyrics segments
    for segment in lyrics_segments:
        # Check if segment is a dictionary or something else
        if isinstance(segment, dict):
            segment_id = segment.get("segment_id", 0)
            segment_start = segment.get("start_time", "00:00:00")
            segment_end = segment.get("end_time", "00:00:00")
        else:
            # Handle non-dictionary segments as best we can
            segment_id = 0
            segment_start = "00:00:00"
            segment_end = "00:00:00"
            # Try to convert to dict if possible
            if hasattr(segment, "__dict__"):
                segment = segment.__dict__
        
        # Find matching visuals for this segment based on overlapping time ranges or segment_id
        matching_visuals = []
        for visual in visuals:
            # Check if visual is a dictionary
            if isinstance(visual, dict):
                visual_start = visual.get("start_time", "00:00:00")
                visual_end = visual.get("end_time", "00:00:00")
                visual_id = visual.get("segment_id", 0)
            else:
                # Handle non-dictionary visuals
                visual_start = "00:00:00"
                visual_end = "00:00:00"
                visual_id = 0
            
            # Check if visual matches by segment_id or overlapping time ranges
            if visual_id == segment_id or (visual_start <= segment_end and visual_end >= segment_start):
                matching_visuals.append(visual)
        
        # Create a synchronized segment
        sync_segment = {
            "segment_id": segment_id,
            "segment_type": "verse",  # Default value
            "start_time": segment_start,
            "end_time": segment_end,
            "text": "",  # Default value
            "image_url": ""
        }
        
        # Add additional properties if segment is a dictionary
        if isinstance(segment, dict):
            sync_segment["segment_type"] = segment.get("segment_type", "verse")
            sync_segment["text"] = segment.get("text", "")
        
        # Add image URL if available
        if matching_visuals:
            # Use the URL directly if it exists
            if "url" in matching_visuals[0]:
                sync_segment["image_url"] = matching_visuals[0]["url"]
            # Otherwise check for visual_elements
            elif "visual_elements" in matching_visuals[0]:
                elements = matching_visuals[0]["visual_elements"]
                if elements and isinstance(elements, list) and len(elements) > 0:
                    sync_segment["image_url"] = elements[0] if isinstance(elements[0], str) else ""
        
        timeline["segments"].append(sync_segment)
    
    # Set the total duration to the end time of the last segment
    if timeline["segments"]:
        timeline["total_duration"] = timeline["segments"][-1].get("end_time", "00:00:00")
    
    return timeline


@tool
def generate_image_for_segment(segment_text: str, segment_type: str = "verse") -> str:
    """
    Generates an appropriate image for a lyric segment based on its content and type.
    
    Args:
        segment_text: The text content of the segment to visualize.
        segment_type: The type of segment (verse, chorus, etc.)
        
    Returns:
        URL to the generated image.
    """
    # Limit the text to ensure prompt isn't too long
    max_text_length = 200
    truncated_text = segment_text[:max_text_length] + "..." if len(segment_text) > max_text_length else segment_text
    
    # Create an appropriate prompt based on segment content and type
    if segment_type == "introduction":
        segment_type = "intro"  # Normalize type names
        
    base_prompt = f"Create a conceptual visualization for this audio transcript: '{truncated_text}'"
    
    # Enhance prompt based on segment type
    if segment_type == "intro":
        prompt = f"{base_prompt} Create an establishing mood image that introduces the themes of this discussion."
    elif segment_type == "main_point":
        prompt = f"{base_prompt} Create a vibrant, meaningful centerpiece image that captures the key concept being discussed."
    elif segment_type == "example":
        prompt = f"{base_prompt} Create a specific illustrative image showing this example concept."
    elif segment_type == "anecdote":
        prompt = f"{base_prompt} Create a narrative image that tells the story within this anecdote."
    elif segment_type == "discussion":
        prompt = f"{base_prompt} Create an image showing intellectual discussion or debate about these concepts."
    elif segment_type == "conclusion":
        prompt = f"{base_prompt} Create a concluding image that gives a sense of resolution or key takeaway."
    else:  # verse or default
        prompt = f"{base_prompt} Create a conceptual image that represents the ideas in this text."

    try:
        # Generate the image using DALL-E
        # Using invoke() instead of __call__ to avoid deprecation warning
        image_url = generate_image.invoke(prompt)
        print(f"Generated image with prompt: {prompt[:100]}...")
        return image_url
    except Exception as e:
        print(f"Error in generate_image_for_segment: {str(e)}")
        # Return a placeholder image as fallback
        return "https://via.placeholder.com/800x600?text=AI+Visualization"

@tool
def finalize_visualization(timeline: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Finalizes the visualization by compiling all elements into a ready-to-render format.
    
    Args:
        timeline: A timeline of synchronized visual and audio elements.
        
    Returns:
        Complete visualization data ready for rendering.
    """
    global GLOBAL_LYRICS_DATA
    
    # Ensure we have a valid timeline object
    if timeline is None:
        # Try to create a default timeline based on global data
        timeline = {"segments": [], "total_duration": "00:00:00"}
        
        # If we have global lyrics data, use it to populate segments
        if GLOBAL_LYRICS_DATA:
            if isinstance(GLOBAL_LYRICS_DATA, list):
                # Create basic segments from the lines
                for i, line in enumerate(GLOBAL_LYRICS_DATA):
                    if isinstance(line, dict):
                        segment = {
                            "segment_id": i,
                            "segment_type": "verse",
                            "start_time": line.get("timestamp", "00:00:00"),
                            "end_time": line.get("end_timestamp", "00:00:00"),
                            "text": line.get("text", ""),
                            "image_url": ""
                        }
                        timeline["segments"].append(segment)
            elif isinstance(GLOBAL_LYRICS_DATA, dict) and "segments" in GLOBAL_LYRICS_DATA:
                timeline["segments"] = GLOBAL_LYRICS_DATA["segments"]
                if "total_duration" in GLOBAL_LYRICS_DATA:
                    timeline["total_duration"] = GLOBAL_LYRICS_DATA["total_duration"]
    if not timeline or not isinstance(timeline, dict) or "segments" not in timeline:
        # Set a default timeline if none is provided
        timeline = {
            "segments": [],
            "total_duration": "00:00:00"
        }
        # Return an error if there are no segments
        if not timeline.get("segments", []):
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
    
    # Process segments and create scenes
    processed_segments = []
    
    # First, generate images for all segments if needed
    for segment in timeline.get("segments", []):
        # Generate an image for this segment if no image_url is provided
        if (not segment.get("image_url") or segment.get("image_url") == "") and segment.get("text"):
            try:
                # Use the .invoke() method instead of direct function call with keyword arguments
                print(f"Generating image for segment with text: {segment.get('text', '')[:50]}...")
                image_url = generate_image_for_segment.invoke({
                    "segment_text": segment.get("text", ""),
                    "segment_type": segment.get("segment_type", "verse")
                })
                
                # Validate the image URL
                if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                    segment["image_url"] = image_url
                    print(f"Successfully generated image: {image_url[:50]}...")
                else:
                    # If the URL is invalid, create a placeholder URL for testing
                    print(f"Invalid image URL generated: {image_url}")
                    segment["image_url"] = "https://via.placeholder.com/800x600?text=AI+Visualization"
            except Exception as e:
                print(f"Error generating image: {str(e)}")
                # Set a fallback image URL for testing
                segment["image_url"] = "https://via.placeholder.com/800x600?text=Error+Generating+Image"
        processed_segments.append(segment)
    
    # Helper function to convert time string to seconds
    def time_str_to_seconds(time_str):
        if not time_str:
            return 0
        parts = time_str.split(":")
        if len(parts) == 2:  # MM:SS format
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS format
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    
    # Helper function to convert seconds to time string
    def seconds_to_time_str(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    # Split segments into scenes, ensuring no scene is longer than 5 seconds
    MAX_SCENE_DURATION = 5  # Maximum duration in seconds
    TARGET_AVG_DURATION = 3  # Target average duration in seconds
    
    for segment in processed_segments:
        start_seconds = time_str_to_seconds(segment.get("start_time", "00:00:00"))
        end_seconds = time_str_to_seconds(segment.get("end_time", "00:00:00"))
        duration = end_seconds - start_seconds
        
        # Skip invalid segments
        if duration <= 0:
            continue
            
        # Determine appropriate transitions based on segment type
        transitions = ["fade_in"]
        if segment.get("segment_type") == "chorus":
            transitions = ["dissolve"]
        elif segment.get("segment_type") == "bridge":
            transitions = ["slide_left"]
        
        # Split long segments into multiple scenes
        if duration > MAX_SCENE_DURATION:
            # Calculate number of scenes needed
            # Use TARGET_AVG_DURATION to create more scenes (but each ≤ MAX_SCENE_DURATION)
            num_scenes = max(2, round(duration / TARGET_AVG_DURATION))
            scene_duration = duration / num_scenes
            
            # Create multiple scenes from this segment
            for i in range(num_scenes):
                scene_start = start_seconds + (i * scene_duration)
                scene_end = min(end_seconds, scene_start + scene_duration)
                
                # Ensure we have an image URL to use
                image_url = segment.get("image_url", "")
                if not image_url or image_url == "":
                    # Use a placeholder if no image is available
                    image_url = "https://via.placeholder.com/800x600?text=AI+Visualization"
                
                scene = {
                    "start_time": seconds_to_time_str(scene_start),
                    "end_time": seconds_to_time_str(scene_end),
                    "visual_elements": [image_url],  # Always include an image URL
                    "transitions": transitions,
                    "text_overlay": segment.get("text", "")
                }
                visualization["scenes"].append(scene)
                
                # Add sync point for first scene only
                if i == 0:
                    visualization["audio_sync_points"].append({
                        "time": seconds_to_time_str(scene_start),
                        "event": f"segment_{segment.get('segment_id', 0)}"
                    })
        else:
            # Create a single scene for this segment (already under MAX_SCENE_DURATION)
            # Ensure we have an image URL to use
            image_url = segment.get("image_url", "")
            if not image_url or image_url == "":
                # Use a placeholder if no image is available
                image_url = "https://via.placeholder.com/800x600?text=AI+Visualization"
                
            scene = {
                "start_time": segment.get("start_time", "00:00:00"),
                "end_time": segment.get("end_time", "00:00:00"),
                "visual_elements": [image_url],  # Always include an image URL
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
    """State for the Audio Visualizer agent."""
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
        model="gpt-4o-mini", 
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
    # Note: tools now include generate_image_for_segment to ensure image generation
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
    # Limit to 10 iterations as mentioned in the memory to prevent recursion limit error
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
    # Sample data for testing - with generic content only
    sample_lyrics_data = [
        {"timestamp": "00:00:05", "text": "Line 1 of transcript"},
        {"timestamp": "00:00:10", "text": "Line 2 of transcript"},
        {"timestamp": "00:00:15", "text": "Line 3 of transcript"},
        {"timestamp": "00:00:20", "text": "Line 4 of transcript"},
    ]
    
    sample_audio_data = {
        "duration": 180,  # seconds
        "tempo": 120,  # BPM
        "key": "C Major",
        "sections": [
            {"start": 0, "end": 20, "type": "section1"},
            {"start": 20, "end": 60, "type": "section2"},
            {"start": 60, "end": 90, "type": "section3"},
        ]
    }
    
    sample_requirements = "Create a visualization based on the sample content."
    
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
