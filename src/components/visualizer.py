import streamlit as st
import json
import os
from typing import Dict, Any, List
import tempfile
import base64
from difflib import SequenceMatcher
import numpy as np
# For MoviePy 2.1.2, need to import classes from their respective modules
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip, ColorClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# Define custom fx functions for video effects
# In MoviePy 2.1.2, these effects are methods on the clip objects directly
def fadein(clip, duration):
    # Apply crossfadein effect manually
    return clip.crossfadein(duration)
    
def fadeout(clip, duration):
    # Apply crossfadeout effect manually
    return clip.crossfadeout(duration)
    
def resize(clip, width=None, height=None):
    return clip.resize(width=width, height=height)
import requests
from io import BytesIO
from PIL import Image
import time
import re


def display_visualization(visualization_data: Dict[str, Any]) -> None:
    """
    Display the visualization results in the Streamlit interface.
    
    Args:
        visualization_data: Dictionary containing visualization results from the agent.
    """
    if not visualization_data:
        st.error("No visualization data received from the agent.")
        return
        
    # Extract visualization data from agent response if needed
    # The visualization data might be directly in the input or in a tool response
    extracted_data = visualization_data
    
    # Check if the data is nested inside tool messages
    if "messages" in visualization_data:
        # Look for visualization data in the last few messages
        for msg in reversed(visualization_data["messages"]):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                try:
                    # Try to parse JSON content from the message
                    content_data = json.loads(msg.content) if msg.content.strip().startswith('{') else {}
                    
                    # Check if this contains visualization data
                    if isinstance(content_data, dict) and "scenes" in content_data:
                        extracted_data = content_data
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass
            elif hasattr(msg, "tool_calls"):
                # Look for the finalize_visualization tool call result
                for tool_call in getattr(msg, "tool_calls", []):
                    if getattr(tool_call, "name", "") == "finalize_visualization":
                        extracted_data = visualization_data
                        break
    
    # For cases where the data is in the 'tool_message' format
    if hasattr(visualization_data.get("messages", []), "__iter__"):
        for msg in visualization_data.get("messages", []):
            if hasattr(msg, "name") and msg.name == "finalize_visualization" and hasattr(msg, "content"):
                try:
                    content_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(content_data, dict) and "scenes" in content_data:
                        extracted_data = content_data
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass
    
    # Create a chat-like interface for agent-user interaction
    st.subheader("Feedback & Interaction")
    chat_container = st.container()
    
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
        # Add initial messages from the agent process to the chat history
        # with deduplication to prevent repetitive messages
        if "messages" in visualization_data:
            existing_messages = set()
            
            for msg in visualization_data["messages"]:
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    
                    # Skip messages that look like JSON error messages or tool output
                    if content.strip().startswith('{') and content.strip().endswith('}'):
                        try:
                            # Try to parse as JSON to confirm it's actually JSON
                            json_obj = json.loads(content)
                            # If it has error fields, it's likely an error message from a tool
                            if 'error' in json_obj or 'example' in json_obj:
                                continue
                        except json.JSONDecodeError:
                            # Not valid JSON, so treat as normal message
                            pass
                    
                    # Create a simplified version of the message to check for similarity
                    # Strip whitespace and lowercase for comparison
                    simple_content = content.strip().lower()
                    
                    # Skip if too similar to previous messages (75% similarity threshold)
                    if any(is_similar_message(simple_content, existing) for existing in existing_messages):
                        continue
                        
                    # Add to existing messages for future comparisons
                    existing_messages.add(simple_content)
                    
                    if msg.type == "ai":
                        st.session_state.chat_history.append({"role": "assistant", "content": content})
                    else:
                        st.session_state.chat_history.append({"role": "user", "content": content})
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div style='background-color: #f0f2f6; color: #333333; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #e1f5fe; color: #333333; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Agent:</strong> {message['content']}</div>", unsafe_allow_html=True)
    

    # Extract and display the visualization result
    st.subheader("Visualization Plan")
    
    # Check for the specific output format
    if isinstance(extracted_data, dict) and "scenes" in extracted_data:
        # Display general information about the visualization
        st.info(f"Created a {extracted_data.get('visualization_type', 'video')} visualization with {len(extracted_data['scenes'])} scenes.")
        
        # Display style information
        if "style" in extracted_data:
            style = extracted_data["style"]
            st.subheader("Visual Style")
            
            # Color palette display
            if "color_palette" in style:
                st.write("Color Palette:")
                cols = st.columns(len(style["color_palette"]))
                for i, color in enumerate(style["color_palette"]):
                    cols[i].markdown(
                        f'<div style="background-color: {color}; height: 50px; border-radius: 5px;"></div>',
                        unsafe_allow_html=True
                    )
            
            # Typography information
            if "typography" in style:
                st.write("Typography:")
                st.markdown(
                    f"Font: **{style['typography'].get('font', 'Default')}**  |  "
                    f"Color: **{style['typography'].get('main_color', '#FFFFFF')}**"
                )
            
            # Transitions list
            if "transitions" in style:
                st.write("Transitions:", ", ".join(style["transitions"]))
        
        # Timeline visualization
        st.subheader("Timeline")
        timeline_data = []
        
        try:
            for i, scene in enumerate(extracted_data["scenes"]):
                timeline_data.append({
                    "scene": i+1,
                    "start": scene.get("start_time", "00:00:00"),
                    "end": scene.get("end_time", "00:00:00"),
                    "elements": ", ".join(scene.get("visual_elements", [])),
                    "text": scene.get("text_overlay", "")
                })
        except KeyError as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.dataframe(timeline_data)
        
        # Scene details with expandable sections
        st.subheader("Scene Details")
        try:
            for i, scene in enumerate(extracted_data["scenes"]):
                with st.expander(f"Scene {i+1}: {scene.get('start_time', '00:00:00')} - {scene.get('end_time', '00:00:00')}"):
                    st.write(f"**Duration**: {scene.get('start_time', '00:00:00')} - {scene.get('end_time', '00:00:00')}")
                    st.write(f"**Text Overlay**: {scene.get('text_overlay', 'None')}")
                    st.write(f"**Transitions**: {', '.join(scene.get('transitions', ['None']))}")
                    
                    # Display visual elements
                    st.write("**Visual Elements**:")
                    for element in scene.get("visual_elements", []):
                        st.write(f"- {element}")
        except KeyError as e:
            st.error(f"An error occurred in Scene Details: {str(e)}")
        
        # Display a mock preview of the visualization
        st.subheader("Preview")
        st.warning("This is a conceptual preview. The actual rendered video would incorporate all elements described above.")
        
        # Check if 'scenes' exists in visualization_data
        if 'scenes' in visualization_data and visualization_data['scenes']:
            # Create a simple preview placeholder
            scenes = visualization_data['scenes']
            preview_html = f"""
            <div style="width: 100%; height: 300px; background: linear-gradient(to right, {style['color_palette'][0]}, {style['color_palette'][-1]}); 
                        display: flex; align-items: center; justify-content: center; border-radius: 10px;">
                <div style="text-align: center; color: white; padding: 20px;">
                    <h3 style="color: white;">Lyrics Visualization</h3>
                    <p style="color: white;">Format: {visualization_data.get('output_format', 'mp4')}</p>
                    <p style="color: white;">Duration: {scenes[-1].get('end_time', '00:00:00')}</p>
                    <p style="color: white;">{len(scenes)} scenes</p>
                </div>
            </div>
            """
        else:
            # Create a placeholder when no scenes are available
            preview_html = f"""
            <div style="width: 100%; height: 300px; background: linear-gradient(to right, {style['color_palette'][0]}, {style['color_palette'][-1]}); 
                        display: flex; align-items: center; justify-content: center; border-radius: 10px;">
                <div style="text-align: center; color: white; padding: 20px;">
                    <h3 style="color: white;">Lyrics Visualization</h3>
                    <p style="color: white;">Format: {visualization_data.get('output_format', 'mp4')}</p>
                    <p style="color: white;">Duration: 00:00:00</p>
                    <p style="color: white;">No scenes available</p>
                </div>
            </div>
            """
        st.markdown(preview_html, unsafe_allow_html=True)
        
        # Provide options to generate and download the video
        st.subheader("Generate Music Video")
        
        # Check if we already have a generated video in session state
        if 'video_result' not in st.session_state:
            st.session_state.video_result = None

        # Create a form for audio file upload
        with st.form("video_generation_form"):
            st.write("Upload audio file for the music video:")
            audio_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])
            generate_button = st.form_submit_button("Generate Music Video")

            if generate_button and audio_file is not None:
                # Save the uploaded audio file to a temporary location
                temp_dir = tempfile.mkdtemp()
                audio_path = os.path.join(temp_dir, audio_file.name)
                with open(audio_path, 'wb') as f:
                    f.write(audio_file.getbuffer())
                
                # Show spinner while processing
                with st.spinner("Generating music video, please wait..."):
                    # Import the stitch_music_video function from this module to avoid circular imports
                    # Call the video stitching function
                    from src.components.visualizer import stitch_music_video as sm
                    result = sm(extracted_data, audio_path)
                    st.session_state.video_result = result

        # Display generated video if available
        if st.session_state.video_result:
            result = st.session_state.video_result
            
            if result.get("success", False):
                output_file = result.get("output_file", "")
                if os.path.exists(output_file):
                    st.success("Music video generated successfully!")
                    
                    # Display video
                    st.video(output_file)
                    
                    # Provide download link
                    st.markdown(get_file_download_link(output_file, "Download Music Video"), unsafe_allow_html=True)
                    
                    # Show metadata
                    st.json({
                        "duration": result.get("duration", 0),
                        "resolution": result.get("resolution", ""),
                        "scenes_count": result.get("scenes_count", 0),
                        "created_at": result.get("created_at", "")
                    })
                else:
                    st.error(f"Generated file not found: {output_file}")
            else:
                st.error(f"Failed to generate video: {result.get('error', 'Unknown error')}")
                if 'scenes' in result:
                    st.warning("Visualization data was parsed correctly, but video generation failed.")
        else:
            st.info("Upload an audio file and click 'Generate Music Video' to create a complete music video with the visualizations.")
        
        # Display scenes in an expander
        with st.expander("Scene Details", expanded=True):
            for i, scene in enumerate(extracted_data["scenes"]):
                st.markdown(f"**Scene {i+1}:** {scene['start_time']} - {scene['end_time']}")
                st.markdown(f"*{scene.get('text_overlay', '(No text overlay)')}*")
                st.markdown("---")
        
        # Output the full JSON data for debugging/development
        with st.expander("Raw Visualization Data (JSON)", expanded=False):
            st.json(extracted_data)
        
        # Always display the full trace data prominently
        st.subheader("Agent Execution Traces")
        st.json(visualization_data)
            
    else:
        # Handle unexpected data formats but still show the traces
        st.warning("The visualization data is not in the expected format. Attempting to display raw data.")
        
        # Display the raw data for debugging
        with st.expander("Raw Visualization Data", expanded=False):
            st.json(visualization_data)
        
        # Always show agent traces if available - display prominently for debugging
        st.subheader("Agent Execution Traces & Debug Information")
        st.json(visualization_data)


def is_similar_message(message1: str, message2: str, threshold: float = 0.75) -> bool:
    """
    Check if two messages are similar using sequence matching.
    
    Args:
        message1: First message to compare
        message2: Second message to compare
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        True if messages are similar, False otherwise
    """
    # Use SequenceMatcher to calculate similarity ratio
    similarity = SequenceMatcher(None, message1, message2).ratio()
    return similarity >= threshold


def get_file_download_link(file_path, link_text="Download File"):
    """Generate a download link for a file."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)
    mime_type = "application/octet-stream"
    href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}">{link_text}</a>'
    return href


def download_file(url, local_path=None):
    """
    Downloads a file from a URL to a local path or returns bytes.
    
    Args:
        url: URL of the file to download
        local_path: Optional path to save the file locally
        
    Returns:
        Path to downloaded file if local_path provided, else file bytes
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        if local_path:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_path
        else:
            return BytesIO(response.content)
    except Exception as e:
        print(f"Error downloading file from {url}: {e}")
        return None


def time_to_seconds(time_str):
    """Convert time string (HH:MM:SS) to seconds."""
    if not time_str:
        return 0
        
    # Check if it's already in seconds (float or int)
    if isinstance(time_str, (int, float)):
        return time_str
        
    # Handle time strings
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        try:
            return float(time_str)
        except:
            return 0


def stitch_music_video(visualization_data, audio_file_path, output_file=None):
    """
    Creates a complete music video by stitching together visuals, audio, and synchronized lyrics.
    
    Args:
        visualization_data: Dictionary containing visualization information, scenes, etc.
        audio_file_path: Path to the audio file (MP3, WAV, etc.)
        output_file: Optional path for output file, will use tempfile if not provided
        
    Returns:
        Dictionary with output file path and metadata
    """
    # Create a tempfile for output if not provided
    if not output_file:
        temp_dir = tempfile.mkdtemp()
        timestamp = int(time.time())
        output_file = os.path.join(temp_dir, f"lyric_video_{timestamp}.mp4")
    
    try:
        # Load audio file - using properly imported classes from MoviePy 2.1.2
        audio = AudioFileClip(audio_file_path)
        audio_duration = audio.duration
        
        # Extract scenes from visualization data - with better error handling
        scenes = []
        
        if isinstance(visualization_data, dict):
            # Direct access if scenes is at the top level
            if "scenes" in visualization_data:
                scenes = visualization_data["scenes"]
            # Check if it might be in a nested structure
            elif "data" in visualization_data and isinstance(visualization_data["data"], dict):
                if "scenes" in visualization_data["data"]:
                    scenes = visualization_data["data"]["scenes"]
            # Check in messages
            elif "messages" in visualization_data:
                # Process messages to find scenes data
                for msg in reversed(visualization_data["messages"]):
                    # Handle dict-like messages
                    if isinstance(msg, dict) and "content" in msg:
                        try:
                            content = msg["content"]
                            if isinstance(content, str) and content.strip().startswith('{'):
                                content_data = json.loads(content)
                                if isinstance(content_data, dict) and "scenes" in content_data:
                                    scenes = content_data["scenes"]
                                    break
                        except (json.JSONDecodeError, KeyError):
                            pass
                    # Handle object-like messages
                    elif hasattr(msg, "content"):
                        try:
                            content = msg.content
                            if isinstance(content, str) and content.strip().startswith('{'):
                                content_data = json.loads(content)
                                if isinstance(content_data, dict) and "scenes" in content_data:
                                    scenes = content_data["scenes"]
                                    break
                        except (json.JSONDecodeError, AttributeError):
                            pass
        
        # If no scenes found, create a default scene
        if not scenes:
            print(f"Warning: Could not find scenes in visualization data. Creating default scene.")
            scenes = [{
                "start_time": 0,
                "end_time": audio_duration,
                "text_overlay": "No lyrics data available",
                "visual_elements": [],
                "transitions": []
            }]
        
        # Get style information for text formatting
        style = visualization_data.get("style", {})
        font = style.get("typography", {}).get("font", "Arial")
        font_color = style.get("typography", {}).get("main_color", "white")
        
        # Prepare video clips for each scene
        video_clips = []
        
        # Create temporary directory for downloaded files
        temp_dir = tempfile.mkdtemp()
        
        # Set default video size and background color
        video_width, video_height = 1280, 720
        bg_color = [0, 0, 0]  # Default black background
        
        # Track the current time position
        current_position = 0
        
        for i, scene in enumerate(scenes):
            # Get scene timing
            start_time = time_to_seconds(scene.get("start_time", current_position))
            end_time = time_to_seconds(scene.get("end_time", min(start_time + 10, audio_duration)))
            duration = end_time - start_time
            
            if duration <= 0:
                continue  # Skip invalid scenes
            
            # Update current position for next scene
            current_position = end_time
            
            # Get visual elements
            visual_elements = scene.get("visual_elements", [])
            
            # Text overlay (lyrics)
            text_overlay = scene.get("text_overlay", "")
            
            # Create base clip (either from visual elements or a color background)
            if visual_elements:
                element_clips = []
                
                for element in visual_elements:
                    # Handle different types of visual elements
                    if isinstance(element, str):
                        # Check if it's a URL, local file, or just a placeholder
                        if element.startswith("http"):
                            # Download from URL
                            local_path = os.path.join(temp_dir, f"element_{i}_{len(element_clips)}.jpg")
                            download_file(element, local_path)
                            
                            # Create clip based on file type
                            if any(element.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif']):
                                clip = VideoFileClip(local_path)
                                clip = clip.set_position("center")
                            else:
                                clip = ImageClip(local_path)
                                clip = clip.set_duration(duration).set_position("center")
                                
                            # Resize to fit video dimensions
                            clip = clip.resize(height=video_height)
                            element_clips.append(clip)
                        elif os.path.exists(element):
                            # Local file
                            if any(element.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif']):
                                clip = VideoFileClip(element)
                                clip = clip.set_position("center")
                            else:
                                clip = ImageClip(element)
                                clip = clip.set_duration(duration).set_position("center")
                                
                            # Resize to fit video dimensions
                            clip = clip.resize(height=video_height)
                            element_clips.append(clip)
                        else:
                            # Placeholder - create a colored background with text
                            color_id = i % len(style.get("color_palette", ["#000000"]))
                            color_hex = style.get("color_palette", ["#000000"])[color_id]
                            color = hex_to_rgb(color_hex)
                            
                            base_clip = ColorClip(size=(video_width, video_height), color=color)
                            base_clip = base_clip.set_duration(duration)
                            element_clips.append(base_clip)
                
                # Composite visual elements if multiple exist
                if element_clips:
                    if len(element_clips) == 1:
                        base_clip = element_clips[0]
                    else:
                        base_clip = CompositeVideoClip(element_clips, size=(video_width, video_height))
                else:
                    # Fallback to colored background
                    color_id = i % len(style.get("color_palette", ["#000000"]))
                    color_hex = style.get("color_palette", ["#000000"])[color_id]
                    color = hex_to_rgb(color_hex)
                    
                    base_clip = ColorClip(size=(video_width, video_height), color=color)
                    base_clip = base_clip.set_duration(duration)
            else:
                # Create colored background
                color_id = i % len(style.get("color_palette", ["#000000"]))
                color_hex = style.get("color_palette", ["#000000"])[color_id]
                color = hex_to_rgb(color_hex)
                
                base_clip = ColorClip(size=(video_width, video_height), color=color)
                base_clip = base_clip.set_duration(duration)
            
            # Ensure the duration is set for the base clip
            base_clip = base_clip.set_duration(duration)
            
            # Add text overlay (lyrics)
            if text_overlay:
                # Create text clip with styling
                text_clip = TextClip(
                    text_overlay,
                    font=font,
                    fontsize=40,
                    color=font_color,
                    bg_color="transparent",
                    stroke_color="black",
                    stroke_width=1.5
                )
                text_clip = text_clip.set_position(("center", "bottom")).set_duration(duration)
                
                # Composite base and text clips
                scene_clip = CompositeVideoClip([base_clip, text_clip], size=(video_width, video_height))
            else:
                scene_clip = base_clip
            
            # Apply transitions if specified
            transitions = scene.get("transitions", [])
            
            if transitions and transitions[0] == "fade_in":
                # Direct application of crossfadein
                fade_duration = min(1.0, duration / 3)
                try:
                    scene_clip = scene_clip.crossfadein(fade_duration)
                except Exception as e:
                    print(f"Warning: Could not apply fade_in transition: {e}")
            
            if "fade_out" in transitions:
                # Direct application of crossfadeout
                fade_duration = min(1.0, duration / 3)
                try:
                    scene_clip = scene_clip.crossfadeout(fade_duration)
                except Exception as e:
                    print(f"Warning: Could not apply fade_out transition: {e}")
            
            # Set the clip to start at the appropriate time
            scene_clip = scene_clip.set_start(start_time)
            
            # Add to the list of video clips
            video_clips.append(scene_clip)
        
        # Combine all clips
        if not video_clips:
            # Default clip if no scenes were successfully processed
            default_clip = ColorClip(size=(video_width, video_height), color=(0, 0, 0))
            default_clip = default_clip.set_duration(audio_duration)
            video_clips.append(default_clip)
        
        # Create the final video by compositing all clips
        final_video = CompositeVideoClip(video_clips, size=(video_width, video_height))
        
        # Set the audio track
        final_video = final_video.set_audio(audio)
        
        # Ensure video duration matches audio duration
        final_video = final_video.set_duration(audio_duration)
        
        # Write the output file
        final_video.write_videofile(
            output_file,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"{output_file}.temp-audio.m4a",
            remove_temp=True,
            fps=24
        )
        
        return {
            "success": True,
            "output_file": output_file,
            "duration": audio_duration,
            "resolution": f"{video_width}x{video_height}",
            "scenes_count": len(scenes),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"Error creating video: {error_message}")
        return {
            "success": False,
            "error": error_message,
            "scenes": visualization_data.get("scenes", [])
        }


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_demo_visualization() -> Dict[str, Any]:
    """Create a demo visualization for testing purposes."""
    return {
        "visualization_type": "video",
        "scenes": [
            {
                "start_time": "00:00:00",
                "end_time": "00:00:05",
                "visual_elements": ["intro_image.jpg"],
                "transitions": ["fade_in"],
                "text_overlay": "Demo Song Title"
            },
            {
                "start_time": "00:00:05",
                "end_time": "00:00:15",
                "visual_elements": ["nature_scene.jpg", "bird_flying.gif"],
                "transitions": ["dissolve"],
                "text_overlay": "First verse lyrics here"
            },
            {
                "start_time": "00:00:15",
                "end_time": "00:00:25",
                "visual_elements": ["ocean_waves.mp4"],
                "transitions": ["slide_left"],
                "text_overlay": "More lyrics continuing the story"
            }
        ],
        "audio_sync_points": [
            {"time": "00:00:00", "event": "start"},
            {"time": "00:00:05", "event": "first_verse"},
            {"time": "00:00:15", "event": "second_verse"}
        ],
        "style": {
            "color_palette": ["#3A86FF", "#FF006E", "#FB5607"],
            "typography": {"font": "Montserrat", "main_color": "#FFFFFF"},
            "transitions": ["fade", "dissolve", "slide"]
        },
        "output_format": "mp4",
        "metadata": {
            "generated_at": "2025-04-19T16:30:00",
            "version": "1.0"
        }
    }


def print_stream(stream):
    """
    Prints the content of a LangGraph agent stream.
    
    Args:
        stream: The stream from agent.stream() containing state dictionaries
    """
    for s in stream:
        if "messages" not in s or not s["messages"]:
            print("Empty or invalid state:", s)
            continue
            
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        elif hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(f"Message type: {type(message).__name__}")
            print(message)


if __name__ == "__main__":
    # Test the visualization component
    demo_data = create_demo_visualization()
    
    # Create a simple Streamlit app for testing
    st.title("Visualization Component Test")
    display_visualization(demo_data)
