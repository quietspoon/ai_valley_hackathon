import streamlit as st
import json
import os
import shutil
from typing import Dict, Any, List, Union, Optional
import base64
import time
import tempfile
from PIL import Image
import io
import requests
from urllib.parse import urlparse
import numpy as np
import re
import difflib
import moviepy.editor as mpy
from difflib import SequenceMatcher
# Import change_settings from moviepy.config
from moviepy.config import change_settings

# Configure ImageMagick
change_settings({"IMAGEMAGICK_BINARY": None})

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
        
        # Display audio duration source - actual or fallback
        if 'using_fallback_duration' in st.session_state and st.session_state.using_fallback_duration:
            st.warning("⚠️ Using fallback duration of 26 seconds. Audio file could not be loaded.")
        elif 'audio_duration' in st.session_state:
            st.success(f"✓ Using actual audio duration: {st.session_state.audio_duration:.2f} seconds")
        else:
            st.info("ℹ️ Audio duration not yet determined")
            
        timeline_data = []
        
        # Check if all timestamps are zeros, which indicates they need to be calculated
        scenes = extracted_data["scenes"]
        all_zero_timestamps = all(s.get("start_time", "00:00:00") == "00:00:00" for s in scenes)
        
        # If all are zero, assign proportional timestamps based on total duration
        if all_zero_timestamps:
            # Get transcript data if available in session state
            transcript_data = st.session_state.get("lyrics_data", [])
            # Use a more reasonable default duration - 126 seconds (2:06 minutes) for the current song
            audio_duration = st.session_state.get("audio_duration", 126)  # Default to 126 seconds (2:06 minutes) for this song
            
            # Calculate timestamps based on scene distribution
            num_scenes = len(scenes)
            for i, scene in enumerate(scenes):
                # Calculate appropriate start and end times
                scene_duration = audio_duration / num_scenes
                start_seconds = i * scene_duration
                end_seconds = (i + 1) * scene_duration
                
                # Format as time strings
                start_minutes = int(start_seconds // 60)
                start_secs = int(start_seconds % 60)
                end_minutes = int(end_seconds // 60)
                end_secs = int(end_seconds % 60)
                
                scene["start_time"] = f"{start_minutes:02d}:{start_secs:02d}:00"
                scene["end_time"] = f"{end_minutes:02d}:{end_secs:02d}:00"
        
        try:
            for i, scene in enumerate(scenes):
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
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'


def create_video_from_visualization(visualization_data: Dict[str, Any], transcript_data: List[Dict[str, str]], audio_path: str, use_existing_visuals: bool = False, scene_folder_name: str = None) -> str:
    """
    Create a video from visualization data, transcript, and audio.
    
    Args:
        visualization_data: Dictionary with scenes, styles, etc.
        transcript_data: List of dictionaries with timestamps and text (transcript lines)
        audio_path: Path to the audio file
        
    Returns:
        Path to the created video file
    """
    # Create a local reference to lyrics_data for compatibility
    # This ensures all code that might try to access lyrics_data will work
    lyrics_data = transcript_data
    
    # Extract scenes from visualization data
    extracted_data = visualization_data
    
    # Debugging: Print the structure of visualization_data
    st.write("Debug: Visualization data type:", type(visualization_data))
    
    # If visualization_data is a dict with "scenes", use it directly
    if isinstance(visualization_data, dict) and "scenes" in visualization_data:
        st.write("Debug: Found scenes directly in visualization_data")
        extracted_data = visualization_data
    # Otherwise, try to extract from messages
    elif "messages" in visualization_data:
        st.write("Debug: Looking for scenes in messages")
        for msg in reversed(visualization_data["messages"]):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                try:
                    content_data = json.loads(msg.content) if msg.content.strip().startswith('{') else {}
                    if isinstance(content_data, dict) and "scenes" in content_data:
                        extracted_data = content_data
                        st.write("Debug: Found scenes in message content")
                        break
                except (json.JSONDecodeError, AttributeError) as e:
                    st.write(f"Debug: Error parsing message content: {str(e)}")
    
    # Additional check if scenes is directly in the data structure
    if isinstance(extracted_data, list) and len(extracted_data) > 0 and isinstance(extracted_data[0], dict) and "image_url" in extracted_data[0]:
        st.write("Debug: Found list of scenes directly")
        extracted_data = {"scenes": extracted_data}
    
    # Verify we have extracted the scenes correctly
    if "scenes" in extracted_data:
        st.write(f"Debug: Found {len(extracted_data['scenes'])} scenes")
    else:
        st.warning("No scenes found in visualization data")
    
    # Create persistent directory for scene images
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generated_dir = os.path.join(base_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    # Create a timestamped folder name for this set of scenes if not provided
    if not scene_folder_name:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        scene_folder_name = f"scenes_{timestamp}"
    
    # Create the scenes directory
    scenes_dir = os.path.join(generated_dir, scene_folder_name)
    
    # If not using existing visuals, create a new directory
    if not use_existing_visuals:
        os.makedirs(scenes_dir, exist_ok=True)
    
    # Create temp directory for video assets and final output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load audio clip
        try:
            audio_clip = mpy.AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            # Always update audio duration in session state to ensure it's current
            st.session_state.audio_duration = audio_duration
            st.session_state.using_fallback_duration = False
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")
            # Create a placeholder audio with 126 seconds duration (2:06 minutes) for the current audio file
            audio_duration = 126
            audio_clip = None
            # Always update session state with the correct duration
            st.session_state.audio_duration = audio_duration
            st.session_state.using_fallback_duration = True
            
        # Create clips list
        clips = []
        
        # Default background if needed
        default_bg = mpy.ColorClip((1280, 720), color=(0, 0, 0), duration=audio_duration)
        
        # Check if we have scenes to work with
        if "scenes" in extracted_data and extracted_data["scenes"]:
            scenes = extracted_data["scenes"]
            
            # Check if all start times are 00:00:00, which indicates a problem
            all_zero_timestamps = all(s.get("start_time", "00:00:00") == "00:00:00" for s in scenes)
            
            # If all are zero, assign timestamps from transcript data
            if all_zero_timestamps and transcript_data:
                # First, sort the transcript data by timestamp to ensure it's in order
                sorted_lyrics = sorted(transcript_data, key=lambda x: time_to_seconds(x.get("timestamp", "00:00:00")))
                
                # Calculate how many transcript lines correspond to each scene roughly
                lines_per_scene = max(1, len(sorted_lyrics) // len(scenes))
                
                # Assign timestamps to scenes based on transcript timing
                for i, scene in enumerate(scenes):
                    # Get starting timestamp for this scene from corresponding transcript
                    start_idx = i * lines_per_scene
                    if start_idx < len(sorted_lyrics):
                        scene["start_time"] = sorted_lyrics[start_idx].get("timestamp", "00:00:00")
                    
                    # Calculate end time based on next scene's start time or audio duration
                    if i < len(scenes) - 1:
                        end_idx = (i + 1) * lines_per_scene
                        if end_idx < len(sorted_lyrics):
                            scene["end_time"] = sorted_lyrics[end_idx].get("timestamp", "00:00:00")
                        else:
                            # If we've run out of transcript lines, use a proportional timing
                            portion = (i + 1) / len(scenes)
                            seconds = int(audio_duration * portion)
                            scene["end_time"] = f"{seconds // 60:02d}:{seconds % 60:02d}"
                    else:
                        # Last scene goes to the end of the audio
                        minutes = int(audio_duration // 60)
                        seconds = int(audio_duration % 60)
                        scene["end_time"] = f"{minutes:02d}:{seconds:02d}"
            
            # Sort scenes by start time
            scenes = sorted(scenes, key=lambda x: time_to_seconds(x.get("start_time", "00:00:00")))
            
            # Process each scene
            for i, scene in enumerate(scenes):
                # Get scene timing
                start_time = time_to_seconds(scene.get("start_time", "00:00:00"))
                
                # For the last scene, use audio duration as end time
                if i == len(scenes) - 1:
                    end_time = audio_duration
                else:
                    end_time = time_to_seconds(scenes[i+1].get("start_time", "00:00:00"))
                
                # Create scene duration
                duration = end_time - start_time
                if duration <= 0:
                    continue  # Skip invalid scenes
                
                # Get background for this scene
                bg_clip = None
                
                # Persistent path for this scene's image
                persistent_img_path = os.path.join(scenes_dir, f"scene_{i}.jpg")
                
                if use_existing_visuals and os.path.exists(persistent_img_path):
                    # Use existing saved image
                    try:
                        # Copy to temp dir for processing
                        temp_img_path = os.path.join(temp_dir, f"scene_{i}.jpg")
                        shutil.copy2(persistent_img_path, temp_img_path)
                        
                        # Create image clip
                        img_clip = mpy.ImageClip(temp_img_path)
                        # Resize to 1280x720 while maintaining aspect ratio
                        img_clip = img_clip.resize(width=1280)
                        # Center the image
                        bg_clip = img_clip.set_position('center').set_duration(duration).set_start(start_time)
                        
                        # Apply fade effects
                        if duration > 1.0:
                            bg_clip = bg_clip.crossfadein(min(1.0, duration/4))
                            bg_clip = bg_clip.crossfadeout(min(1.0, duration/4))
                    except Exception as e:
                        st.warning(f"Error loading saved image for scene {i+1}: {str(e)}")
                        bg_clip = None
                # If not using existing visuals or if no saved image exists
                elif not use_existing_visuals:
                    # Get image URL from various possible locations
                    image_url = None
                    
                    # Check different possible field names for image URL
                    for field in ["image_url", "visual_element", "image", "url"]:
                        if field in scene and scene[field] and isinstance(scene[field], str):
                            image_url = scene[field]
                            st.write(f"Debug: Found image URL in field '{field}': {image_url[:30]}...")
                            break
                    
                    # Also check visual_elements list if present
                    if not image_url and "visual_elements" in scene and isinstance(scene["visual_elements"], list) and len(scene["visual_elements"]) > 0:
                        for element in scene["visual_elements"]:
                            if isinstance(element, str) and element and (element.startswith("http") or element.startswith("https")):
                                image_url = element
                                st.write(f"Debug: Found image URL in visual_elements list: {element[:30]}...")
                                break
                    
                    # Print the scene structure for debugging
                    st.write(f"Debug scene {i}: {str(scene)[:500]}...")
                    
                    # Check if we have a valid image URL
                    if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                        try:
                            # Log the image being processed
                            st.info(f"Processing image for scene {i+1}")
                            st.write(f"Image URL: {image_url[:60]}...")
                            
                            # Download image to temp directory
                            temp_img_path = os.path.join(temp_dir, f"scene_{i}.jpg")
                            result = download_file(image_url, temp_img_path)
                            
                            if result and os.path.exists(temp_img_path) and os.path.getsize(temp_img_path) > 0:
                                # Save a copy to the persistent directory
                                shutil.copy2(temp_img_path, persistent_img_path)
                                
                                try:
                                    # Verify the image can be opened by PIL first
                                    from PIL import Image
                                    img = Image.open(temp_img_path)
                                    img.verify()  # Verify it's a valid image
                                    st.write(f"Image verification passed: {img.format} {img.size}")
                                    
                                    # Create image clip
                                    img_clip = mpy.ImageClip(temp_img_path)
                                    # Resize to 1280x720 while maintaining aspect ratio
                                    img_clip = img_clip.resize(width=1280)
                                    # Center the image
                                    bg_clip = img_clip.set_position('center').set_duration(duration).set_start(start_time)
                                    
                                    # Apply fade effects
                                    if duration > 1.0:
                                        bg_clip = bg_clip.crossfadein(min(1.0, duration/4))
                                        bg_clip = bg_clip.crossfadeout(min(1.0, duration/4))
                                        
                                    # Log success
                                    st.success(f"Successfully added image to scene {i+1}")
                                except Exception as e:
                                    st.error(f"Error creating image clip: {str(e)}")
                                    # Fall back to a colored background but with a different color
                                    color_hex = "#FF5733"  # Bright orange for error cases
                                    rgb_color = hex_to_rgb(color_hex)
                                    bg_clip = mpy.ColorClip((1280, 720), color=rgb_color, duration=duration).set_start(start_time)
                            else:
                                st.error(f"Failed to download image for scene {i+1}")
                                # Fall back to a purple background to distinguish download failures
                                bg_clip = mpy.ColorClip((1280, 720), color=(128, 0, 128), duration=duration).set_start(start_time)
                        except Exception as e:
                            st.error(f"Error processing image for scene {i+1}: {str(e)}")
                            # Print more detailed error to help diagnose issues
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            # Fall back to a red background to distinguish processing failures
                            bg_clip = mpy.ColorClip((1280, 720), color=(255, 0, 0), duration=duration).set_start(start_time)
                        except Exception as e:
                            st.error(f"Error processing image for scene {i+1}: {str(e)}")
                            # Print more detailed error to help diagnose issues
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            bg_clip = None
                
                # Use color background as fallback
                if bg_clip is None:
                    # Use style colors if available
                    if "style" in extracted_data and "color_palette" in extracted_data["style"]:
                        try:
                            # Get a color from the palette
                            color_hex = extracted_data["style"]["color_palette"][i % len(extracted_data["style"]["color_palette"])]
                            rgb_color = hex_to_rgb(color_hex)
                            bg_clip = mpy.ColorClip((1280, 720), color=rgb_color, duration=duration).set_start(start_time)
                        except Exception:
                            # Fallback to grayscale color based on scene index
                            gray_value = 20 + (i * 30) % 235
                            bg_clip = mpy.ColorClip((1280, 720), color=(gray_value, gray_value, gray_value), duration=duration).set_start(start_time)
                    else:
                        # Default grayscale color
                        gray_value = 20 + (i * 30) % 235
                        bg_clip = mpy.ColorClip((1280, 720), color=(gray_value, gray_value, gray_value), duration=duration).set_start(start_time)
                
                # Add text overlay for transcript
                text_overlay = scene.get("text_overlay") or ""
                
                # Find matching transcript lines for this time period
                matching_lyrics = []
                for transcript_line in transcript_data:
                    transcript_time = time_to_seconds(transcript_line.get("timestamp", "00:00:00"))
                    if start_time <= transcript_time < end_time:
                        matching_lyrics.append(transcript_line["text"])
                
                # Add this scene to clips list
                clips.append(bg_clip)
            
            # If we have any clips, create the composite
            if clips:
                # Create final composite clip
                final_clip = mpy.CompositeVideoClip(clips, size=(1280, 720))
                final_clip = final_clip.set_duration(audio_duration)
                
                # Add audio if available
                if audio_clip:
                    final_clip = final_clip.set_audio(audio_clip)
            else:
                # Fallback to default background with audio
                final_clip = default_bg
                if audio_clip:
                    final_clip = final_clip.set_audio(audio_clip)
        else:
            # No scenes available, create a simple clip with default background
            final_clip = default_bg
            if audio_clip:
                final_clip = final_clip.set_audio(audio_clip)
        
        # Create output file path
        output_file = os.path.join(temp_dir, "visualization.mp4")
        
        # Write the clip to a file
        final_clip.write_videofile(
            output_file, 
            fps=24, 
            codec='libx264', 
            audio_codec='aac',
            threads=4,
            verbose=False,
            logger=None  # Suppress moviepy output
        )
        
        # Copy to a persistent location to avoid tempfile deletion
        persistent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "generated")
        os.makedirs(persistent_dir, exist_ok=True)
        
        # Create a timestamped filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        persistent_file = os.path.join(persistent_dir, f"visualization_{timestamp}.mp4")
        
        # Copy the file
        with open(output_file, "rb") as src_file, open(persistent_file, "wb") as dst_file:
            dst_file.write(src_file.read())
        
        return persistent_file


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    # Remove the hash at the beginning if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def display_video_player(video_path: str) -> None:
    """Display a video player in the Streamlit app."""
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return
        
    # Get file stats
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
    
    # Display video stats
    st.write(f"Video file: {os.path.basename(video_path)} ({file_size:.2f} MB)")
    
    # Check if the file is a valid video
    try:
        # Show some debug info about the video using MoviePy
        video = mpy.VideoFileClip(video_path)
        st.write(f"Video info: {video.size} resolution, {video.duration:.2f} seconds, {video.fps} fps")
        video.close()
    except Exception as e:
        st.error(f"Error reading video file: {str(e)}")
    
    # Use HTML5 video tag for better control
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
    # Create a download button
    st.download_button(
        label="Download Video",
        data=video_bytes,
        file_name=os.path.basename(video_path),
        mime="video/mp4"
    )


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
        # Ensure parent directory exists
        if local_path and not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Log the URL being downloaded (for debugging)
        st.write(f"Downloading image from: {url[:100]}...")
        
        # Special handling for Azure Blob Storage URLs
        if "blob.core.windows.net" in url:
            # Add headers to handle Azure Blob Storage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            # Try without streaming for Azure Blob Storage
            response = requests.get(url, timeout=30, headers=headers)
            st.write("Using Azure Blob Storage specific headers")
        else:
            response = requests.get(url, timeout=30)
            
        response.raise_for_status()
        st.write(f"Download status code: {response.status_code}")
        st.write(f"Content type: {response.headers.get('content-type', 'unknown')}")
        st.write(f"Content length: {response.headers.get('content-length', 'unknown')} bytes")
        
        if local_path:
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
            # Verify the file was actually written and has contents
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                file_size = os.path.getsize(local_path)
                st.success(f"Successfully downloaded image to {os.path.basename(local_path)} (Size: {file_size} bytes)")
                
                # Try to verify the image is valid
                try:
                    from PIL import Image
                    with Image.open(local_path) as img:
                        st.write(f"Image info: {img.format} {img.size} {img.mode}")
                        # Create a small thumbnail to verify it can be processed
                        thumb = img.copy()
                        thumb.thumbnail((100, 100))
                except Exception as img_error:
                    st.error(f"Downloaded file is not a valid image: {str(img_error)}")
                    return None
                    
                return local_path
            else:
                error_msg = f"Warning: Downloaded file is empty or does not exist: {local_path}"
                st.warning(error_msg)
                return None
        else:
            content = BytesIO(response.content)
            if content.getbuffer().nbytes > 0:
                st.success(f"Successfully downloaded content (Size: {content.getbuffer().nbytes} bytes)")
                return content
            else:
                error_msg = f"Warning: Downloaded content is empty from URL: {url[:60]}..."
                st.warning(error_msg)
                return None
    except Exception as e:
        error_msg = f"Error downloading image: {str(e)}"
        st.error(error_msg)
        import traceback
        st.code(traceback.format_exc(), language="python")
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
                "text_overlay": "Opening statement from the podcast"
            },
            {
                "start_time": "00:00:15",
                "end_time": "00:00:25",
                "visual_elements": ["ocean_waves.mp4"],
                "transitions": ["slide_left"],
                "text_overlay": "Continued discussion from the podcast"
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
