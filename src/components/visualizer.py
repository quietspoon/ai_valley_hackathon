import streamlit as st
import json
import os
from typing import Dict, Any, List
import tempfile
import base64
from difflib import SequenceMatcher


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
        
        # Create a simple preview placeholder
        preview_html = f"""
        <div style="width: 100%; height: 300px; background: linear-gradient(to right, {style['color_palette'][0]}, {style['color_palette'][-1]}); 
                    display: flex; align-items: center; justify-content: center; border-radius: 10px;">
            <div style="text-align: center; color: white; padding: 20px;">
                <h3 style="color: white;">Lyrics Visualization</h3>
                <p style="color: white;">Format: {visualization_data.get('output_format', 'mp4')}</p>
                <p style="color: white;">Duration: {visualization_data['scenes'][-1].get('end_time', '00:00:00')}</p>
                <p style="color: white;">{len(visualization_data['scenes'])} scenes</p>
            </div>
        </div>
        """
        st.markdown(preview_html, unsafe_allow_html=True)
        
        # Provide download option (this would be a real file in the full implementation)
        st.subheader("Export")
        st.info("In a complete implementation, this would allow downloading the generated video file.")
        
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
