import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.agent.react_agent import create_lyrics_visualizer_agent
from src.utils.file_processor import process_lyrics_file, process_audio_file
from src.components.visualizer import display_visualization

# Load environment variables
load_dotenv()

# Initialize session state variables if they don't exist
if 'agent' not in st.session_state:
    st.session_state.agent = None
    
if 'lyrics_data' not in st.session_state:
    st.session_state.lyrics_data = None
    
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

# Set page configuration
st.set_page_config(
    page_title="Lyrics Visualizer (Sample Mode)",
    page_icon="🎵",
    layout="wide"
)

def main():
    st.title("🎵 Lyrics Visualizer (Sample Mode)")
    st.subheader("Create visual experiences from sample lyrics")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("How it works")
        st.markdown("""
        1. The app automatically uses sample lyrics and audio files
        2. Enter your requirements for visualization
        3. Click 'Generate Visualization'
        """)
        
        # Add debug mode toggle
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show agent's thought process")
        
        st.header("About")
        st.markdown("""
        This app uses an AI agent to analyze lyrics and audio, 
        then creates visuals that match the mood and meaning of the content.
        
        Currently using samples from The Beatles' "Her Majesty".
        """)
    
    # Main content area - Vertical layout
    # Input section
    st.header("Input")
    
    # Display info about sample files
    st.info("Using sample files from the 'sample' directory:")
    st.markdown("- **Lyrics**: Her Majesty (Remastered 2009) (128kbit_AAC).txt")
    st.markdown("- **Audio**: Her Majesty (Remastered 2009) (128kbit_AAC).mp3")
    
    # User requirements input
    st.subheader("Visualization Requirements")
    user_requirements = st.text_area(
        "Describe what kind of visualization you want",
        height=150,
        placeholder="Example: Create a nostalgic visualization with warm colors that highlights emotional moments in the lyrics. Use nature imagery for chorus sections."
    )
    
    # Generate button
    generate_button = st.button("Generate Visualization", type="primary")
    
    # Output section
    st.header("Output")
    # This section will be populated with visualization results
        
    # Process when generate button is clicked
    if generate_button:
        try:
            with st.spinner("Processing files and generating visualization..."):
                # Define paths to sample files
                sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")
                lyrics_path = os.path.join(sample_dir, "Her Majesty (Remastered 2009) (128kbit_AAC).txt")
                audio_path = os.path.join(sample_dir, "Her Majesty (Remastered 2009) (128kbit_AAC).mp3")
                
                # Process lyrics and audio
                try:
                    # Process the files and store in session state for reuse
                    st.session_state.lyrics_data = process_lyrics_file(lyrics_path)
                    st.session_state.audio_data = process_audio_file(audio_path)
                except FileNotFoundError as e:
                    st.warning(f"Sample files not found: {str(e)}. Using minimal default data.")
                    # Create valid minimal data structures to prevent validation errors
                    st.session_state.lyrics_data = [
                        {"timestamp": "00:00:05", "text": "First line of lyrics"}, 
                        {"timestamp": "00:00:10", "text": "Second line of lyrics"}
                    ]
                    st.session_state.audio_data = {
                        "duration": 60,
                        "metadata": {"tempo": 120, "key": "C Major"}
                    }
                
                # Create and store the agent in session state
                st.session_state.agent = create_lyrics_visualizer_agent()
                
                # Get validated data from session state
                lyrics_data = st.session_state.lyrics_data
                audio_data = st.session_state.audio_data
                
                try:
                    # Format a more detailed initial message
                    initial_message = f"""Create a detailed visualization that captures the essence of the lyrics and audio. 
                    
                    I've already loaded the lyrics data with {len(lyrics_data)} lines and audio data with duration {audio_data.get('duration', 'unknown')} seconds.
                    
                    My requirements for the visualization are: {user_requirements if user_requirements else 'Create something visually appealing that matches the mood of the song.'}
                    
                    Please use the analyze_lyrics and analyze_audio tools with the data I've provided."""
                    
                    # Create the input for the agent
                    agent_input = {
                        "lyrics_data": lyrics_data,
                        "audio_data": audio_data,
                        "user_requirements": user_requirements,
                        "messages": [HumanMessage(content=initial_message)]
                    }
                    
                    # Check if debug mode is enabled
                    if debug_mode:
                        from src.components.visualizer import print_stream
                        
                        st.subheader("Agent Thought Process")
                        with st.expander("View agent's reasoning", expanded=True):
                            st.write("Running agent with streaming enabled...")
                            
                            # Use streaming and collect outputs
                            stream = st.session_state.agent.stream(agent_input)
                            
                            # Capture stream for display in Streamlit
                            thoughts = []
                            all_states = []
                            
                            # Create placeholder for real-time updates
                            thought_placeholder = st.empty()
                            
                            # Process stream in real-time
                            for i, s in enumerate(stream):
                                all_states.append(s)
                                if "messages" in s and s["messages"]:
                                    message = s["messages"][-1]
                                    if hasattr(message, "content"):
                                        thought_content = message.content
                                    else:
                                        thought_content = str(message)
                                    
                                    thoughts.append(thought_content)
                                    thought_placeholder.text_area(f"Step {i+1}", thought_content, height=150)
                            
                            # Get final result from the last state
                            result = all_states[-1] if all_states else None
                    else:
                        # Use regular non-streaming invoke
                        result = st.session_state.agent.invoke(agent_input)
                except Exception as e:
                    st.error(f"Agent encountered an error: {str(e)}")
                    # Create minimal result to prevent further errors
                    result = {
                        "visualization_type": "error",
                        "message": f"Failed to generate visualization: {str(e)}",
                        "scenes": [],
                        "style": {"color_palette": ["#FF0000", "#880000"], "typography": {"font": "Arial"}}
                    }
                
                # Display the visualization results first, before video creation
                st.subheader("Step 1: Generated Visualization")
                # Display initial visualization result without wrapping in an expander
                display_visualization(result)
                
                # Step 2: Create and display video
                st.subheader("Step 2: Creating Video")
                with st.spinner("Creating video from visualization and audio..."): 
                    try:
                        # Get the path to the sample audio file
                        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")
                        audio_path = os.path.join(sample_dir, "Her Majesty (Remastered 2009) (128kbit_AAC).mp3")
                        
                        # Check if audio file exists
                        if not os.path.exists(audio_path):
                            st.warning(f"Audio file not found at: {audio_path}")
                            st.info("Creating video with default background color only.")
                        
                        # Get the lyrics data from session state
                        lyrics_data = st.session_state.lyrics_data
                        
                        # Create video from visualization data
                        from src.components.visualizer import create_video_from_visualization, display_video_player
                        
                        # Create the video file
                        video_path = create_video_from_visualization(
                            visualization_data=result,
                            lyrics_data=lyrics_data,
                            audio_path=audio_path
                        )
                        
                        # Display the video player
                        display_video_player(video_path)
                        
                        # Store the video path in session state for future reference
                        st.session_state.last_video_path = video_path
                        
                    except Exception as e:
                        st.error(f"Error creating video: {str(e)}")
                        st.info(
                            "Troubleshooting tips:\n" 
                            "1. Make sure moviepy is correctly installed\n"
                            "2. Check if the audio file exists and is readable\n"
                            "3. Verify that the visualization data has the correct format"
                        )
                
        except Exception as e:
            error_msg = str(e)
            if "recursion limit" in error_msg.lower():
                st.error(
                    "Recursion limit reached. The visualization task is too complex for the agent to process. "  
                    "Try simplifying your requirements or using fewer lyrics lines."
                )
                # Show a more specific message with suggestions
                st.info(
                    "To fix this, you can:\n\n"
                    "1. Use shorter, more concise requirements\n"
                    "2. Install the latest version of LangGraph with: `pip install --upgrade langgraph`\n"
                    "3. Consider breaking down your request into smaller parts"
                )
            else:
                st.error(f"An error occurred: {error_msg}")

if __name__ == "__main__":
    main()
