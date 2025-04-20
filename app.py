import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage  # Add this import
from src.agent.react_agent import create_lyrics_visualizer_agent
from src.utils.file_processor import process_lyrics_file, process_audio_file
from src.components.visualizer import display_visualization

# Load environment variables
load_dotenv()

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
        
        st.header("About")
        st.markdown("""
        This app uses an AI agent to analyze lyrics and audio, 
        then creates visuals that match the mood and meaning of the content.
        
        Currently using samples from Sabrina Carpenter's "Espresso".
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Display info about sample files
        st.info("Using sample files from the 'sample' directory:")
        st.markdown("- **Lyrics**: Sabrina Carpenter - Espresso -Official Video.txt")
        st.markdown("- **Audio**: Sabrina Carpenter - Espresso (Official Video) (128kbit_AAC).mp3")
        
        # User requirements input
        st.subheader("Visualization Requirements")
        user_requirements = st.text_area(
            "Describe what kind of visualization you want",
            height=150,
            placeholder="Example: Create a nostalgic visualization with warm colors that highlights emotional moments in the lyrics. Use nature imagery for chorus sections."
        )
        
        # Generate button
        generate_button = st.button("Generate Visualization", type="primary")
        
    with col2:
        st.header("Output")
        # This section will be populated with visualization results
        
    # Process when generate button is clicked
    if generate_button:
        try:
            with st.spinner("Processing files and generating visualization..."):
                # Define paths to sample files
                sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample")
                lyrics_path = os.path.join(sample_dir, "Sabrina Carpenter - Espresso -Official Video.txt")
                audio_path = os.path.join(sample_dir, "Sabrina Carpenter - Espresso (Official Video) (128kbit_AAC).mp3")
                
                # Process lyrics and audio
                lyrics_data = process_lyrics_file(lyrics_path)
                audio_data = process_audio_file(audio_path)
                
                # Create and run the agent
                agent = create_lyrics_visualizer_agent()
                
                # Split the agent invocation and add a shorter user request to avoid excessive recursion
                simplified_requirements = user_requirements[:200] if user_requirements else "Create a basic visualization"
                result = agent.invoke({
                    "lyrics_data": lyrics_data[:10],  # Use fewer lyrics lines to reduce complexity
                    "audio_data": audio_data,
                    "user_requirements": simplified_requirements,
                    "messages": [HumanMessage(content="Create a simple visualization matching these requirements.")]
                })
                
                # Display result in the second column
                with col2:
                    display_visualization(result)
                    
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
