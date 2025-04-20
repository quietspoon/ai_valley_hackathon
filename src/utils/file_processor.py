import re
import os
from typing import List, Dict, Any, Optional
import json

try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo
except ImportError:
    AudioSegment = None
    mediainfo = None


def parse_timestamp(timestamp: str) -> float:
    """Convert a timestamp string to seconds."""
    # Handle different timestamp formats (HH:MM:SS, MM:SS, SS.MS, etc.)
    if re.match(r'^\d+:\d+:\d+(\.\d+)?$', timestamp):  # HH:MM:SS format
        h, m, s = timestamp.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif re.match(r'^\d+:\d+(\.\d+)?$', timestamp):  # MM:SS format
        m, s = timestamp.split(':')
        return int(m) * 60 + float(s)
    elif re.match(r'^\d+(\.\d+)?$', timestamp):  # SS or SS.MS format
        return float(timestamp)
    else:
        raise ValueError(f"Unsupported timestamp format: {timestamp}")


def process_lyrics_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a lyrics/transcript file with timestamps and extract the content.
    
    Args:
        file_path: Path to the lyrics/transcript file.
        
    Returns:
        A list of dictionaries with timestamp and text information.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lyrics file not found: {file_path}")
    
    lyrics_data = []
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Process based on file format
    if file_extension == '.srt':  # SubRip format
        pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\n\d+\s+\d{2}:\d{2}:\d{2},\d{3}|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for _, start, end, text in matches:
            # Convert comma to period for milliseconds
            start = start.replace(',', '.')
            lyrics_data.append({
                "timestamp": start,
                "end_timestamp": end.replace(',', '.'),
                "text": text.strip()
            })
    
    elif file_extension == '.lrc':  # LRC format
        pattern = r'\[(\d{2}:\d{2}\.\d{2})\](.*?)(?=\[\d{2}:\d{2}\.\d{2}\]|\Z)'
        matches = re.findall(pattern, content)
        
        for timestamp, text in matches:
            lyrics_data.append({
                "timestamp": timestamp,
                "text": text.strip()
            })
    
    else:  # Default to simple timestamp format (assume [MM:SS] Text or MM:SS Text)
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try various timestamp formats
            timestamp_match = re.match(r'^\[?(\d{1,2}:\d{2}(:\d{2})?(\.\d+)?)\]?\s*(.*)', line)
            
            if timestamp_match:
                timestamp, _, _, text = timestamp_match.groups()
                lyrics_data.append({
                    "timestamp": timestamp,
                    "text": text.strip()
                })
            else:
                # If no timestamp, append to previous entry or create a new one
                if lyrics_data:
                    lyrics_data[-1]["text"] += f" {line}"
                else:
                    # For the first line without timestamp, create entry with 0:00
                    lyrics_data.append({
                        "timestamp": "00:00",
                        "text": line
                    })
    
    # Convert timestamp strings to seconds for easier processing
    for item in lyrics_data:
        item["timestamp_seconds"] = parse_timestamp(item["timestamp"])
        if "end_timestamp" in item:
            item["end_timestamp_seconds"] = parse_timestamp(item["end_timestamp"])
    
    # Sort by timestamp
    lyrics_data.sort(key=lambda x: x["timestamp_seconds"])
    
    return lyrics_data


def process_audio_file(file_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Process an audio file to extract metadata and analyze content.
    
    Args:
        file_path: Path to the audio file (mp3, wav, etc.)
        
    Returns:
        A dictionary with audio metadata and analysis, or None if file_path is None
    """
    if not file_path:
        return None
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check if pydub is available
    if AudioSegment is None:
        print("Warning: pydub not available. Limited audio processing functionality.")
        # Return basic file info without audio processing
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "duration": None,
            "format": os.path.splitext(file_path)[1][1:],
        }
    
    # Process audio with pydub if available
    try:
        # Get audio information
        info = mediainfo(file_path)
        
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        
        # Extract basic metadata
        duration_seconds = len(audio) / 1000.0
        channels = audio.channels
        sample_width = audio.sample_width
        frame_rate = audio.frame_rate
        
        # Create segments for basic analysis (e.g., every 5 seconds)
        segment_duration = 5000  # 5 seconds in milliseconds
        segments = []
        
        for i in range(0, len(audio), segment_duration):
            segment = audio[i:i + segment_duration]
            if segment:
                # Calculate segment volume (dBFS)
                segment_data = {
                    "start_time": i / 1000.0,
                    "end_time": min((i + segment_duration) / 1000.0, duration_seconds),
                    "volume_dBFS": segment.dBFS,
                }
                segments.append(segment_data)
        
        # Construct the result dictionary
        audio_data = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "duration": duration_seconds,
            "channels": channels,
            "sample_width": sample_width,
            "frame_rate": frame_rate,
            "format": audio.frame_rate,
            "segments": segments,
            # Additional metadata from mediainfo
            "metadata": {k: v for k, v in info.items() if k not in ["filename", "filepath"]},
        }
        
        return audio_data
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        # Return basic file info in case of error
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "duration": None,
            "format": os.path.splitext(file_path)[1][1:],
            "error": str(e)
        }


if __name__ == "__main__":
    # Test function
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Processing file: {test_file}")
        
        if os.path.splitext(test_file)[1].lower() in ['.mp3', '.wav', '.ogg']:
            result = process_audio_file(test_file)
        else:
            result = process_lyrics_file(test_file)
            
        print(json.dumps(result, indent=2))
