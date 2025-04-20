import openai
import os
import dotenv
import json
import asyncio
import pydantic
import prompts
import requests
import moviepy.editor as mp
from datetime import datetime

from moviepy.config import change_settings
# Configure MoviePy to use your specific ImageMagick installation
change_settings({"IMAGEMAGICK_BINARY": r"D:\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})


def get_api_key():
    dotenv.load_dotenv()
    return os.getenv("OPENAI_API_KEY")

class ImageDescResponse(pydantic.BaseModel):
    image_desc: str


class LyricLine:
    def __init__(self, lyric, start_time, end_time, image_path = None):
        self.lyric = lyric
        self.start_time = start_time
        self.end_time = end_time
        self.image_path = image_path

    def __str__(self):
        return f"{self.lyric} ({self.start_time} - {self.end_time})"

    def to_dict(self):
        return {
            "lyric": self.lyric,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "image_path": self.image_path
        }

# This agent generates MV based on the lyrics given by the user. 
class MusicAgent:
    def __init__(self, song_name):
        self.song_name = song_name
        self.raw_lyrics_by_line = []
        self.lyrics_by_line = []
        self.grouped_lyrics = []
        self.music = None
        api_key = get_api_key()
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.base_path = f"data/{self.song_name}"
        os.makedirs(self.base_path, exist_ok=True)

    @staticmethod
    def serialize_to_file_name(text, max_length=100):
        """
        Sanitizes a string to be safely used as a filename.
        """
        import re
        safe = re.sub(r'[\\/*?:"<>|\'\n\r]', '_', text)
        return safe[:max_length]


    def _read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            self.raw_lyrics_by_line = file.readlines()
            return


    
    def convert_to_time(self, time_string):
        # Example input: "01:39.353"
        if time_string.startswith('[') and time_string.endswith(']'):
            time_string = time_string[1:-1]
        parts = time_string.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds, millis = map(float, parts[1].split('.')) if '.' in parts[1] else (float(parts[1]), 0.0)
            total_seconds = minutes * 60 + seconds + millis / 1000
            return total_seconds
        else:
            raise ValueError(f"Invalid time format: {time_string}")
    
    def _parse_lyrics(self):
        # extract all the time stamps
        timestamps = []
        for line in self.raw_lyrics_by_line:
            if "[" in line:
                timestamps.append(line.split("]")[0][1:])
        for index,line in enumerate(self.raw_lyrics_by_line[:-1]):
            rawLyric = line.split("]")[1].strip()
            # if the line is empty, skip it
            if rawLyric == "":
                continue
            newLyricLine = LyricLine(rawLyric, timestamps[index], timestamps[index+1])
            self.lyrics_by_line.append(newLyricLine)
        return self.lyrics_by_line

    def load_lyrics(self, file_path):
        self._read_file(file_path)
        parsed_lines = self._parse_lyrics()
        return parsed_lines

    def generate_mv_image(self):
        pass

    async def generate_image_based_on_description(self, description):
        response = await self.client.images.generate(
                model="dall-e-3",
                prompt=description,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        image_url = response.data[0].url
        return image_url
    

    def group_lyrics(self):
        # Mock, group each 4 lyrics into one lyrics
        grouped_lyrics = []
        for i in range(0, len(self.lyrics_by_line), 4):
            lyric_group = [x.lyric for x in self.lyrics_by_line[i:i+4]] 
            combined_lyric = " ".join(lyric_group)
            last_index = min(i+3, len(self.lyrics_by_line)-1)
            group = LyricLine(combined_lyric, self.lyrics_by_line[i].start_time, self.lyrics_by_line[last_index].end_time)
            grouped_lyrics.append(group)
        self.grouped_lyrics = grouped_lyrics
        return grouped_lyrics
        
    async def generate_image_description(self, lyric):
        response = await self.client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": prompts.IMAGE_DESCRIPTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": lyric,
                },
            ],
            response_format=ImageDescResponse,
        )
        return response.choices[0].message.parsed.image_desc
    
    def save_state(self):
        # Save the state of the music agent
        # Save the lyrics_by_line to a json file
        # Save the music to a json f
        with open(f"{self.base_path}/state.json", "w+", encoding="utf-8") as f:
            state = {
                "lyrics_by_line": [x.to_dict() for x in self.lyrics_by_line],
                "grouped_lyrics": [x.to_dict() for x in self.grouped_lyrics],
            }
            json.dump(state, f, ensure_ascii=False)
    
    def load_state(self):
        # Load the state of the music agent
        # Load the lyrics_by_line from a json file
        # Load the music from a json file
        with open(f"{self.base_path}/state.json", "r",encoding="utf-8") as f:
            state = json.load(f)
            self.lyrics_by_line = [LyricLine(x["lyric"], x["start_time"], x["end_time"]) for x in state["lyrics_by_line"]]
            self.grouped_lyrics = [LyricLine(x["lyric"], x["start_time"], x["end_time"]) for x in state["grouped_lyrics"]]


    async def generate_image_based_on_lyrics(self, lyric_line):
        lyric = lyric_line.lyric
        lyric_desc = await self.generate_image_description(lyric)
        lyric_image_url = await self.generate_image_based_on_description(lyric_desc)
        print(lyric_image_url)
        
        # Use the serialization function for the filename
        safe_lyric = self.serialize_to_file_name(lyric)
        image_path = os.path.join(self.base_path, f"{safe_lyric}.png")
        try:
            response = requests.get(lyric_image_url)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(response.content)
            lyric_line.image_path = image_path
            print(f"Image saved to {image_path}")
        except Exception as e:
            print(f"Failed to download image: {e}")
        return image_path
    
    async def generate_title_image(self):
        title_image_url = await self.generate_image_based_on_description(prompts.TITLE_IMAGE_PROMPT.format(title=self.song_name))
        print(title_image_url)
        # Use the serialization function for the filename
        image_path = os.path.join(self.base_path, "title.png")
        try:
            response = requests.get(title_image_url)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(response.content)
            self.tilte_image_path = image_path
            print(f"Image saved to {image_path}")
        except Exception as e:
            print(f"Failed to download image: {e}")
        return image_path

    async def generate_all_images(self):
        # Generate all images based on lyrics
        # Create the directory if it doesn't exist
        for lyric in self.grouped_lyrics:
            await self.generate_image_based_on_lyrics(lyric)
        self.save_state()

    def generate_music_video(self):
        """
        Generate the music video based on the lyrics
        Step One: Combine all images into a video based on grouped_lyrics
        Step Two: Combine all texts into a video based on lyrics_by_line
        Step Three: Overlay texts over images
        Step Four: Optionally add music
        """
        img_clips = []
        txt_clips = []

        # Calculate padding duration (in seconds)
        padding = self.convert_to_time(self.grouped_lyrics[0].start_time)
        # Add an empty image clip at the beginning to match the music delay
        if padding > 0:
            empty_img_path = "data/image/empty.png"
            if not os.path.exists(empty_img_path):
                os.makedirs(os.path.dirname(empty_img_path), exist_ok=True)
                from PIL import Image
                Image.new("RGB", (1024, 1024), color=(0, 0, 0)).save(empty_img_path)
            img_clips.append(mp.ImageClip(empty_img_path).set_duration(padding))

        # Step 1: Create image clips for each grouped lyric (with no offset)
        for lyric in self.grouped_lyrics:
            start = self.convert_to_time(lyric.start_time)
            end = self.convert_to_time(lyric.end_time)
            duration = max(0.1, end - start)
            # Use the serialization function for the filename
            safe_lyric = self.serialize_to_file_name(lyric.lyric)
            image_path = os.path.join(self.base_path, f"{safe_lyric}.png")
            img_clip = mp.ImageClip(image_path).set_duration(duration)
            img_clips.append(img_clip)

        # Step 2: Concatenate all image clips
        image_video = mp.concatenate_videoclips(img_clips, method="compose")

        # Step 3: Create text clips for each lyric line and position them at the correct time (with offset)
        # Add an empty text clip for the initial padding

        for lyric in self.lyrics_by_line:
            start = self.convert_to_time(lyric.start_time) + padding
            end = self.convert_to_time(lyric.end_time) + padding
            duration = max(0.1, end - start)
            txt_clip = (
                mp.TextClip(
                    lyric.lyric,
                    fontsize=48,
                    color='black',
                    font=r"C:\Windows\Fonts\msyh.ttc",  # Use Microsoft YaHei for Chinese support
                    method='caption',
                    size=image_video.size
                )
                .set_start(start)
                .set_duration(duration)
                .set_position('center')
            )
            txt_clip2 = (
                mp.TextClip(
                    lyric.lyric,
                    fontsize=54,
                    color='white',
                    font=r"C:\Windows\Fonts\msyh.ttc",  # Use Microsoft YaHei for Chinese support
                    method='caption',
                    size=image_video.size
                )
                .set_start(start)
                .set_duration(duration)
                .set_position('center')
            )
            
            txt_clips.append(txt_clip)
            txt_clips.append(txt_clip2)

        # Step 4: Overlay all text clips over the image video
        final_video = mp.CompositeVideoClip([image_video] + txt_clips)

        # Step 5: Optionally, add music if you have a music file
        music_path = os.path.join(self.base_path, "music.mp3")
        if os.path.exists(music_path):
            audio = mp.AudioFileClip(music_path)
            final_video = final_video.set_audio(audio)

        # Output path
        output_path = os.path.join(self.base_path, f"{self.song_name}_mv.mp4")
        final_video.write_videofile(output_path, fps=15, codec="libx264", audio_codec="aac")
        print(f"Music video saved to {output_path}")



if __name__ == "__main__":
    # music_agent = MusicAgent("dazhanhongtu")
    # lyrics = music_agent.load_lyrics("./data/dazhanhongtu.txt")
    # grouped_lyrics = music_agent.group_lyrics()
    # asyncio.run(music_agent.generate_all_images())
    music_agent = MusicAgent("dazhanhongtu")
    music_agent.load_state()
    music_agent.generate_music_video()
