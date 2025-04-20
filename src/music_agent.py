import openai
import os
import dotenv
import json
import asyncio
import pydantic
import prompts
import requests
from datetime import datetime

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
            "end_time": self.end_time
        }

# This agent generates MV based on the lyrics given by the user. 
class MusicAgent:
    def __init__(self):
        self.raw_lyrics_by_line = []
        self.lyrics_by_line = []
        self.music = None

        api_key = get_api_key()
        self.client = openai.AsyncOpenAI(api_key=api_key)


    
    def _read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            self.raw_lyrics_by_line = file.readlines()
            print(self.raw_lyrics_by_line)
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
        self.lyrics_by_line = grouped_lyrics
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


    async def generate_image_based_on_lyrics(self, lyric_line, image_dir = "data/image"):
        lyric = lyric_line.lyric
        lyric_desc = await self.generate_image_description(lyric)
        lyric_image_url = await self.generate_image_based_on_description(lyric_desc)
        print(lyric_image_url)
        
        # Download the image to data/image
        image_path = os.path.join(image_dir, f"{lyric.replace(' ','_')}.png")
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

    async def generate_all_images(self):
        # Generate all images based on lyrics
        # Create the directory if it doesn't exist
        image_dir = "data/image/session_" + datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(image_dir, exist_ok=True)

        for lyric in self.lyrics_by_line:
            await self.generate_image_based_on_lyrics(lyric,image_dir)

    
    async def generate_music(self, lyric):
        pass


if __name__ == "__main__":
    music_agent = MusicAgent()
    lyrics = music_agent.load_lyrics("./data/dazhanhongtu.txt")
    grouped_lyrics = music_agent.group_lyrics()    
    asyncio.run(music_agent.generate_all_images())
