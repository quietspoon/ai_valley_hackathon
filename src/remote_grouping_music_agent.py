from music_agent import MusicAgent,LyricLine
from pydantic import BaseModel, Field
from prompts import GROUP_LYRICS_PROMPT
import asyncio 
from typing import List

class GroupsPhrases(BaseModel):
    """
    Represents a phrase in the lyrics.
    """
    groupedLines: List[List[str]] 



class RemoteGroupingMusicAgent(MusicAgent):
    async def group_lyrics(self):
        """
        Groups the lyrics into phrases based on the remote grouping algorithm.
        """
        lyrics_txt = "\n".join([line.lyric for line in self.lyrics_by_line])
        
        response = await self.client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": GROUP_LYRICS_PROMPT},
                {"role": "user", "content": lyrics_txt} 
            ],
            response_format=GroupsPhrases,
        )
        self.phrases = response.choices[0].message.parsed.groupedLines
        index = 0
        grouped_temp = []
        for phrase in self.phrases:
            l = len(phrase)
            grouping_needed = self.lyrics_by_line[index:index+l]
            lyrics_group = [lyric.lyric for lyric in grouping_needed]
            combined_lyric = " ".join(lyrics_group)
            group = LyricLine(combined_lyric, self.lyrics_by_line[0].start_time, self.lyrics_by_line[-1].end_time)
            grouped_temp.append(group)
            index += l
        self.grouped_lyrics = grouped_temp
        self.save_state()
        return self.grouped_lyrics

if __name__ == "__main__":
    # agent = RemoteGroupingMusicAgent("espresso")
    # agent.load_lyrics("./data/espresso.txt")
    # asyncio.run(agent.group_lyrics())
    music_agent = RemoteGroupingMusicAgent("espresso")
    music_agent.load_state()
    # asyncio.run(music_agent.generate_all_images())
    music_agent.generate_music_video()