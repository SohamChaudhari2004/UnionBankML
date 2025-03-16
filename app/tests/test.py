import asyncio
from edge_tts import Communicate
import os

async def main():
    text = "Hello, this is a test of Edge-TTS"
    voice = "en-US-JennyNeural"
    
    communicate = Communicate(text, voice)
    await communicate.save("temp.mp3")

    os.system("start temp.mp3")  # Windows: Use 'afplay' for MacOS, 'xdg-open' for Linux

asyncio.run(main())