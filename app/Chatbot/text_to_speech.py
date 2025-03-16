import asyncio
from edge_tts import Communicate
import os

async def generate_tts(text: str, voice: str, file_name: str = "output.mp3"):
    
    
    communicate = Communicate(text, voice)
    await communicate.save("temp1.mp3")

    # os.system("start temp1.mp3")  # Windows: Use 'afplay' for MacOS, 'xdg-open' for Linux

# asyncio.run(main())
