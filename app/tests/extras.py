# @app.post("/voice", response_model=TranscriptionResponse)
# async def handle_voice(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     session_id: Optional[str] = Form(None)
# ):
#     """Process voice input and return transcribed text."""
#     chatbot, session_id = get_chatbot(session_id)
    
#     # Save uploaded file to temporary location
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     try:
#         temp_file.write(await file.read())
#         temp_file.close()
        
#         # Process the voice file
#         transcribed_text = process_voice_input(
#             groq_client=chatbot.groq_client,
#             audio_file_path=temp_file.name,
#             model=TRANSCRIPTION_MODEL
#         )
        
#         if not transcribed_text:
#             raise HTTPException(status_code=400, detail="Failed to transcribe audio")
            
#         # Delete temp file in the background after response is sent
#         background_tasks.add_task(os.unlink, temp_file.name)

#         response = chatbot.process_query(transcribed_text)
#         return {
#             "text": transcribed_text,
#             "session_id": session_id,
#             "response" : response,
#         }
        
#     except Exception as e:
#         # Make sure to clean up temp file in case of an error
#         if os.path.exists(temp_file.name):
#             os.unlink(temp_file.name)
#         raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")

# ---------------------------------------------------------------------------------------------



# @app.post("/query_voice_direct")
# async def query_from_voice_direct(
#     background_tasks: BackgroundTasks,
#     file_path: str,  # Accept the file path directly
#     session_id: Optional[str] = None
# ):
#     """Process voice input from a direct file path and return both transcription and chatbot answer."""
#     chatbot, session_id = get_chatbot(session_id)

#     try:
#         # Process the voice file directly from the given path
#         transcribed_text = process_voice_input(
#             groq_client=chatbot.groq_client,
#             audio_file_path=file_path,
#             model=TRANSCRIPTION_MODEL
#         )

#         if not transcribed_text:
#             raise HTTPException(status_code=400, detail="Failed to transcribe audio")

#         # Process the transcribed text as a query
#         result = chatbot.process_query(transcribed_text)

#         return {
#             "transcription": transcribed_text,
#             "answer": result["answer"],
#             "sources": result["sources"],
#             "session_id": session_id
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing voice query: {str(e)}")

# ---------------------------------------------------------------------------------------------



# @app.post("/extract-embedding/")
# async def extract_embedding_from_voice(file: UploadFile = File(...)):
#     """
#     Extracts and returns the speaker embedding from an uploaded audio file.
#     """
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     # Debugging information
#     print(f"File saved at: {file_path}")
#     print(f"File exists: {os.path.exists(file_path)}")
    
#     try:
#         # Check if the file exists
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {file_path} does not exist.")

#         # Convert the file to WAV format
#         wav_file_path = convert_to_wav(file_path)
#         if wav_file_path is None:
#             raise ValueError("Failed to convert audio file to WAV format.")

#         # Extract embedding
#         embedding = extract_embedding(wav_file_path)
#         if embedding is None:
#             raise ValueError("Failed to extract embedding from the audio file.")

#         # Schedule file deletion after 5 minutes
#         threading.Thread(target=delete_file_after_delay, args=(file_path,)).start()
#         threading.Thread(target=delete_file_after_delay, args=(wav_file_path,)).start()

#         return {"embedding": embedding.tolist()}
#     except Exception as e:
#         # Clean up temp files in case of error
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         if wav_file_path and os.path.exists(wav_file_path):
#             os.remove(wav_file_path)
#         raise HTTPException(status_code=500, detail=f"Error extracting embedding: {str(e)}")
