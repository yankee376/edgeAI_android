# LiteRT-LM and imported voice questions

The 640x640 app keeps its YOLO `.tflite` in `YOLOView` and uses LiteRT-LM as a separate Android runtime for text generation. The first prototype loads **Gemma 3 1B IT INT4** on CPU only when requested; it does not run LLM inference for camera frames.

1. Download `gemma3-1b-it-int4.litertlm` from the [LiteRT Community Gemma 3 1B model page](https://huggingface.co/litert-community/Gemma3-1B-IT). Accept the model terms on Hugging Face if prompted.
2. Launch the 640x640 app and tap the chat icon.
3. Tap **Chọn và nạp model**, then choose the downloaded `.litertlm` file in Android's file picker. The app copies it into its private files directory and remembers it for the next launch.
4. Select a detected class (optional), tap **Chọn file giọng nói để hỏi**, and choose a short audio or video recording from Downloads. MP4/M4A/3GP and common audio formats are decoded on-device. Vosk converts the recording to Vietnamese text offline; Gemma receives that transcript together with YOLO's latest class, confidence and normalized box location. There is no text-entry flow.
5. The app reports audio decoding, Vosk recognition and Gemma generation times separately. It reads the answer aloud only when Android has an offline Vietnamese TTS voice installed; otherwise it shows Gemma's text answer and explains how to add the voice.

The Gemma model is intentionally not bundled into the APK. It is hundreds of megabytes and requires separate provisioning. The compact Vietnamese Vosk model is bundled so file transcription works without depending on the handset's speech-recognition service or network. Audio decoding and Vosk inference run sequentially with Gemma on the app's background executor while the camera/YOLO loop stays active. Android TTS still depends on an installed offline Vietnamese voice, and ASR/Gemma may temporarily reduce camera FPS due to CPU contention.
