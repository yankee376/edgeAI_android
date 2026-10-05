package com.example.android_ai_gateway

import android.app.Activity
import android.content.Intent
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipInputStream
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {
    companion object {
        private const val CHANNEL = "edgeai/litertlm"
        private const val PICK_MODEL_REQUEST = 8126
        private const val PICK_AUDIO_REQUEST = 8128
        private const val PREFS = "litert_lm"
        private const val MODEL_PATH_KEY = "model_path"
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val llmExecutor = Executors.newSingleThreadExecutor()
    private var channel: MethodChannel? = null
    private var pendingPickerResult: MethodChannel.Result? = null
    private var pendingAudioPickerResult: MethodChannel.Result? = null
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var offlineVietnameseTts = false
    private var ttsResult: MethodChannel.Result? = null
    private var ttsUtteranceId: String? = null
    private var ttsStartedAtNs = 0L
    @Volatile private var engine: Engine? = null
    @Volatile private var modelPath: String? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
        channel?.setMethodCallHandler { call, result ->
            when (call.method) {
                "pickModel" -> pickModel(result)
                "getModelPath" -> result.success(savedModelFile()?.absolutePath)
                "loadModel" -> loadModel(call.argument<String>("path"), result)
                "generate" -> generate(call.argument<String>("prompt"), result)
                "pickAudioAndTranscribe" -> pickAudioAndTranscribe(result)
                "speak" -> speak(call.argument<String>("text"), result)
                "stopSpeaking" -> stopSpeaking(result)
                "voiceCapabilities" -> result.success(voiceCapabilities())
                else -> result.notImplemented()
            }
        }
        modelPath = savedModelFile()?.absolutePath
        initializeTextToSpeech()
    }

    private fun voiceCapabilities(): Map<String, Any> = mapOf(
        "offlineVietnameseTtsAvailable" to offlineVietnameseTts,
        "ttsReady" to ttsReady,
    )

    private fun initializeTextToSpeech() {
        tts = TextToSpeech(this) { status ->
            if (status != TextToSpeech.SUCCESS) return@TextToSpeech
            ttsReady = true
            val voice = tts?.voices
                ?.filter { it.locale.language == "vi" && !it.isNetworkConnectionRequired }
                ?.maxByOrNull { if (it.locale.country == "VN") 2 else 1 }
            if (voice != null) {
                tts?.voice = voice
                tts?.setSpeechRate(0.95f)
                offlineVietnameseTts = true
            }
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) = Unit

                override fun onDone(utteranceId: String?) {
                    mainHandler.post { finishTts(utteranceId, null) }
                }

                @Deprecated("Deprecated in Android API 21")
                override fun onError(utteranceId: String?) {
                    mainHandler.post { finishTts(utteranceId, "Không phát được giọng nói.") }
                }

                override fun onError(utteranceId: String?, errorCode: Int) {
                    mainHandler.post { finishTts(utteranceId, "TTS error $errorCode.") }
                }

                override fun onStop(utteranceId: String?, interrupted: Boolean) {
                    mainHandler.post {
                        finishTts(
                            utteranceId,
                            if (interrupted) "__stopped__" else null,
                        )
                    }
                }
            })
        }
    }

    private fun speak(text: String?, result: MethodChannel.Result) {
        val phrase = text?.trim().orEmpty()
        if (phrase.isEmpty()) {
            result.error("empty_speech", "Không có nội dung để đọc.", null)
            return
        }
        val speaker = tts
        if (!ttsReady || speaker == null) {
            result.error("tts_not_ready", "Đang khởi động bộ đọc giọng nói. Thử lại sau ít giây.", null)
            return
        }
        if (!offlineVietnameseTts) {
            result.error(
                "offline_vietnamese_tts_unavailable",
                "Máy chưa có giọng đọc tiếng Việt offline. Hãy cài giọng tiếng Việt offline trong cài đặt Text-to-speech của Android.",
                null,
            )
            return
        }
        if (ttsResult != null) {
            result.error("tts_busy", "Đang đọc câu trả lời trước.", null)
            return
        }
        val utteranceId = System.nanoTime().toString()
        ttsResult = result
        ttsUtteranceId = utteranceId
        ttsStartedAtNs = System.nanoTime()
        val status = speaker.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        if (status == TextToSpeech.ERROR) finishTts(utteranceId, "Không phát được âm thanh TTS.")
    }

    private fun stopSpeaking(result: MethodChannel.Result) {
        tts?.stop()
        result.success(null)
    }

    private fun finishTts(utteranceId: String?, error: String?) {
        if (utteranceId == null || utteranceId != ttsUtteranceId) return
        val result = ttsResult ?: return
        ttsResult = null
        ttsUtteranceId = null
        if (error == "__stopped__") {
            result.success(mapOf("stopped" to true))
        } else if (error != null) {
            result.error("speech_output_failed", error, null)
        } else {
            val elapsedMs = (System.nanoTime() - ttsStartedAtNs) / 1_000_000.0
            result.success(mapOf("playbackMs" to elapsedMs))
        }
    }

    private fun pickModel(result: MethodChannel.Result) {
        if (pendingPickerResult != null) {
            result.error("picker_busy", "Đang mở bộ chọn tệp model.", null)
            return
        }
        pendingPickerResult = result
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            startActivityForResult(intent, PICK_MODEL_REQUEST)
        } catch (error: Exception) {
            pendingPickerResult = null
            result.error("picker_failed", error.message, null)
        }
    }

    private fun pickAudioAndTranscribe(result: MethodChannel.Result) {
        if (pendingAudioPickerResult != null) {
            result.error("audio_picker_busy", "Đang xử lý một tệp âm thanh khác.", null)
            return
        }
        pendingAudioPickerResult = result
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/mp4", "video/3gpp"))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            startActivityForResult(intent, PICK_AUDIO_REQUEST)
        } catch (error: Exception) {
            pendingAudioPickerResult = null
            result.error("audio_picker_failed", error.message, null)
        }
    }

    @Deprecated("Uses the system document picker for Android model selection")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_AUDIO_REQUEST) {
            transcribePickedAudio(resultCode, data)
            return
        }
        if (requestCode != PICK_MODEL_REQUEST) return
        val result = pendingPickerResult ?: return
        pendingPickerResult = null
        if (resultCode != Activity.RESULT_OK || data?.data == null) {
            result.success(null)
            return
        }
        val uri: Uri = data.data!!
        llmExecutor.execute {
            try {
                val displayName = contentResolver.query(
                    uri,
                    arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                    null,
                    null,
                    null,
                )?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getString(0) else null
                }
                require(displayName?.endsWith(".litertlm", ignoreCase = true) == true) {
                    "Hãy chọn model có đuôi .litertlm (Gemma 3 1B IT INT4)."
                }
                engine?.close()
                engine = null
                val destination = File(filesDir, "gemma3-1b-it-int4.litertlm")
                contentResolver.openInputStream(uri).use { input ->
                    requireNotNull(input) { "Không đọc được tệp đã chọn." }
                    destination.outputStream().buffered().use { output -> input.copyTo(output) }
                }
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putString(MODEL_PATH_KEY, destination.absolutePath).apply()
                modelPath = destination.absolutePath
                mainHandler.post { result.success(destination.absolutePath) }
            } catch (error: Exception) {
                mainHandler.post {
                    result.error("model_copy_failed", error.message, null)
                }
            }
        }
    }

    private fun transcribePickedAudio(resultCode: Int, data: Intent?) {
        val result = pendingAudioPickerResult ?: return
        pendingAudioPickerResult = null
        val uri = data?.data
        if (resultCode != Activity.RESULT_OK || uri == null) {
            result.success(null)
            return
        }
        llmExecutor.execute {
            val totalStartedAt = System.nanoTime()
            val inputFile = File(cacheDir, "voice-question-${System.nanoTime()}.audio")
            try {
                val displayName = contentResolver.query(
                    uri,
                    arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                    null,
                    null,
                    null,
                )?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
                contentResolver.openInputStream(uri).use { input ->
                    requireNotNull(input) { "Không mở được tệp âm thanh đã chọn." }
                    inputFile.outputStream().buffered().use { output -> input.copyTo(output) }
                }
                val decoded = decodeAudioTo16kMono(inputFile)
                val decodedAt = System.nanoTime()
                require(decoded.samples.isNotEmpty()) { "Tệp không có track âm thanh hoặc không giải mã được." }
                val transcript = transcribePcm(decoded.samples)
                val finishedAt = System.nanoTime()
                mainHandler.post {
                    result.success(
                        mapOf(
                            "transcript" to transcript,
                            "fileName" to (displayName ?: "Tệp âm thanh"),
                            "decodeMs" to (decodedAt - totalStartedAt) / 1_000_000.0,
                            "recognitionMs" to (finishedAt - decodedAt) / 1_000_000.0,
                            "totalMs" to (finishedAt - totalStartedAt) / 1_000_000.0,
                            "audioDurationMs" to decoded.samples.size * 1000.0 / 16000.0,
                        ),
                    )
                }
            } catch (error: Exception) {
                mainHandler.post {
                    result.error(
                        "audio_transcription_failed",
                        error.message ?: "Không chuyển được file giọng nói thành văn bản.",
                        null,
                    )
                }
            } finally {
                inputFile.delete()
            }
        }
    }

    private data class DecodedAudio(val samples: ShortArray)

    private fun decodeAudioTo16kMono(file: File): DecodedAudio {
        val extractor = MediaExtractor()
        var decoder: MediaCodec? = null
        try {
            extractor.setDataSource(file.absolutePath)
            var audioFormat: MediaFormat? = null
            for (track in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(track)
                if (format.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                    audioFormat = format
                    extractor.selectTrack(track)
                    break
                }
            }
            val inputFormat = requireNotNull(audioFormat) { "Không tìm thấy track âm thanh trong tệp." }
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)
                ?: error("Không đọc được định dạng âm thanh.")
            if (inputFormat.containsKey(MediaFormat.KEY_DURATION)) {
                require(inputFormat.getLong(MediaFormat.KEY_DURATION) <= 60_000_000L) {
                    "File dài hơn 60 giây. Hãy chọn đoạn ghi âm ngắn hơn để giảm tải CPU."
                }
            }
            val inputRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val inputChannels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(inputFormat, null, null, 0)
            decoder.start()

            val output = ByteArrayOutputStream()
            val info = MediaCodec.BufferInfo()
            var inputEnded = false
            var outputEnded = false
            var outputChannels = inputChannels
            var outputEncoding = android.media.AudioFormat.ENCODING_PCM_16BIT
            var sourceFrameIndex = 0L
            var nextTargetFrame = 0.0
            var targetStep = inputRate / 16000.0
            while (!outputEnded) {
                if (!inputEnded) {
                    val inputIndex = decoder.dequeueInputBuffer(10_000)
                    if (inputIndex >= 0) {
                        val inputBuffer = decoder.getInputBuffer(inputIndex)
                            ?: error("Không lấy được buffer giải mã âm thanh.")
                        val size = extractor.readSampleData(inputBuffer, 0)
                        if (size < 0) {
                            decoder.queueInputBuffer(
                                inputIndex, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputEnded = true
                        } else {
                            decoder.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = decoder.dequeueOutputBuffer(info, 10_000)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val format = decoder.outputFormat
                        val outputRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        outputChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        targetStep = outputRate / 16000.0
                        outputEncoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                            format.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        } else {
                            android.media.AudioFormat.ENCODING_PCM_16BIT
                        }
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (outputIndex >= 0) {
                        val buffer = decoder.getOutputBuffer(outputIndex)
                        if (buffer != null && info.size > 0) {
                            buffer.position(info.offset)
                            buffer.limit(info.offset + info.size)
                            buffer.order(ByteOrder.LITTLE_ENDIAN)
                            val bytesPerSample = when (outputEncoding) {
                                android.media.AudioFormat.ENCODING_PCM_FLOAT -> 4
                                android.media.AudioFormat.ENCODING_PCM_8BIT -> 1
                                else -> 2
                            }
                            val frameBytes = bytesPerSample * outputChannels
                            while (buffer.remaining() >= frameBytes) {
                                var sum = 0f
                                repeat(outputChannels) {
                                    sum += when (outputEncoding) {
                                        android.media.AudioFormat.ENCODING_PCM_FLOAT -> buffer.float
                                        android.media.AudioFormat.ENCODING_PCM_8BIT ->
                                            ((buffer.get().toInt() and 0xff) - 128) / 128f
                                        else -> buffer.short / 32768f
                                    }
                                }
                                val mono = (sum / outputChannels).coerceIn(-1f, 1f)
                                while (sourceFrameIndex >= nextTargetFrame) {
                                    val sample = (mono * 32767f).toInt().coerceIn(-32768, 32767)
                                    output.write(sample and 0xff)
                                    output.write((sample shr 8) and 0xff)
                                    nextTargetFrame += targetStep
                                }
                                sourceFrameIndex++
                            }
                        }
                        outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        decoder.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }
            val pcm = output.toByteArray()
            val samples = ShortArray(pcm.size / 2) { index ->
                val low = pcm[index * 2].toInt() and 0xff
                val high = pcm[index * 2 + 1].toInt()
                ((high shl 8) or low).toShort()
            }
            return DecodedAudio(samples)
        } finally {
            try { decoder?.stop() } catch (_: Exception) { }
            decoder?.release()
            extractor.release()
        }
    }

    private fun transcribePcm(samples: ShortArray): String {
        val modelDirectory = File(filesDir, "vosk-model-small-vn-0.4")
        if (!File(modelDirectory, "am/final.mdl").isFile) {
            extractVoskModel(modelDirectory)
        }
        val model = Model(modelDirectory.absolutePath)
        val recognizer = Recognizer(model, 16000.0f)
        try {
            val transcripts = mutableListOf<String>()
            var offset = 0
            while (offset < samples.size) {
                val count = minOf(4000, samples.size - offset)
                if (recognizer.acceptWaveForm(samples.copyOfRange(offset, offset + count), count)) {
                    JSONObject(recognizer.result).optString("text")
                        .takeIf { it.isNotBlank() }
                        ?.let(transcripts::add)
                }
                offset += count
            }
            JSONObject(recognizer.finalResult).optString("text")
                .takeIf { it.isNotBlank() }
                ?.let(transcripts::add)
            return transcripts.joinToString(" ").trim().also {
                require(it.isNotBlank()) { "Không nhận ra lời nói trong file. Hãy chọn file rõ tiếng và nói ngắn hơn." }
            }
        } finally {
            recognizer.close()
            model.close()
        }
    }

    private fun extractVoskModel(destination: File) {
        destination.deleteRecursively()
        val root = destination.canonicalFile
        ZipInputStream(assets.open("flutter_assets/assets/models/vosk-model-small-vn-0.4.zip").buffered()).use { zip ->
            var entry = zip.nextEntry
            while (entry != null) {
                val relativePath = entry.name.removePrefix("vosk-model-small-vn-0.4/")
                if (relativePath.isNotEmpty()) {
                    val output = File(root, relativePath).canonicalFile
                    require(output.path.startsWith(root.path + File.separator)) { "Model ZIP chứa đường dẫn không hợp lệ." }
                    if (entry.isDirectory) output.mkdirs() else {
                        output.parentFile?.mkdirs()
                        output.outputStream().buffered().use { zip.copyTo(it) }
                    }
                }
                zip.closeEntry()
                entry = zip.nextEntry
            }
        }
        require(File(destination, "am/final.mdl").isFile) { "Không giải nén được model Vosk." }
    }

    private fun loadModel(requestedPath: String?, result: MethodChannel.Result) {
        val path = requestedPath ?: modelPath ?: savedModelFile()?.absolutePath
        if (path.isNullOrBlank() || !File(path).isFile) {
            result.error("model_missing", "Chưa chọn tệp .litertlm hợp lệ.", null)
            return
        }
        llmExecutor.execute {
            val clock = System.nanoTime()
            try {
                engine?.close()
                engine = null
                val next = Engine(
                    EngineConfig(
                        modelPath = path,
                        backend = Backend.CPU(),
                        cacheDir = cacheDir.absolutePath,
                    ),
                )
                next.initialize()
                engine = next
                modelPath = path
                val elapsedMs = (System.nanoTime() - clock) / 1_000_000.0
                mainHandler.post {
                    result.success(mapOf("path" to path, "loadMs" to elapsedMs, "backend" to "CPU"))
                }
            } catch (error: Exception) {
                engine?.close()
                engine = null
                mainHandler.post {
                    result.error("model_load_failed", error.message, null)
                }
            }
        }
    }

    private fun generate(prompt: String?, result: MethodChannel.Result) {
        val request = prompt?.trim().orEmpty()
        if (request.isEmpty()) {
            result.error("empty_prompt", "Hãy nhập câu hỏi trước.", null)
            return
        }
        llmExecutor.execute {
            val activeEngine = engine
            if (activeEngine == null) {
                mainHandler.post {
                    result.error("model_not_loaded", "Hãy nạp model trước khi hỏi.", null)
                }
                return@execute
            }
            val clock = System.nanoTime()
            try {
                val conversation = activeEngine.createConversation()
                val answer = try {
                    conversation.sendMessage(request).contents.contents
                        .filterIsInstance<Content.Text>()
                        .joinToString(separator = "") { it.text }
                } finally {
                    conversation.close()
                }
                val elapsedMs = (System.nanoTime() - clock) / 1_000_000.0
                mainHandler.post {
                    result.success(mapOf("answer" to answer, "generationMs" to elapsedMs))
                }
            } catch (error: Exception) {
                mainHandler.post {
                    result.error("generation_failed", error.message, null)
                }
            }
        }
    }

    private fun savedModelFile(): File? {
        val savedPath = getSharedPreferences(PREFS, MODE_PRIVATE)
            .getString(MODEL_PATH_KEY, null) ?: return null
        return File(savedPath).takeIf { it.isFile }
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        ttsResult?.error("activity_destroyed", "Đã đóng màn hình đọc câu trả lời.", null)
        ttsResult = null
        llmExecutor.execute {
            engine?.close()
            engine = null
        }
        llmExecutor.shutdown()
        super.onDestroy()
    }
}
