import Foundation
import AgentSDK_Swift
import AVFoundation

/// Example showing how to use the voice agent capabilities
struct VoiceAgentExample {
    /// Run the example
    static func run() async throws {
        // Get API key from environment
        guard let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] else {
            print("Error: OPENAI_API_KEY environment variable not set")
            return
        }
        
        // Register OpenAI models
        await ModelProvider.shared.registerOpenAIModels(apiKey: apiKey)
        
        // 1. Create a simple agent
        let agent = Agent<Void>(
            name: "Voice Assistant",
            instructions: """
            You are a helpful voice assistant that responds to user queries.
            Keep your responses concise and conversational, as they will be spoken aloud.
            If asked about the weather, use the weather tool.
            """
        )
        
        // 2. Add a tool to the agent
        agent.addTool(
            Tool<Void>(
                name: "get_weather",
                description: "Get the current weather at a location",
                parameters: [
                    "location": .string(description: "The location to get the weather for")
                ],
                handler: { parameters, _ in
                    // This is a mock implementation
                    guard let location = parameters["location"] as? String else {
                        return "Cannot get weather without a location"
                    }
                    
                    let conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"]
                    let temperatures = [65, 70, 75, 80, 55, 60]
                    
                    let condition = conditions.randomElement() ?? "sunny"
                    let temperature = temperatures.randomElement() ?? 70
                    
                    return "The weather in \(location) is currently \(condition) with a temperature of \(temperature)°F"
                }
            )
        )
        
        // 3. Create the voice pipeline using OpenAI's services
        let pipeline = VoicePipeline(
            openAIApiKey: apiKey,
            agent: agent,
            whisperModel: "whisper-1",
            ttsModel: "tts-1",
            ttsVoice: "alloy"
        )
        
        // 4. Example of processing audio from a file
        print("Processing audio file...")
        if let audioURL = Bundle.main.url(forResource: "sample_input", withExtension: "wav"),
           let audioData = try? Data(contentsOf: audioURL) {
            
            // Process the audio through the pipeline
            let result = try await pipeline.process(
                audioData: audioData,
                context: (),
                progressHandler: { progress in
                    switch progress {
                    case .recognizingSpeech:
                        print("Recognizing speech...")
                    case .processingWithAgent(let transcription):
                        print("Transcription: \(transcription)")
                        print("Processing with agent...")
                    case .generatingSpeech(let response):
                        print("Agent response: \(response)")
                        print("Generating speech...")
                    }
                }
            )
            
            // Print the result
            print("Input transcription: \(result.inputTranscription)")
            print("Agent response: \(result.agentResponse)")
            print("Response audio size: \(result.audioResponse.count) bytes")
            print("Total processing time: \(result.metrics.totalDuration) seconds")
            
            // Play the audio response (optional)
            playAudio(result.audioResponse)
        } else {
            print("Example audio file not found. Proceeding with live recording...")
            
            // 5. Example of processing live audio
            print("Please speak after the beep...")
            
            // Play a beep sound
            playBeepSound()
            
            // Start recording and process the audio
            let listeningTask = pipeline.listen(
                context: (),
                progressHandler: { progress in
                    switch progress {
                    case .recognizingSpeech:
                        print("Recognizing speech...")
                    case .processingWithAgent(let transcription):
                        print("Transcription: \(transcription)")
                        print("Processing with agent...")
                    case .generatingSpeech(let response):
                        print("Agent response: \(response)")
                        print("Generating speech...")
                    }
                }
            ) { result in
                switch result {
                case .success(let pipelineResult):
                    print("Input transcription: \(pipelineResult.inputTranscription)")
                    print("Agent response: \(pipelineResult.agentResponse)")
                    print("Response audio size: \(pipelineResult.audioResponse.count) bytes")
                    print("Total processing time: \(pipelineResult.metrics.totalDuration) seconds")
                    
                    // Play the audio response
                    playAudio(pipelineResult.audioResponse)
                    
                case .failure(let error):
                    print("Error: \(error)")
                }
            }
            
            // Wait for the listening task to complete
            try await listeningTask.value
        }
        
        print("Example completed")
    }
    
    /// Play audio data through the device's speakers
    private static func playAudio(_ audioData: Data) {
        do {
            // Create a temporary file for the audio
            let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("response.wav")
            try audioData.write(to: tempURL)
            
            // Configure audio session for playback
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
            
            // Create and configure audio player
            let player = try AVAudioPlayer(contentsOf: tempURL)
            player.prepareToPlay()
            
            // Play the audio
            player.play()
            
            // Wait for playback to finish
            while player.isPlaying {
                usleep(100000) // 0.1 seconds
            }
            
            // Clean up
            try audioSession.setActive(false)
        } catch {
            print("Error playing audio: \(error)")
        }
    }
    
    /// Play a beep sound to indicate recording start
    private static func playBeepSound() {
        do {
            // Configure audio session
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
            
            // Generate a simple beep
            let sampleRate = 44100.0
            let duration = 0.5
            let numSamples = Int(duration * sampleRate)
            let frequency = 880.0 // A5 note
            
            var audioData = Data(capacity: numSamples * 2)
            
            for i in 0..<numSamples {
                let t = Double(i) / sampleRate
                let amplitude = 0.5
                let sample = Int16(amplitude * 32767.0 * sin(2.0 * .pi * frequency * t))
                
                // Convert to little-endian and append to data
                var littleEndianSample = CFSwapInt16HostToLittle(UInt16(bitPattern: sample))
                audioData.append(UnsafeBufferPointer(start: &littleEndianSample, count: 1))
            }
            
            // Create WAV header
            var header = Data()
            
            // RIFF chunk
            header.append("RIFF".data(using: .ascii)!)
            var fileLength = UInt32(audioData.count + 36).littleEndian
            header.append(UnsafeBufferPointer(start: &fileLength, count: 1))
            header.append("WAVE".data(using: .ascii)!)
            
            // fmt chunk
            header.append("fmt ".data(using: .ascii)!)
            var fmtLength = UInt32(16).littleEndian
            header.append(UnsafeBufferPointer(start: &fmtLength, count: 1))
            var audioFormat = UInt16(1).littleEndian // PCM
            header.append(UnsafeBufferPointer(start: &audioFormat, count: 1))
            var numChannels = UInt16(1).littleEndian // Mono
            header.append(UnsafeBufferPointer(start: &numChannels, count: 1))
            var sampleRateInt = UInt32(sampleRate).littleEndian
            header.append(UnsafeBufferPointer(start: &sampleRateInt, count: 1))
            var byteRate = UInt32(sampleRate * 2).littleEndian // 2 bytes per sample
            header.append(UnsafeBufferPointer(start: &byteRate, count: 1))
            var blockAlign = UInt16(2).littleEndian // 2 bytes per sample
            header.append(UnsafeBufferPointer(start: &blockAlign, count: 1))
            var bitsPerSample = UInt16(16).littleEndian
            header.append(UnsafeBufferPointer(start: &bitsPerSample, count: 1))
            
            // data chunk
            header.append("data".data(using: .ascii)!)
            var dataLength = UInt32(audioData.count).littleEndian
            header.append(UnsafeBufferPointer(start: &dataLength, count: 1))
            
            // Combine header and audio data
            var wavData = header
            wavData.append(audioData)
            
            // Save to temporary file
            let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("beep.wav")
            try wavData.write(to: tempURL)
            
            // Play the beep
            let player = try AVAudioPlayer(contentsOf: tempURL)
            player.prepareToPlay()
            player.play()
            
            // Wait for playback to finish
            while player.isPlaying {
                usleep(100000) // 0.1 seconds
            }
        } catch {
            print("Error playing beep sound: \(error)")
        }
    }
}

// Entry point for example
@main
struct VoiceAgentExampleApp {
    static func main() async throws {
        try await VoiceAgentExample.run()
    }
} 