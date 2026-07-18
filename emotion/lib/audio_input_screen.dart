import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'config.dart';
import 'music_player_screen.dart';

class AudioInputScreen extends StatefulWidget {
  const AudioInputScreen({super.key});

  @override
  _AudioInputScreenState createState() => _AudioInputScreenState();
}

class _AudioInputScreenState extends State<AudioInputScreen> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;
  bool _isLoading = false;
  String _detectedEmotion = '';
  String? _filePath;

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  @override
  void dispose() {
    _recorder.closeRecorder();
    if (_filePath != null) {
      try {
        final file = File(_filePath!);
        if (file.existsSync()) {
          file.deleteSync();
        }
      } catch (e) {
        print('Error deleting file: $e');
      }
    }
    super.dispose();
  }

  Future<void> _initRecorder() async {
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Microphone permission is required')),
      );
      return;
    }

    try {
      await _recorder.openRecorder();
      final tempDir = await getTemporaryDirectory();
      _filePath = "${tempDir.path}/audio_sample.wav";
    } catch (e) {
      print('Error initializing recorder: $e');
    }
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (_filePath != null) {
      try {
        await _recorder.startRecorder(toFile: _filePath, codec: Codec.pcm16WAV);
        setState(() => _isRecording = true);
      } catch (e) {
        print('Error starting recording: $e');
      }
    }
  }

  Future<void> _stopRecording() async {
    try {
      await _recorder.stopRecorder();
      setState(() => _isRecording = false);
      _detectEmotionFromAudio();
    } catch (e) {
      print('Error stopping recording: $e');
    }
  }

  Future<void> _detectEmotionFromAudio() async {
    if (_filePath == null) return;
    setState(() => _isLoading = true);

    try {
      print("DEBUG AUDIO: Sending recording to multimodal engine pipeline...");

      // FIX 1: Routing audio payload through the working multimodal pipeline
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${AppConfig.baseUrl}/detect-emotion-multimodal'),
      );
      request.files.add(await http.MultipartFile.fromPath('audio', _filePath!));

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final decodedResponse = json.decode(responseData);

      // FIX 2: Mapping key names to dominant_emotion to match the pipeline backend
      String finalEmotion = decodedResponse['dominant_emotion'] ?? 'Unknown';
      String? completeMusicUrl = decodedResponse['music_url'];

      setState(() {
        _detectedEmotion = finalEmotion;
      });

      if (finalEmotion.isNotEmpty && finalEmotion != 'Unknown') {
        int randomNumber = Random().nextInt(1000);

        // FIX 3: Safely passing audioUrl down to prevent fallback loops
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => MusicPlayerScreen(
              emotion: finalEmotion,
              audioUrl: completeMusicUrl,
              randomNumber: randomNumber,
            ),
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not accurately process audio emotion.')),
        );
      }
    } catch (e) {
      setState(() => _detectedEmotion = 'Error: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: const Text('Record Your Audio'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.purple.shade800, Colors.blue.shade600],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Animated Record Button with color feedback
                ElevatedButton(
                  onPressed: _toggleRecording,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isRecording ? Colors.red : Colors.green,
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                    textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(15),
                    ),
                    elevation: 5,
                  ),
                  child: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
                ),
                const SizedBox(height: 30),

                // Loading indicator
                if (_isLoading) const CircularProgressIndicator(),

                // Emotion feedback message with fade-in animation
                if (_detectedEmotion.isNotEmpty && !_isLoading)
                  AnimatedOpacity(
                    opacity: 1.0,
                    duration: const Duration(milliseconds: 500),
                    child: Padding(
                      padding: const EdgeInsets.all(10.0),
                      child: Text(
                        'Detected Emotion: $_detectedEmotion',
                        style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),

                const SizedBox(height: 40),

                // Informational Text
                if (!_isRecording && _detectedEmotion.isEmpty)
                  const Text(
                    "Please record some audio to detect emotion.",
                    style: TextStyle(
                      fontSize: 18,
                      fontStyle: FontStyle.italic,
                      color: Colors.white,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}