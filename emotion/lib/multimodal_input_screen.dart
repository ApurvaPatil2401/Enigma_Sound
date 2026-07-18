// multimodal_input_screen.dart
import 'dart:math';
import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:google_fonts/google_fonts.dart';

import 'config.dart';
import 'music_player_screen.dart';

class MultimodalInputScreen extends StatefulWidget {
  const MultimodalInputScreen({super.key});

  @override
  State<MultimodalInputScreen> createState() => _MultimodalInputScreenState();
}

class _MultimodalInputScreenState extends State<MultimodalInputScreen> {
  // Input Controllers & States
  final TextEditingController _textController = TextEditingController();
  CameraController? _cameraController;
  Future<void>? _initializeControllerFuture;
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();

  bool _isRecordingAudio = false;
  bool _isLoading = false;
  String? _audioPath;
  XFile? _capturedImage;
  bool _permissionsGranted = false;

  @override
  void initState() {
    super.initState();
    _initPermissionsAndHardware();
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _recorder.closeRecorder();
    _textController.dispose();
    super.dispose();
  }

  Future<void> _initPermissionsAndHardware() async {
    final statuses = await [
      Permission.camera,
      Permission.microphone
    ].request();

    if (statuses[Permission.camera]!.isGranted && statuses[Permission.microphone]!.isGranted) {
      setState(() => _permissionsGranted = true);
      await _initCamera();
      await _initAudioRecorder();
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Camera and Microphone permissions are required.')),
      );
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final frontCam = cameras.firstWhere(
            (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      _cameraController = CameraController(frontCam, ResolutionPreset.medium, enableAudio: false);
      _initializeControllerFuture = _cameraController!.initialize();
      setState(() {});
    } catch (e) {
      print("Camera init error: $e");
    }
  }

  Future<void> _initAudioRecorder() async {
    try {
      await _recorder.openRecorder();
      final tempDir = await getTemporaryDirectory();
      _audioPath = "${tempDir.path}/multimodal_audio.wav";
    } catch (e) {
      print("Audio init error: $e");
    }
  }

  Future<void> _toggleAudioRecording() async {
    if (_isRecordingAudio) {
      await _recorder.stopRecorder();
      setState(() => _isRecordingAudio = false);
    } else {
      if (_audioPath != null) {
        await _recorder.startRecorder(toFile: _audioPath, codec: Codec.pcm16WAV);
        setState(() => _isRecordingAudio = true);
      }
    }
  }

  Future<void> _processMultimodalSubmission() async {
    setState(() => _isLoading = true);

    try {
      // 1. Capture image framework if active camera preview is live
      if (_cameraController != null && _cameraController!.value.isInitialized && _capturedImage == null) {
        await _initializeControllerFuture;
        _capturedImage = await _cameraController!.takePicture();
      }

      // 2. Prepare Multipart payload request
      final request = http.MultipartRequest(
        'POST',
        Uri.parse(AppConfig.multimodalEndpoint),
      );

      // Attach text if present
      if (_textController.text.trim().isNotEmpty) {
        request.fields['text'] = _textController.text.trim();
      }

      // Attach image file
      if (_capturedImage != null) {
        request.files.add(await http.MultipartFile.fromPath('image', _capturedImage!.path));
      }

      // Attach audio file
      if (_audioPath != null && File(_audioPath!).existsSync() && !_isRecordingAudio) {
        request.files.add(await http.MultipartFile.fromPath('audio', _audioPath!));
      }

      // 3. Dispatch to server
      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final decoded = json.decode(responseData);

      // Extract 'dominant_emotion' instead of 'emotion'
      String finalEmotion = decoded['dominant_emotion'] ?? 'Unknown';

      // Extract 'music_url' directly from the payload response
      String? completeMusicUrl = decoded['music_url'];

      if (decoded['success'] == true && finalEmotion != 'Unknown' && completeMusicUrl != null) {
        print("DEBUG MULTIMODAL: Final Emotion -> $finalEmotion");
        print("DEBUG MULTIMODAL: Streaming target URL -> $completeMusicUrl");

        // Clear references for next iteration round
        setState(() {
          _capturedImage = null;
          _textController.clear();
        });

        // 4. Pass the parameters down safely to the player screen
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => MusicPlayerScreen(
                emotion: finalEmotion,
                audioUrl: completeMusicUrl,
                randomNumber: Random().nextInt(1000)
            ),
          ),
        );
      } else {
        String errorMsg = decoded['error'] ?? 'Could not definitively calculate emotion.';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Server Error: $errorMsg')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Submission Error: $e')),
      );
    } // The missing catch block
    finally {
      setState(() => _isLoading = false);
    } // The missing finally block
  }

  @override
  Widget build(BuildContext context) {
    if (!_permissionsGranted) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Multimodal AI Input', style: GoogleFonts.poppins(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: _isLoading
          ? const Center(child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [CircularProgressIndicator(), SizedBox(height: 16), Text("Analyzing cross-modal cues...")],
      ))
          : SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // SECTION 1: Camera Frame Preview
            Text('1. Show Your Face Expression', style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Container(
              height: MediaQuery.of(context).size.height * 0.3,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                color: Colors.black12,
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: _capturedImage != null
                    ? Image.file(File(_capturedImage!.path), fit: BoxFit.cover)
                    : (_cameraController != null && _initializeControllerFuture != null)
                    ? FutureBuilder<void>(
                  future: _initializeControllerFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.done) {
                      return CameraPreview(_cameraController!);
                    }
                    return const Center(child: CircularProgressIndicator());
                  },
                )
                    : const Center(child: Text('Camera Loading...')),
              ),
            ),
            if (_capturedImage != null)
              TextButton.icon(
                onPressed: () => setState(() => _capturedImage = null),
                icon: const Icon(Icons.refresh),
                label: const Text("Retake Photo"),
              ),
            const SizedBox(height: 20),

            // SECTION 2: Audio Recording Input
            Text('2. Say Something (Voice Mood)', style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: _toggleAudioRecording,
              style: ElevatedButton.styleFrom(
                backgroundColor: _isRecordingAudio ? Colors.redAccent : Colors.deepPurpleAccent,
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
              icon: Icon(_isRecordingAudio ? Icons.stop : Icons.mic, color: Colors.white),
              label: Text(
                _isRecordingAudio ? 'Stop Voice Recording' : 'Record Voice Sample',
                style: const TextStyle(color: Colors.white),
              ),
            ),
            const SizedBox(height: 20),

            // SECTION 3: Text Input Context
            Text('3. Describe What You Are Feeling', style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            TextField(
              controller: _textController,
              maxLines: 3,
              decoration: InputDecoration(
                hintText: "Type out your current thoughts...",
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                filled: true,
                fillColor: Colors.grey.shade100,
              ),
            ),
            const SizedBox(height: 32),

            // SUBMIT ALL MODALITIES TOGETHER
            ElevatedButton(
              onPressed: _processMultimodalSubmission,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green.shade700,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              child: Text(
                'Analyze Complete Mood',
                style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
              ),
            ),
          ],
        ),
      ),
    );
  }
}