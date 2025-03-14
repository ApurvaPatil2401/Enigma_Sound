import 'dart:math';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:permission_handler/permission_handler.dart';
import 'music_player_screen.dart';

class FaceInputScreen extends StatefulWidget {
  const FaceInputScreen({super.key});

  @override
  _FaceInputScreenState createState() => _FaceInputScreenState();
}

class _FaceInputScreenState extends State<FaceInputScreen> {
  CameraController? _cameraController;
  Future<void>? _initializeControllerFuture;
  bool _isLoading = false;
  String _detectedEmotion = '';
  bool _permissionsGranted = false;
  CameraDescription? _currentCamera;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    final cameraStatus = await Permission.camera.request();
    if (cameraStatus.isGranted) {
      _permissionsGranted = true;
      await _initializeCamera(CameraLensDirection.front); // Default to front camera
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Camera permission is required')),
      );
    }
  }

  Future<void> _initializeCamera(CameraLensDirection lensDirection) async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        print('No cameras available');
        return;
      }

      // Select the camera based on the direction
      _currentCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == lensDirection,
        orElse: () => cameras.first, // Fallback to the first available camera
      );

      _cameraController = CameraController(_currentCamera!, ResolutionPreset.medium);
      _initializeControllerFuture = _cameraController!.initialize();
      setState(() {});
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  Future<void> _captureImage() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      print('Camera not initialized');
      return;
    }

    if (_isLoading) return; // Prevent multiple submissions

    setState(() {
      _isLoading = true;
    });

    try {
      await _initializeControllerFuture;
      final image = await _cameraController!.takePicture();

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://192.168.0.104:5000/detect-emotion-face'),
      );
      request.files.add(await http.MultipartFile.fromPath('image', image.path));

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final decodedResponse = json.decode(responseData);

      setState(() {
        _detectedEmotion = decodedResponse['emotion'] ?? 'Unknown';
      });

      if (_detectedEmotion.isNotEmpty) {
        int randomNumber = Random().nextInt(1000);
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => MusicPlayerScreen(emotion: _detectedEmotion, randomNumber: randomNumber),
          ),
        );
      }
    } catch (e) {
      print('Error capturing or processing image: $e');
      setState(() {
        _detectedEmotion = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _toggleCamera() {
    if (_currentCamera?.lensDirection == CameraLensDirection.front) {
      _initializeCamera(CameraLensDirection.back); // Switch to back camera
    } else {
      _initializeCamera(CameraLensDirection.front); // Switch to front camera
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Facial Expression Detection'),
        centerTitle: true,
        backgroundColor: Colors.purpleAccent,
      ),
      body: Center(
        child: _permissionsGranted
            ? Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Camera Preview with a frame
            if (_cameraController != null && _initializeControllerFuture != null)
              FutureBuilder<void>(
                future: _initializeControllerFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done) {
                    return ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: CameraPreview(_cameraController!),
                    );
                  } else {
                    return const Center(child: CircularProgressIndicator());
                  }
                },
              ),
            const SizedBox(height: 20),
            // Capture Image Button with a modern rounded design
            ElevatedButton(
              onPressed: _captureImage,
              style: ElevatedButton.styleFrom(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ), backgroundColor: Colors.purpleAccent,
                padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
              ),
              child: const Text('Capture Image', style: TextStyle(fontSize: 18)),
            ),
            const SizedBox(height: 20),
            // Camera Flip Button with more prominent design
            ElevatedButton(
              onPressed: _toggleCamera,
              style: ElevatedButton.styleFrom(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ), backgroundColor: Colors.pinkAccent,
                padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
              ),
              child: const Text('Flip Camera', style: TextStyle(fontSize: 18)),
            ),
            const SizedBox(height: 20),
            // Loading Indicator
            if (_isLoading) const CircularProgressIndicator(),
            // Detected Emotion Text with more emphasis
            if (_detectedEmotion.isNotEmpty && !_isLoading)
              Padding(
                padding: const EdgeInsets.all(10.0),
                child: Text(
                  'Detected Emotion: $_detectedEmotion',
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.blueAccent,
                  ),
                ),
              ),
          ],
        )
            : const Center(child: Text("Camera permission is required", style: TextStyle(fontSize: 18, color: Colors.red))),
      ),
    );
  }
}
