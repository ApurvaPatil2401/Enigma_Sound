import 'dart:math';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'music_player_screen.dart';

class TextInputScreen extends StatefulWidget {
  const TextInputScreen({super.key});

  @override
  _TextInputScreenState createState() => _TextInputScreenState();
}

class _TextInputScreenState extends State<TextInputScreen> {
  final TextEditingController _controller = TextEditingController();
  String _detectedEmotion = '';
  bool _isLoading = false;

  Future<void> _detectEmotionFromText() async {
    if (_controller.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter some text')),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final response = await http.post(
        Uri.parse('http://192.168.0.104:5000/detect-emotion-text'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'text': _controller.text}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        String detectedEmotion = data['emotion'] ?? 'Unknown';
        print("Detected Emotion: $detectedEmotion"); // Debugging step

        setState(() => _detectedEmotion = detectedEmotion);

        if (detectedEmotion.isNotEmpty && detectedEmotion != 'Unknown') {
          int randomNumber = Random().nextInt(1000);
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => MusicPlayerScreen(
                emotion: detectedEmotion,
                randomNumber: randomNumber,
              ),
            ),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No emotion detected. Try again!')),
          );
        }

        _controller.clear();
      } else {
        setState(() => _detectedEmotion = 'Error: Unable to detect emotion.');
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
      extendBodyBehindAppBar: true, // Allows background to extend behind AppBar
      appBar: AppBar(
        title: const Text(
          'Enter Text for Emotion Detection',
          style: TextStyle(color: Colors.white), // Set title color to white
        ),
        backgroundColor: Colors.transparent,
        elevation: 0, // Removes shadow for a modern look
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
                // Animated TextField
                Material(
                  elevation: 5,
                  borderRadius: BorderRadius.circular(12),
                  child: TextField(
                    controller: _controller,
                    style: const TextStyle(fontSize: 18, color: Colors.black),
                    decoration: InputDecoration(
                      hintText: 'Enter text here...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: Colors.white,
                      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // Animated Detect Emotion Button
                ElevatedButton(
                  onPressed: _detectEmotionFromText,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple.shade700,
                    padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                    textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 5,
                  ),
                  child: const Text(
                    'Detect Emotion',
                    style: TextStyle(color: Colors.white), // Set text color to white
                  ),
                ),
                const SizedBox(height: 20),

                // Loading Indicator
                if (_isLoading) const CircularProgressIndicator(),

                // Emotion Display with Fade-In Effect
                if (_detectedEmotion.isNotEmpty && !_isLoading)
                  AnimatedOpacity(
                    opacity: 1.0,
                    duration: const Duration(milliseconds: 500),
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        'Detected Emotion: $_detectedEmotion',
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
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
