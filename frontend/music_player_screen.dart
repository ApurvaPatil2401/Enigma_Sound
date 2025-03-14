import 'dart:async';
import 'dart:convert';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:http/http.dart' as http;
import 'package:google_fonts/google_fonts.dart';

class MusicPlayerScreen extends StatefulWidget {
  final String emotion;
  final int randomNumber;

  const MusicPlayerScreen({super.key, required this.emotion, required this.randomNumber});


  @override
  _MusicPlayerScreenState createState() => _MusicPlayerScreenState();
}

class _MusicPlayerScreenState extends State<MusicPlayerScreen> {
  late AudioPlayer _audioPlayer;
  bool _isPlaying = false;
  bool _isLoading = true;
  String? _musicUrl;

  @override
  void initState() {
    super.initState();
    _audioPlayer = AudioPlayer();
    generateMusic(widget.emotion);
  }

  Future<void> generateMusic(String emotion) async {
    if (emotion.isEmpty || emotion == 'Unknown') {
      setState(() => _isLoading = false);
      return;
    }
    try {
      final response = await http.post(
        Uri.parse('http://192.168.0.104:5000/generate_music'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'detected_emotion': emotion}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _musicUrl = data['music_url'];
          _isLoading = false;
        });
      } else {
        setState(() => _isLoading = false);
      }
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _playGeneratedMusic() async {
    if (_musicUrl != null) {
      await _audioPlayer.stop();
      await _audioPlayer.setSourceUrl(_musicUrl!);
      await _audioPlayer.setReleaseMode(ReleaseMode.loop);
      await _audioPlayer.resume();
      setState(() => _isPlaying = true);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No music available to play')));
    }
  }

  Future<void> _launchSpotify(String emotion) async {
    final Uri url = Uri.parse('https://open.spotify.com/search/$emotion');
    if (await canLaunchUrl(url)) {
      await _audioPlayer.stop();
      await launchUrl(url, mode: LaunchMode.externalApplication);
      setState(() => _isPlaying = false);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Could not open Spotify.')),
      );
    }
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Music for ${widget.emotion}',
          style: GoogleFonts.poppins(
            color: Colors.white, // Set title text color to white
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.purple[700],
      ),
      body: Stack(
        children: [
          // Background Image
          Positioned.fill(
            child: Image.asset(
              'assets/music_bg.jpg', // Replace with your image path
              fit: BoxFit.cover,
              color: Colors.black.withOpacity(0.3),
              colorBlendMode: BlendMode.darken,
            ),
          ),

          // Content
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_isLoading)
                const CircularProgressIndicator()
              else ...[
                if (_musicUrl != null)
                  _buildMusicTile(
                    icon: Icons.play_circle_fill,
                    title: "Generated Music",
                    subtitle: _isPlaying ? "Playing..." : "Tap to play generated music",
                    onTap: _playGeneratedMusic,
                  ),
                _buildMusicTile(
                  icon: Icons.music_note,
                  title: "Spotify Recommendations",
                  subtitle: "Discover songs for your mood",
                  onTap: () => _launchSpotify(widget.emotion),
                ),
                if (!_isPlaying && _musicUrl == null)
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Text(
                      'No music available',
                      style: GoogleFonts.poppins(fontSize: 18, color: Colors.red),
                    ),
                  ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMusicTile({required IconData icon, required String title, required String subtitle, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        margin: const EdgeInsets.symmetric(vertical: 10, horizontal: 15),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(15),
          gradient: LinearGradient(
            colors: [Colors.purple.shade700, Colors.purple.shade400],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          boxShadow: [
            BoxShadow(
                color: Colors.black26, blurRadius: 6, offset: Offset(2, 4))
          ],
        ),
        child: Row(
          children: [
            Icon(icon, color: Colors.white, size: 40),
            const SizedBox(width: 20),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: GoogleFonts.poppins(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    subtitle,
                    style: GoogleFonts.poppins(
                        fontSize: 14, color: Colors.white70),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
