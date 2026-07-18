import 'dart:async';
import 'dart:convert';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:http/http.dart' as http;
import 'package:google_fonts/google_fonts.dart';

import 'config.dart'; // Imports your AppConfig properties dynamically

class MusicPlayerScreen extends StatefulWidget {
  final String emotion;
  final String? audioUrl; // Optional parameter so separate modules don't break!
  final int randomNumber;

  const MusicPlayerScreen({
    super.key,
    required this.emotion,
    this.audioUrl, // Not required, defaults to null
    required this.randomNumber,
  });

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

    // Case 1: If the multimodal screen passed an audioUrl directly, use it!
    if (widget.audioUrl != null) {
      setState(() {
        _musicUrl = widget.audioUrl;
        _isLoading = false;
      });
      _playGeneratedMusic();
    } else {
      // Case 2: Fallback for individual modules (Face, Text, Audio)
      generateMusic(widget.emotion);
    }
  }

  // Fallback music generator utilizing the existing multimodal pipeline dynamically
  Future<void> generateMusic(String emotion) async {
    if (emotion.isEmpty || emotion == 'Unknown') {
      setState(() => _isLoading = false);
      return;
    }
    try {
      print("DEBUG FALLBACK: Triggering music production via AppConfig...");

      // Target the active base URL configuration cleanly
      String targetBase = 'http://192.168.1.6:5000';
      try {
        if (AppConfig.baseUrl.isNotEmpty) {
          targetBase = AppConfig.baseUrl;
        }
      } catch (_) {}

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$targetBase/detect-emotion-multimodal'),
      );

      request.fields['text'] = emotion;

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final decoded = json.decode(responseData);

      if (decoded['success'] == true && decoded['music_url'] != null) {
        setState(() {
          _musicUrl = decoded['music_url'];
          _isLoading = false;
        });

        // Auto-play the fallback track instantly
        _playGeneratedMusic();
      } else {
        setState(() => _isLoading = false);
      }
    } catch (e) {
      print("Error in fallback music generation: $e");
      setState(() => _isLoading = false);
    }
  }

  Future<void> _playGeneratedMusic() async {
    if (_musicUrl != null) {
      try {
        await _audioPlayer.stop();
        await _audioPlayer.setSourceUrl(_musicUrl!);
        await _audioPlayer.setReleaseMode(ReleaseMode.loop);
        await _audioPlayer.resume();
        setState(() => _isPlaying = true);
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Playback Error: $e')),
        );
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No music available to play')));
    }
  }

  Future<void> _launchSpotify(String emotion) async {
    final Uri url = Uri.parse('https://open.spotify.com/search/$emotion');
    if (await canLaunchUrl(url)) {
      await _audioPlayer.stop();
      setState(() => _isPlaying = false);
      await launchUrl(url, mode: LaunchMode.externalApplication);
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
            color: Colors.white,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.purple[700],
      ),
      body: Stack(
        children: [
          // Background Matrix Panel
          Positioned.fill(
            child: Container(
              color: Colors.black87,
            ),
          ),

          // Content Display Hierarchy
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_isLoading)
                const Center(child: CircularProgressIndicator())
              else ...[
                if (_musicUrl != null)
                  _buildMusicTile(
                    icon: _isPlaying ? Icons.pause_circle_filled : Icons.play_circle_fill,
                    title: "Generated Music",
                    subtitle: _isPlaying ? "Playing..." : "Tap to play generated music",
                    onTap: () async {
                      if (_isPlaying) {
                        await _audioPlayer.pause();
                        setState(() => _isPlaying = false);
                      } else {
                        await _playGeneratedMusic();
                      }
                    },
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
            const BoxShadow(color: Colors.black26, blurRadius: 6, offset: Offset(2, 4))
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
                    style: GoogleFonts.poppins(fontSize: 14, color: Colors.white70),
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