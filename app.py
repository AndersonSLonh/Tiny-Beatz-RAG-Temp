import streamlit as st
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import RAG_py
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Tiny Beatz",
    page_icon="",
    layout="wide"
)

# Title
st.title("Tiny Beatz - AI Music Recommender")
st.markdown("### Tell me how you're feeling...")

with st.sidebar:
    st.title("🎧 Spotify Connection")
    
    # Test Spotify API
    if st.button("🔌 Test Spotify API"):
        with st.spinner("Testing connection..."):
            try:
                # Get credentials from .env
                client_id = os.getenv("SPOTIFY_CLIENT_ID")
                client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
                
                if not client_id or not client_secret:
                    st.error("No credentials found in .env file!")
                else:
                    # Initialize Spotify client
                    auth_manager = SpotifyClientCredentials(
                        client_id=client_id,
                        client_secret=client_secret
                    )
                    sp = spotipy.Spotify(auth_manager=auth_manager)
                    
                    # Test search
                    results = sp.search(q="happy", type="track", limit=3)
                    
                    st.success("Connected to Spotify!")
                    st.markdown("**Test Search Results:**")
                    
                    for idx, track in enumerate(results['tracks']['items'], 1):
                        st.markdown(f"{idx}. **{track['name']}** by {track['artists'][0]['name']}")
                    
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
    
    st.divider()
    st.markdown("**Status:**")
    if os.getenv("SPOTIFY_CLIENT_ID"):
        st.info("Credentials loaded")
    else:
        st.warning("No credentials found")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How are you vibing/feeling today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your emotions..."):
            time.sleep(5)

    numGenres = 3
    genres, scores = RAG_py.predict_genre(prompt, numGenres)
    recommended_tracks = []
    if(not genres or scores[0] != 1): 
        RAG_py.spotify_genres.append(prompt) 
        genres, scores = RAG_py.predict_genre(prompt, (numGenres + 1))
    for g in genres:
        recommended_tracks.extend(RAG_py.get_songs_by_genre(g, limit=2))
    if(not recommended_tracks):
        st.markdown("I am sorry, I couldn't find a matching genre in spotify. Please try again with different genres or press Enter to exit: ") 

    # AI response
    k = 0
    for track in recommended_tracks:
        response = f"Match {k+1}: [Name: {track['name']} by {track['artist']}]({track['url']})"
        k += 1
        st.markdown(response)
        st.image(track['image'], width=150)
        st.session_state.messages.append({"role": "assistant", "content": response})
    # Add to chat history
    #st.session_state.messages.append({"role": "assistant", "content": response})

# Welcome message
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("Welcome! Tell me how you're vibing/feeling and I'll recommend music!")
