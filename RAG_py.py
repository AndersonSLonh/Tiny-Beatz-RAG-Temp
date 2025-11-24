import faiss
import spotipy
import os
import random
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials


## Retrieve User response
# Look through data, which is the spotify API
# Tools to use:
# Vector Search (FAISS (local) or ChromaDB), Find similar songs
# Faiss-cpu: A vector database library made by Facebook (Meta) = AI-memory
# Music Data - Spotify API
# Provide output based on data and user response matches


load_dotenv() # Load environment variables from .env file
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = 0.45

# Spotify API setup
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
    )
)

def get_songs_by_genre(genre, limit=20):
    results = sp.search(q=f"genre:{genre}", type="track", limit=50)
    tracks = results["tracks"]["items"]

    random.shuffle(tracks)
    tracks = tracks[:limit]

    tracks = sorted(tracks, key=lambda t: t["popularity"], reverse = True) # Get popularity from tracks and sort them largest to smallest

    return[{
        "name": t["name"],
        "artist": t["artists"][0]["name"],
        "url": t["external_urls"]["spotify"],
        "image": t["album"]["images"][0]["url"]
    } for t in tracks]

# Genre data base
spotify_genres = [
    "pop", "rock", "hip hop", "edm", "country", "jazz",
    "classical", "study", "sleep", "chill", "sad", "happy",
    "romance", "party", "metal", "r&b", "dance", "ambient"
]

# User input data base
userInputHistory = []
outputHistory = []


def predict_genre(user_text, k):
    song_vectors = model.encode(spotify_genres, convert_to_numpy=True).astype("float32") # Encoding data base labels into vectors 
    d = song_vectors.shape[1] # getting the dimension of the vectors
    index = faiss.IndexFlatIP(d) # making space for label vectors 

    # normalize so IP = cosine similarity
    faiss.normalize_L2(song_vectors) # scaling vector to unit length
    index.add(song_vectors) # storing vectors in index table

    user_vec = model.encode([user_text], convert_to_numpy=True).astype("float32") # Encoding user input into vector
    faiss.normalize_L2(user_vec) # scaling vector to unit length

    # ids is which vectors matched (their position)
    scores, ids = index.search(user_vec, k) # searching for top k similar vectors in index table
    # getting genre labels with similarity scores above threshold (0.75)
    
    top_genres = [ 
        spotify_genres[i] 
        for i, scores in zip(ids[0], scores[0])
        if scores >= SIMILARITY_THRESHOLD
    ]

    return top_genres, scores[0]

# THIS RUNS ON STREAMLIT!!!

# userInput = input("Welcome! Tell me how you're vibing/feeling and I'll recommend music! ")
# numGenres = 3
# while(userInput):
#     userInputHistory.append(userInput)
#     genres, scores = predict_genre(userInput, numGenres) # Calling predict_genre function that compares genre vectors to user input vector, then getting the genre and scores of it.
#     recommended_tracks = []
#     if(not genres or scores[0] != 1): # if no genres match above threshold
#         spotify_genres.append(userInput) # adding user input to genre database if its not already there
#         genres, scores = predict_genre(userInput, (numGenres + 1)) # (numGenres + 1) so you can get the previous genres that were recieved if there was one, and the new genre that was missing in data set
#     for g in genres:
#         recommended_tracks.extend(get_songs_by_genre(g, limit=5)) # getting limit = n songs for each genre 
#         k = 0 # Used to print matches 
#     if(not recommended_tracks):
#         userInput = input("I am sorry, I couldn't find a matching genre in spotify. Please try again with different genres or press Enter to exit: ")
#     else:
#         for i in recommended_tracks:
#             print(f"Match {k+1}: {i}")
#             k += 1
#             outputHistory.append(i)
#         userInput = input("You can enter another vibe/feeling or press Enter to exit: ")

# quitCommand = "Thank you for using Tiny Beatz! Enjoy your music!"
# print(quitCommand)