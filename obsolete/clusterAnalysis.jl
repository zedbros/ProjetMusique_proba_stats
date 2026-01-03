include("sql_client.jl")

punkQuery = """
    SELECT 
        t.title AS track_title,
        a.name AS artist_name,
        g.title AS genre,
        e.acousticness,
        e.danceability,
        e.energy,
        e.instrumentalness,
        e.liveness,
        e.speechiness,
        e.tempo,
        e.valence
    FROM tracks t
    INNER JOIN echonest_features e ON t.track_id = e.track_id
    INNER JOIN artists a ON t.artist_id = a.artist_id
    INNER JOIN track_genres tg ON t.track_id = tg.track_id
    INNER JOIN genres g ON tg.genre_id = g.genre_id
    WHERE e.acousticness IS NOT NULL 
      AND e.danceability IS NOT NULL
      AND e.energy IS NOT NULL
      AND g.title = 'Punk'
    LIMIT 10;
"""

classicalQuery = """
    SELECT 
        t.title AS track_title,
        a.name AS artist_name,
        g.title AS genre,
        e.acousticness,
        e.danceability,
        e.energy,
        e.instrumentalness,
        e.liveness,
        e.speechiness,
        e.tempo,
        e.valence
    FROM tracks t
    INNER JOIN echonest_features e ON t.track_id = e.track_id
    INNER JOIN artists a ON t.artist_id = a.artist_id
    INNER JOIN track_genres tg ON t.track_id = tg.track_id
    INNER JOIN genres g ON tg.genre_id = g.genre_id
    WHERE e.acousticness IS NOT NULL 
      AND e.danceability IS NOT NULL
      AND e.energy IS NOT NULL
      AND g.title = 'Classical'
    LIMIT 10;
"""

rockQuery = """
    SELECT 
        t.title AS track_title,
        a.name AS artist_name,
        g.title AS genre,
        e.acousticness,
        e.danceability,
        e.energy,
        e.instrumentalness,
        e.liveness,
        e.speechiness,
        e.tempo,
        e.valence
    FROM tracks t
    INNER JOIN echonest_features e ON t.track_id = e.track_id
    INNER JOIN artists a ON t.artist_id = a.artist_id
    INNER JOIN track_genres tg ON t.track_id = tg.track_id
    INNER JOIN genres g ON tg.genre_id = g.genre_id
    WHERE e.acousticness IS NOT NULL 
      AND e.danceability IS NOT NULL
      AND e.energy IS NOT NULL
      AND g.title = 'Rock'
    LIMIT 11;
"""

# punkData = getDbData(punkQuery)
# classicalData = getDbData(classicalQuery)
rockData = getDbData(rockQuery)

print(rockData)

# Plots.scatter(punkData.danceability, )
