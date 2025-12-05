using MySQL
using DBInterface
using DataFrames

conn = DBInterface.connect(MySQL.Connection, "crossover.proxy.rlwy.net", "root", "tLeGwnVSAGVQLYbaMcdSJLunevRZRghw", db="railway", port=17837)

query1 = """
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
LIMIT 10;
"""


query = "
select title as AAAAAAAAAAAAAAAAAAAAAAAAAA
from tracks
limit 10;"

DBInterface.execute(conn, query)

# Have to write the data into a file just to test that the values selected
# in the query are correct.
