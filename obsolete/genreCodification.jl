include("sql_client.jl")
# Using a function this way works because of Julia's package system.
query = """
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
allData = getDbData(query)
#print(allData)

function getMinMax(genre, columnName)
    min = 0
    max = 0
    for i in 1:length(allData.danceability)
        d = allData.danceability[i]
        if d > max
            max = d
        elseif d < min
            min = d
        else
            continue
        end
    end
    return min, max
end

function runAnalysis(song, weights, data)
    result = false
    dataRow = allData[1, :] # gives me the first row of the entire allData
    # Want to make this so instead of 1, it is the song.
    print(weights, dataRow.danceability)

    if dataRow.danceability in weights[1]:weights[2] # marche pas why ?
        return result
    end

    return result
end

lowerBound, upperBound = getMinMax("Electronic", "danceability")
runAnalysis("Into the Abyss of Perdition", [lowerBound, upperBound], "danceability")
