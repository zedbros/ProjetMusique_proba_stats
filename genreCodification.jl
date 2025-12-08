include("sql_client.jl")
# Using a function this way works because of Julia's package system.
allData = getDbData()
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

    if dataRow.danceability in weights
        result = true
    end

    return result
end

lowerBound, upperBound = getMinMax("Electronic", "danceability")
runAnalysis("Into the Abyss of Perdition", [lowerBound, upperBound], "danceability")
