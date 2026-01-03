include("sql_client.jl")
using Plots, StatsBase

begin
    genreList = ["Rock", "Electronic", "Folk", "Pop"]
    queryList = []

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
        LIMIT 200;
        """

    electronicQuery = """
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
        AND g.title = 'Electronic'
        LIMIT 200;
        """
    folkQuery = """
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
        AND g.title = 'Folk'
        LIMIT 200;
        """
    popQuery = """
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
        AND g.title = 'Pop'
        LIMIT 200;
        """

    
    push!(queryList, rockQuery)
    push!(queryList, electronicQuery)
    push!(queryList, folkQuery)
    push!(queryList, popQuery)

    dataList = [getDbData(i) for i in queryList]


    function euclidian_distance(n1,n2)
        sum = 0
        for i in eachindex(n1)
            sum += (n1[i]-n2[i])^2
        end
        return sqrt(sum)
    end

    function weighted_distance(n1,n2,weights)
        sum = 0
        for i in eachindex(n1)
            sum += weights[i] * (n1[i]-n2[i])^2
        end
        return sqrt(sum)
    end


    function k_nn(data,labels,point,k)
        k = min(k,length(labels))
        distances = []
        for i in eachindex(data)
            distance = euclidian_distance(data[i],point)
            push!(distances,[distance,labels[i]])
        end
        sort!(distances)
        k_nearest = distances[1:k]
        nearest_label = [subarr[2] for subarr in k_nearest]
        label = mode(nearest_label)

        return label
    end

    function k_nn(weights,data,labels,point,k)
        k = min(k,length(labels))
        distances = []
        for i in eachindex(data)
            distance = weighted_distance(data[i],point,weights)
            push!(distances,[distance,labels[i]])
        end
        sort!(distances)
        k_nearest = distances[1:k]
        nearest_label = [subarr[2] for subarr in k_nearest]
        label = mode(nearest_label)

        return label
    end

    function unittest(weights,data,labels,testpoint,testlabel,k)
        estim = k_nn(weights,data,labels,testpoint,k)
        if estim == testlabel return true
        end
        return false
    end

    function correctness(weights, data, labels, testdata, testlabels, k)
        score = 0
        for (testpoint, testlabel) in zip(testdata, testlabels)
            if unittest(weights,data,labels,testpoint,testlabel,k)
                score += 1
            end
        end

        score /= length(testdata)
        return score
    end

    function generate_possibilities(array1, array2, array3)
        n = length(array1)
        # For each index, create a vector of the three options
        options = [ [array1[i], array2[i], array3[i]] for i in 1:n ]

        # Use Iterators.product to get all combinations
        result = [vcat(t...) for t in Iterators.product(options...)]
        
        return vec(result)
    end

    function optimize_weights(base_weights, data, labels, testdata, testlabels, k, iterations = 10, multiplier = 2)
        if iterations == 0
            return base_weights
        end

        weights_mult = base_weights .* multiplier
        weights_div = base_weights ./ multiplier
        
        candidate_weight_sets = generate_possibilities(weights_div,base_weights,weights_mult)

        best_weights = base_weights
        best_score = correctness(base_weights, data, labels, testdata, testlabels, k)
        
        for weights in candidate_weight_sets
            score = correctness(weights, data, labels, testdata, testlabels, k)
            if score > best_score
                best_score = score
                best_weights = weights
            end
            println(best_weights)
            println(best_score)
        end
        println(best_weights)
        println(best_score)

        return optimize_weights(best_weights, data, labels, testdata, testlabels, k, iterations -1, multiplier / 2 + 1)
    end


    function test()
        # Have to make a function that populates the data with the correct values from the
        # database. So we determined the 3 best choices: acousticness, danceability and
        # energy.
        # The 4 total genres we are going to use in order by highest number of songs:
        # Rock, Electronic, Folk and Pop.

        nbrOfDataPointsPerGenre = 160
        nbrOfTestsPointsPerGenre = 40

        data = []
        test_data = []
        for genreDF in dataList
            for i in 1:nbrOfDataPointsPerGenre
                attribute1 = genreDF.acousticness[i]
                attribute2 = genreDF.danceability[i]
                attribute3 = genreDF.energy[i]
                push!(data, [attribute1, attribute2, attribute3])
            end
            for i in nbrOfDataPointsPerGenre:(nbrOfDataPointsPerGenre + nbrOfTestsPointsPerGenre)
                attribute1 = genreDF.acousticness[i]
                attribute2 = genreDF.danceability[i]
                attribute3 = genreDF.energy[i]
                push!(test_data, [attribute1, attribute2, attribute3])
            end
        end

        labels = []
        test_labels = []
        for i in genreList
            for j in 1:nbrOfDataPointsPerGenre
                push!(labels, i)
            end
            for j in nbrOfDataPointsPerGenre:(nbrOfDataPointsPerGenre + nbrOfTestsPointsPerGenre)
                push!(test_labels, i)
            end
        end

        weights = [3,1,2]
        k = 10
        # This would be the song's info. So am testing the 11th rock song:
        # II - Saint Pancrace by ZUHN
        point = [0.831, 0.115, 0.007]


        acc1 = correctness(weights, data, labels, test_data, test_labels, k)
        println("Correctness without weigth: " , acc1)

        optimized_weights = optimize_weights(weights, data, labels, test_data, test_labels, k)

        acc2 = correctness(optimized_weights, data, labels, test_data, test_labels, k)
        println("Correctness with weigth: " , acc2)
        println("Optimized weights: ", optimized_weights)
        


        estim = k_nn(optimized_weights,data,labels,point,k)

        x = [d[1] for d in data]
        y = [d[2] for d in data]
        z = [d[3] for d in data]

        Plots.scatter3d(x, y, z, group=labels, title = "K-nn: dist=euclidean, datasize=" * string(length(data)) * ", k=" * string(k))
        Plots.scatter!([point[1]],[point[2]],[point[3]], label = "ESTIMATION = " * string(estim))
    end

   test()
end