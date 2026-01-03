include("sql_client.jl")
using Plots, StatsBase

begin
    genreList = ["Rock", "Electronic", "Folk"]
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
        LIMIT 500;
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
        LIMIT 500;
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
        LIMIT 500;
        """
    push!(queryList, rockQuery)
    push!(queryList, electronicQuery)
    push!(queryList, folkQuery)

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

    function optimize_weights(base_weights, data, labels, testdata, testlabels, k, iterations = 50, multiplier = 2)
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
        end
        println(best_weights)
        println(best_score)

        return optimize_weights(best_weights, data, labels, testdata, testlabels, k, iterations -1, multiplier / 2 + 1)
    end
        
    #data = [datapoint,datapoint,datapoint]
    #labels = [rock, metal, rock]
    #datapoint = [chroma, tempo, danceability]

    function test()
        # Have to make a function that populates the data with the correct values from the
        # database.
        # So class A for example would be the Rock classe.
        # The 6 total genres we are going to use in order by highest number of songs:
        # Rock, Electronic, Folk, Punk, Hip-Hop, Pop.

        data_old = [
            #Point = [attribute A, attribute B, attribute C]

            # Class A
            [1.0, 1.0, 0.5], [1.2, 1.1, 0.6], [1.1, 1.3, 0.4], [1.3, 1.0, 0.7], [1.4, 1.2, 0.5],
            [1.5, 1.3, 0.6], [1.6, 1.4, 0.7], [1.7, 1.2, 0.5], [1.8, 1.3, 0.6], [1.9, 1.1, 0.5],

            # Class B (overlapping with A)
            [2.0, 1.5, 1.0], [2.1, 1.6, 1.1], [2.2, 1.4, 0.9], [2.3, 1.5, 1.2], [2.4, 1.6, 1.0],
            [2.0, 1.3, 0.8], [2.1, 1.2, 0.9], [2.2, 1.5, 1.0], [2.3, 1.4, 1.1], [2.4, 1.3, 0.9],

            # Class C (overlapping with B)
            [2.5, 1.7, 1.5], [2.6, 1.6, 1.4], [2.7, 1.8, 1.6], [2.8, 1.7, 1.5], [2.9, 1.6, 1.4],
            [3.0, 1.8, 1.6], [3.1, 1.7, 1.5], [3.2, 1.9, 1.7], [3.0, 1.6, 1.4], [2.8, 1.5, 1.3],

            # Class C/B overlap
            [3.0, 2.0, 1.8], [3.1, 2.1, 1.9], [3.2, 2.0, 1.7], [3.3, 2.1, 1.8], [3.4, 2.0, 1.9],
            [2.9, 2.0, 1.6], [2.8, 2.1, 1.7], [2.7, 2.0, 1.5], [2.6, 2.1, 1.6], [2.5, 2.0, 1.7],

            # B/C/A overlap
            [2.0, 2.2, 1.2], [2.1, 2.3, 1.3], [2.2, 2.2, 1.1], [2.3, 2.3, 1.2], [2.4, 2.2, 1.1],
            [1.8, 2.1, 0.9], [1.9, 2.2, 1.0], [2.0, 2.1, 1.1], [1.7, 2.0, 0.8], [1.6, 2.1, 0.9]
        ]
        labels_old = [
            # Class A
            'A','A','A','A','A','A','A','A','A','A',

            # Class B
            'B','B','B','B','B','B','B','B','B','B',

            # Class C
            'C','C','C','C','C','C','C','C','C','C',

            # Class C/B overlap
            'C','C','C','C','C','C','C','C','C','C',

            # B/C/A overlap
            'B','B','B','B','B','A','A','B','A','A'
        ]


        # Let's start with 10 songs from each genre.
        nbrOfDataPointsPerGenre = 500
        data = []
        for genreDF in dataList
            for i in 1:nbrOfDataPointsPerGenre
                attribute1 = genreDF.acousticness[i]
                attribute2 = genreDF.danceability[i]
                attribute3 = genreDF.energy[i]
                push!(data, [attribute1, attribute2, attribute3])
            end
        end

        labels = []
        for i in genreList
            for j in 1:nbrOfDataPointsPerGenre
                push!(labels, i)
            end
        end

        weights = [3,1,2]
        k = 300
        # This would be the song's info. So am testing the 11th rock song:
        # II - Saint Pancrace by ZUHN
        point = [0.831, 0.115, 0.007]

        # /////////////////////// CURRENT STATUS ///////////////////////
        # We now have a function that takes the first 500 songs in
        # the genres Rock, Electronic and Folk.
        # It then takes each song's acousticness, danceability and energy
        # values and are represented as one point on the graph.
        # We then take our points (which is our song of choice's values)
        # and we use knn to try and tetermine which genre it belongs too.

        # In order to improve this algorithm's accuracy, an idea would
        # be to choose our 500 songs well, by using those who's values
        # are closest to that genre's average as an example.
        # The average isn't the best option so will try something
        # more concrete.
        # ////////////////////// /CURRENT STATUS/ //////////////////////

        acc = correctness(weights, data, labels, data, labels, k)
        print(acc)


        print(optimize_weights(weights, data, labels, data, labels, k))
        
        estim = k_nn(weights,data,labels,point,k)

        x = [d[1] for d in data]
        y = [d[2] for d in data]
        z = [d[3] for d in data]

        Plots.scatter3d(x, y, z, group=labels, title = "K-nn, dist= euclidean, datasize= " * string(length(data)) * " k= " * string(k))
        Plots.scatter!([point[1]],[point[2]],[point[3]], label = "ESTIMATION = " * string(estim))
    end

   test()
end