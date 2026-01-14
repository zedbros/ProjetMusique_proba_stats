include("sql_client.jl")
using Plots, StatsBase
using Random
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
        LIMIT 750;
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
        LIMIT 750;
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
        LIMIT 750;
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
        LIMIT 750;
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

    function generate_possibilities(arrays...)
        n = length(arrays[1])
        
        # Verify all arrays have same length
        @assert all(length(arr) == n for arr in arrays) "All arrays must have the same length"
        
        # For each index, collect the options from all arrays
        options = [[arr[i] for arr in arrays] for i in 1:n]
        
        # Generate all combinations
        result = [vcat(t...) for t in Iterators.product(options...)]
        
        return vec(result)
    end


    function optimize_weights(base_weights,dataList,genreList, k, iterations = 5, diff = 1, no_improvements = 0,global_best_weights = nothing, global_best_score = 0)
        #todo : find a way to escape local optima (simulated annealing,random restarts or momentum)

        if iterations == 0
            return base_weights
        end
        data, testdata, labels, testlabels = getDataWithrandomTest(dataList, genreList)

        # Initialize global best on first call
        if global_best_weights === nothing
            global_best_weights = base_weights
            global_best_score = correctness(base_weights, data, labels, testdata, testlabels, k)
            println("Initial score: $global_best_score")
        end
        
        println("Iteration: $(-iterations), diff: $diff")


        weights_add2 = base_weights .+ diff
        weights_add1 = base_weights .+ 0.5 * diff
        weights_sus1 = max.(base_weights .- 0.5 * diff, 0)
        weights_sus2 = max.(base_weights .- diff, 0)
        
        candidate_weight_sets = generate_possibilities(weights_add2,weights_add1,base_weights,weights_sus1,weights_sus2)
        #println(candidate_weight_sets)

        best_weights = base_weights
        best_score = correctness(base_weights, data, labels, testdata, testlabels, k)
        
        for weights in candidate_weight_sets
            score = correctness(weights, data, labels, testdata, testlabels, k)
            if score > best_score
                best_score = score
                best_weights = weights
            end
        end

        if best_score > global_best_score
            global_best_score = best_score
            global_best_weights = best_weights
        end

        println(best_weights)
        println(best_score)


        # if best_weights == base_weights
        #     no_improvements += 1
        #  else
        #     no_improvements = 0  # Reset counter on improvement
        # end
        new_diff = diff/ 2
        new_weights = best_weights

        if no_improvements >= 1 && diff < 0.2  #2 times same
            println("-> Random restart triggered!")
            new_weights = rand(length(base_weights))
            new_diff = 1.0
            no_improvements = 0
        end

        return optimize_weights(new_weights,dataList,genreList, k, iterations -1, new_diff,no_improvements,global_best_weights, global_best_score)
    end

    function getDataWithrandomTest(dataList, genreList)
        data = []
        test_data = []

        labels = []
        test_labels = []

        DFsize = nrow(dataList[1])
        println(DFsize)
        nbrOfDataPointsPerGenre = DFsize * 95 ÷ 100
        println(nbrOfDataPointsPerGenre)
        nbrOfTestsPointsPerGenre = DFsize - nbrOfDataPointsPerGenre
        println(nbrOfTestsPointsPerGenre)

        
        for (genreDF, genre_name) in zip(dataList, genreList)

            shuffled_indices = shuffle(1:nrow(genreDF))
            genreDF = genreDF[shuffled_indices, :]

            for i in 1:nbrOfDataPointsPerGenre
                attribute1 = genreDF.acousticness[i]
                attribute2 = genreDF.danceability[i]
                attribute3 = genreDF.energy[i]

                push!(data, [attribute1, attribute2, attribute3])

                push!(labels, genre_name)
            end
            for i in (nbrOfDataPointsPerGenre+1):(nbrOfDataPointsPerGenre + nbrOfTestsPointsPerGenre)
                attribute1 = genreDF.acousticness[i]
                attribute2 = genreDF.danceability[i]
                attribute3 = genreDF.energy[i]

                push!(test_data, [attribute1, attribute2, attribute3])

                push!(test_labels, genre_name)
            end
        end

        return data, test_data, labels, test_labels

    end

    function splitDataWithFinalTest(dataList, genreList, final_test_percent = 20)
        """
        Splits data into:
        - training_dataList: DataFrames for optimization (will be further split in each iteration)
        - final_test_data: Fixed test set (never touched during optimization)
        - final_test_labels: Labels for final test set
        """
        training_dataList = []
        final_test_data = []
        final_test_labels = []
        
        for (genreDF, genre_name) in zip(dataList, genreList)
            DFsize = nrow(genreDF)
            
            # Shuffle the dataframe
            shuffled_indices = shuffle(1:nrow(genreDF))
            genreDF_shuffled = genreDF[shuffled_indices, :]
            
            # Calculate split point
            final_test_size = DFsize * final_test_percent ÷ 100
            training_size = DFsize - final_test_size
            
            # Split into training dataframe and final test
            training_df = genreDF_shuffled[1:training_size, :]
            final_test_df = genreDF_shuffled[(training_size+1):end, :]
            
            push!(training_dataList, training_df)
            
            # Extract final test points
            for i in 1:nrow(final_test_df)
                attribute1 = final_test_df.acousticness[i]
                attribute2 = final_test_df.danceability[i]
                attribute3 = final_test_df.energy[i]
                
                push!(final_test_data, [attribute1, attribute2, attribute3])
                push!(final_test_labels, genre_name)
            end
        end
        
        return training_dataList, final_test_data, final_test_labels
    end

    

    function test()
        # Have to make a function that populates the data with the correct values from the
        # database. So we determined the 3 best choices: acousticness, danceability and
        # energy.
        # The 4 total genres we are going to use in order by highest number of songs:
        # Rock, Electronic, Folk and Pop.

        #nbrOfDataPointsPerGenre = 900
        #nbrOfTestsPointsPerGenre = 100

        training_dataList, final_test_data, final_test_labels = splitDataWithFinalTest(dataList, genreList)
        

        data, test_data, labels, test_labels = getDataWithrandomTest(training_dataList, genreList)

        weights = [3,1,2]
        k = 20
        # This would be the song's info. So am testing the 11th rock song:
        # II - Saint Pancrace by ZUHN
        point = [0.831, 0.115, 0.007]


        acc1 = correctness(weights, data, labels, final_test_data, final_test_labels, k)
        println("Correctness without weigth: " , acc1)

        optimized_weights = optimize_weights([1,1,1], training_dataList, genreList, k)

        acc2 = correctness(optimized_weights, data, labels, final_test_data, final_test_labels, k)
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