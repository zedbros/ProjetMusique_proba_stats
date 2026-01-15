include("sql_client.jl")
using Plots, StatsBase
using Random
using LinearAlgebra

begin
    genreList = ["Rock", "Electronic", "Folk", "Pop"]
    queryList = []
    print("start")
    print("start")


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
        LIMIT 790;
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
        LIMIT 790;
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
        LIMIT 790;
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
        LIMIT 790;
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


    function optimize_weights(base_weights,dataList,genreList, k, iterations = 5, diff = 0.5, no_improvements = 0,global_best_weights = nothing, global_best_score = 0)
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
        #println(DFsize)
        nbrOfDataPointsPerGenre = DFsize * 80 ÷ 100
        #println(nbrOfDataPointsPerGenre)
        nbrOfTestsPointsPerGenre = DFsize - nbrOfDataPointsPerGenre
        #println(nbrOfTestsPointsPerGenre)

        
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

    function splitDataWithFinalTest(dataList, genreList, final_test_percent = 15)
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

    # GRADIENT DESCENT APPROACH
    function compute_numerical_gradient(weights, data, labels, testdata, testlabels, k, epsilon = 0.01)
        """
        Compute numerical gradient using finite differences
        gradient[i] ≈ (f(w + ε*e_i) - f(w - ε*e_i)) / (2ε)
        """
        gradient = zeros(length(weights))
        base_score = correctness(weights, data, labels, testdata, testlabels, k)
        
        for i in eachindex(weights)
            # Perturb weight i positively
            weights_plus = copy(weights)
            weights_plus[i] += epsilon
            score_plus = correctness(weights_plus, data, labels, testdata, testlabels, k)
            
            # Perturb weight i negatively
            weights_minus = copy(weights)
            weights_minus[i] = max(weights[i] - epsilon, 0)  # Keep non-negative
            score_minus = correctness(weights_minus, data, labels, testdata, testlabels, k)
            
            # Central difference
            gradient[i] = (score_plus - score_minus) / (2 * epsilon)
        end
        
        return gradient
    end

    function adam_optimize(initial_weights, training_dataList, genreList, k,
            max_iterations = 15,
            learning_rate = 0.3,
            beta1 = 0.9,
            beta2 = 0.999,
            epsilon_adam = 1e-8,
            epsilon_grad = 0.05,
            lambda = 0.01)
        """
        Optimize weights using Adam optimizer (adaptive moment estimation)
        Generally more robust than vanilla gradient descent
        """
        target_weights = ones(length(initial_weights))
        weights = copy(initial_weights)
        m = zeros(length(weights))  # First moment estimate
        v = zeros(length(weights))  # Second moment estimate
        best_weights = copy(weights)
        best_score = 0.0
        
        # Get initial data split
        data, testdata, labels, testlabels = getDataWithrandomTest(training_dataList, genreList)
        
        println("Starting Adam Optimization")
        println("=" ^ 50)
        
        for iter in 1:max_iterations
            # Reshuffle data every 5 iterations
           #=  if iter % 5 == 1 && iter > 1
                data, testdata, labels, testlabels = getDataWithrandomTest(training_dataList, genreList)
            end =#
            
            # Compute gradient
            gradient = compute_numerical_gradient(weights, data, labels, testdata, testlabels, k, epsilon_grad)

            #regularization_gradient = -1 * lambda * (weights - target_weights)
            #gradient = gradient + regularization_gradient
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (gradient .^ 2)
            
            # Compute bias-corrected moment estimates
            m_hat = m / (1 - beta1^iter)
            v_hat = v / (1 - beta2^iter)
            
            # Update weights
            weights = weights + learning_rate * m_hat ./ (sqrt.(v_hat) .+ epsilon_adam)
            
            # Keep weights non-negative
            weights = max.(weights, 0)
            
            # Evaluate current weights
            current_score = correctness(weights, data, labels, testdata, testlabels, k)
            
            # Track best weights
            if current_score > best_score
                best_score = current_score
                best_weights = copy(weights)
                println("✓ Iter $iter: score = $(round(current_score, digits=4)), weights = $(round.(weights, digits=3))")
            else
                println("  Iter $iter: score = $(round(current_score, digits=4)), weights = $(round.(weights, digits=3))")
            end
            
            #= # Early stopping if gradient becomes very small
            if norm(gradient) < 0.001
                println("Gradient too small, stopping early")
                break
            end =#
        end
        
        println("=" ^ 50)
        println("Best score achieved: $(round(best_score, digits=4))")
        println("Best weights: $(round.(best_weights, digits=3))")
        
        return best_weights
    end
    

    function test()
        print("start")
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
        println("Correctness with 3-1-2 weigth: " , acc1)

        adam_optimised_weights = adam_optimize([1.0, 1.0, 1.0], training_dataList, genreList, k)

        acc3 = correctness(adam_optimised_weights, data, labels, final_test_data, final_test_labels, k)
        println("Correctness with adam-optimized weigth: " , acc3)
        println("Optimized weights: ", adam_optimised_weights)

        optimized_weights = optimize_weights([1,1,1], training_dataList, genreList, k)



        acc2 = correctness(optimized_weights, data, labels, final_test_data, final_test_labels, k)
        println("Correctness with custom-optimized weigth: " , acc2)
        println("Optimized weights: ", optimized_weights)
        

        estim = k_nn(optimized_weights,data,labels,point,k)

        x = [d[1] for d in data]
        y = [d[2] for d in data]
        z = [d[3] for d in data]

        plots = []
        for genre in genreList
            # Filter data for this genre
            genre_indices = findall(l -> l == genre, labels)
            x_genre = [data[i][1] for i in genre_indices]
            y_genre = [data[i][2] for i in genre_indices]
            z_genre = [data[i][3] for i in genre_indices]
            
            # Create individual scatter plot
            p = Plots.scatter3d(x_genre, y_genre, z_genre,
                            title = genre,
                            xlabel = "Acousticness",
                            ylabel = "Danceability",
                            zlabel = "Energy",
                            legend = false,
                            markersize = 3)
            
            push!(plots, p)
        end

        # Display all plots in a grid (2x2 for 4 genres)
        plot(plots..., layout = (2, 2), size = (1200, 1000))

       #=  Plots.scatter3d(x, y, z, group=labels, title = "",
                xlabel = "Acousticness",
                ylabel = "Danceability",
                zlabel = "Energy") =#
        #Plots.scatter3d(x, y, z, group=labels, title = "K-nn: dist=euclidean, datasize=" * string(length(data)) * ", k=" * string(k))
        #Plots.scatter!([point[1]],[point[2]],[point[3]], label = "ESTIMATION = " * string(estim))
    end

    function create_confusion_matrix(weights, data, labels, testdata, testlabels, k, genreList)
        """
        Create and display a confusion matrix for the k-NN classifier
        Returns the confusion matrix as a 2D array
        """
        n_classes = length(genreList)
        confusion_mat = zeros(Int, n_classes, n_classes)
        
        # Create a mapping from genre name to index
        genre_to_idx = Dict(genre => i for (i, genre) in enumerate(genreList))
        print(length(test_data))
        
        # Predict each test point and update confusion matrix
        for (testpoint, true_label) in zip(testdata, testlabels)
            predicted_label = k_nn(weights, data, labels, testpoint, k)
            
            true_idx = genre_to_idx[true_label]
            pred_idx = genre_to_idx[predicted_label]
            
            confusion_mat[true_idx, pred_idx] += 1
        end
        
        # Display the confusion matrix
        println("\n" * "="^60)
        println("CONFUSION MATRIX")
        println("="^60)
        println("Rows = Actual, Columns = Predicted")
        println()
        
        # Header
        print("              ")
        for genre in genreList
            print(rpad(genre, 12))
        end
        println()
        println("-"^60)
        
        # Matrix rows
        for (i, genre) in enumerate(genreList)
            print(rpad(genre, 12) * "  ")
            for j in 1:n_classes
                print(rpad(string(confusion_mat[i, j]), 12))
            end
            println()
        end
        println("="^60)
        
        # Calculate per-class metrics
        println("\nPER-CLASS METRICS:")
        println("-"^60)
        
        for (i, genre) in enumerate(genreList)
            true_positives = confusion_mat[i, i]
            false_positives = sum(confusion_mat[:, i]) - true_positives
            false_negatives = sum(confusion_mat[i, :]) - true_positives
            true_negatives = sum(confusion_mat) - true_positives - false_positives - false_negatives
            
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            println("$genre:")
            println("  Precision: $(round(precision, digits=3))")
            println("  Recall:    $(round(recall, digits=3))")
            println("  F1-Score:  $(round(f1_score, digits=3))")
            println()
        end
        
        # Overall accuracy
        total_correct = sum(confusion_mat[i, i] for i in 1:n_classes)
        total_samples = sum(confusion_mat)
        accuracy = total_correct / total_samples
        
        println("Overall Accuracy: $(round(accuracy, digits=4))")
        println("="^60)
        
        return confusion_mat
    end

    function plot_confusion_matrix(confusion_mat, genreList)
        """
        Create a visual heatmap of the confusion matrix
        """
        
        # Normalize by row (actual class) to show percentages
        # Need to handle the division carefully to avoid NaN
        row_sums = sum(confusion_mat, dims=2)
        confusion_normalized = confusion_mat ./ max.(row_sums, 1)  # Avoid division by zero
        
        heatmap(1:length(genreList), 1:length(genreList), confusion_normalized,
                xlabel="Predicted",
                ylabel="Actual",
                title="Confusion Matrix (Normalized by Row)",
                color=:Blues,
                clim=(0, 1),
                aspect_ratio=:equal,
                xticks=(1:length(genreList), genreList),
                yticks=(1:length(genreList), genreList))
        
        # Add text annotations - centered in squares
        # Show both count and percentage
        n = length(genreList)
        for i in 1:n
            for j in 1:n
                # Determine text color based on background darkness
                text_color = confusion_normalized[i, j] > 0.5 ? :white : :black
                count = confusion_mat[i, j]
                percentage = round(confusion_normalized[i, j] * 100, digits=1)
                label = "$count\n$(percentage)%"
                annotate!(j, i, text(label, 8, text_color))
            end
        end
    
        return current()
    end

    weights =  [0.099, 0.971, 2.439]
    test()
    #weights = [0.65625, 0.78125, 1.171875]
    #weights = [0.1875, 0.4375, 1.875]

 #=    training_dataList, final_test_data, final_test_labels = splitDataWithFinalTest(dataList, genreList)
    data, test_data, labels, test_labels = getDataWithrandomTest(training_dataList, genreList)
    confusion_matrix = create_confusion_matrix(weights, data, labels, final_test_data, final_test_labels, 20, genreList)
    plot_confusion_matrix(confusion_matrix, genreList) =#
   #test()
end