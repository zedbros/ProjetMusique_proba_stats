using Plots, StatsBase

begin
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
        data = [
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
        labels = [
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


        weights = [1,1,1]
        k = 5
        point = [3,1.2,0.5]

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