using CSV, DataFrames, StatsPlots

begin
	f = CSV.read("year_genre_data.csv", DataFrame)
	local d = Dict() # 2012 dictionary (can do them all in one by using arrays. Dict([2012], [2013])
    local aOfD = []
    for yearOfAnalysis in 1980:2025
    # Fills a dictionary with "Genre" => "Nbr of occurences" in a specific year: yearOfAnalysis
        for i in eachrow(f)
            if i.year == yearOfAnalysis
                if i.genre in keys(d)
                    d[i.genre] = d[i.genre] + 1
                else
                    d[i.genre] = 1
                end
            end
        end

    push!(aOfD)
    println("")
    println("")
	println(yearOfAnalysis, " : ", d)
    d = Dict()
    end
    

    local x = [1 2 3; 4 5 6; 7 8 9]
    local y = [11,12,13]

    StatsPlots.groupedbar(x, bar_position = :stack, bar_width = 0.7)
end