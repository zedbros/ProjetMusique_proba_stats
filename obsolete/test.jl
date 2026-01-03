begin
    using Plots
    print("Hello world lmao")

    Plots.plot([1,2,3,4,5,6,7,8,9], [4,5,4,5,4,5,4,5,4], label="test_label")
    Plots.plot!(title="Test")
end