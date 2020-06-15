using EmDee
using Test
using CUDA

function test_cells()
    N = 1000
    L = 1.0
    cutoff = 0.2
    x = CUDA.rand(Float32, 3, N)
    y = x .+ 0.01
    M, neighbors, head_x, next_x, index_x, collected = create_cells(x, L, cutoff)
    M, neighbors, head_y, next_y, index_y, collected = create_cells(y, L, cutoff)
    update_cells!(head_x, next_x, index_x, collected, y, L, M)
    return all(Array(index_x) .== Array(index_y))
end

@testset "EmDee.jl" begin
    @test test_cells()
    # Write your tests here.
end
