using EmDee
using Test
using CUDA

function test_cells()
    N = 1000
    L = 1.0
    cutoff = 0.2
    x = CUDA.rand(Float32, 3, N)
    y = x .+ 0.01
    cells_x = Cells(x, L, cutoff)
    cells_y = Cells(y, L, cutoff)
    update_cells!(cells_x, y, L)
    return all(Array(cells_x.index) .== Array(cells_y.index))
end

@testset "EmDee.jl" begin
    @test test_cells()
    # Write your tests here.
end
