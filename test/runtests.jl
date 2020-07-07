using EmDee
using Test
using CUDA
using DelimitedFiles

# function test_cells()
#     N = 1000
#     L = 1.0
#     cutoff = 0.2
#     x = CUDA.rand(Float32, 3, N)
#     y = x .+ 0.01
#     cells_x = Cells(x, L, cutoff)
#     cells_y = Cells(y, L, cutoff)
#     update_cells!(cells_x, y, L)
#     return all(Array(cells_x.index) .== Array(cells_y.index)) &&
#            all(Array(cells_x.population) .== Array(cells_y.population))
# end

function test_compute_nonbonded(xyz_file, L, cutoff, switch)
    xyz_data = [row[i] for i=2:4, row in eachrow(readdlm(xyz_file; skipstart=2))]

    positions = CUDA.cu(xyz_data)
    N = size(xyz_data, 2)

    model = LennardJonesModel(cutoff, switch)
    atoms = CUDA.cu(fill(LennardJonesAtom(1, 1), N))

    forces_ref = CUDA.zeros(Float32, 3, N)
    energies_ref = CUDA.zeros(N)
    virials_ref = CUDA.zeros(N)

    naively_compute_nonbonded!(forces_ref, energies_ref, virials_ref, positions, L, model, atoms)

    tiles = nonbonded_computation_tiles(N)

    forces = CUDA.zeros(Float32, 3, N)
    energies = CUDA.zeros(N)
    virials = CUDA.zeros(N)

    compute_nonbonded!(forces, energies, virials, positions, L,
                       tiles, model, atoms, Val(FORCES | ENERGIES | VIRIALS))

    return maximum(forces .- forces_ref) < 1.0f-4 &&
           maximum(energies .- energies_ref) < 1.0f-4 &&
           maximum(virials .- virials_ref) < 1.0f-4
end

function test_system(pdb_file)
    system = System(pdb_file)
    return length(system.frame) == 1519
end

@testset "EmDee.jl" begin
    # @test test_cells()
    @test test_compute_nonbonded(joinpath(@__DIR__, "data", "lj_sample.xyz"), 10, 3, 2.5)
    @test test_system(joinpath(@__DIR__, "data", "dibenzo-p-dioxin-in-water.pdb"))
end
