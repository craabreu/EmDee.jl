import CUDA
using StaticArrays
using LinearAlgebra
CUDA.allowscalar(false)

include("lennard_jones.jl")

const WARPSIZE = 32
const Vec3 = StaticArrays.SVector{3,Float32}

function off_diagonal_pairs(n, T=Int32)
    pairs = Vector{Tuple{T,T}}(undef, n*(n-1)÷2)
    k = 0
    for i=1:n-1, j=1:n-i
        pairs[k+=1] = (j, j+i)
    end
    return pairs
end

function upper_triangular_pairs(n, T=Int32)
    pairs = Vector{Tuple{T,T}}(undef, n*(n+1)÷2)
    k = 0
    for i=0:n-1, j=1:n-i
        pairs[k+=1] = (j, j+i)
    end
    return pairs
end

function size_adjusted(array::Vector{T}) where {T}
    num_lacking_elements = WARPSIZE*cld(length(array), WARPSIZE) - length(array)
    return vcat(array, Vector{T}(undef, num_lacking_elements))
end

function size_adjusted(array::Matrix{T}, transpose=false) where {T}
    rows, cols = transpose ? (2, 1) : (1, 2)
    num_additions = WARPSIZE*cld(size(array, cols), WARPSIZE) - size(array, cols)
    return hcat(transpose ? Matrix(array') : array, Matrix{T}(undef, size(array, rows), num_additions))
end

@inline minimum_image(s) = s - round(s)

function compute_tiles!(forces, energies, virials, positions, L, tiles, model,
                        atoms::CUDA.CuDeviceArray{Atom,1}) where {Atom}

    tid = CUDA.threadIdx().x
    bid = CUDA.blockIdx().x
    block_I, block_J = tiles[bid]
    is_diagonal_tile = block_J == block_I
    I = (block_I - 1)*WARPSIZE + tid
    J = (block_J - 1)*WARPSIZE + tid

    atoms_shmem = CUDA.@cuStaticSharedMem(Atom, WARPSIZE)
    atom_i = atoms[I]
    atoms_shmem[tid] = atoms[J]

    scaled_positions_shmem = CUDA.@cuStaticSharedMem(Vec3, WARPSIZE)
    scaled_position_i = positions[I]/L
    scaled_positions_shmem[tid] = positions[J]/L

    forces_shmem = CUDA.@cuStaticSharedMem(Vec3, WARPSIZE)
    force_i = forces_shmem[tid] = zeros(Vec3)

    energies_shmem = CUDA.@cuStaticSharedMem(Float32, WARPSIZE)
    energy_i = energies_shmem[tid] = 0.0f0

    virials_shmem = CUDA.@cuStaticSharedMem(Float32, WARPSIZE)
    virial_i = virials_shmem[tid] = 0.0f0

    for k = tid+1:tid+WARPSIZE-Int(is_diagonal_tile)
        j = (k - 1)%WARPSIZE + 1
        rᵥ = L*minimum_image.(scaled_positions_shmem[j] - scaled_position_i)
        r² = rᵥ⋅rᵥ
        E, r⁻¹E′ = interaction(r², model, atom_i, atoms_shmem[j])
        pair_force = r⁻¹E′*rᵥ
        force_i += pair_force
        forces_shmem[j] -= pair_force

        energy_i += E
        energies_shmem[j] += E

        W = r⁻¹E′*r²
        virial_i += W
        virials_shmem[j] += W
    end

    start = 3*(I - 1)
    for (k, f) in enumerate(force_i)
        CUDA.atomic_add!(pointer(forces, start + k), f)
    end
    CUDA.atomic_add!(pointer(energies, I), 0.5f0*energy_i)
    CUDA.atomic_add!(pointer(virials, I), 0.5f0*virial_i)

    if !is_diagonal_tile
        start = 3*(J - 1)
        for (k, f) in enumerate(forces_shmem[tid])
            CUDA.atomic_add!(pointer(forces, start + k), f)
        end
        CUDA.atomic_add!(pointer(energies, J), 0.5f0*energies_shmem[tid])
        CUDA.atomic_add!(pointer(virials, J), 0.5f0*virials_shmem[tid])
    end

    return nothing
end
