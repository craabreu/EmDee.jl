import CUDA
using StaticArrays
using LinearAlgebra
CUDA.allowscalar(false)

include("lennard_jones.jl")

const WARPSIZE = 32
const Vec3 = StaticArrays.SVector{3,Float32}

function upper_triangular_pairs(n, T=Int32)
    pairs = Vector{Tuple{T,T}}(undef, n*(n+1)÷2)
    k = 0
    for i=0:n-1, j=1:n-i
        pairs[k+=1] = (j, j+i)
    end
    return pairs
end

function size_adjusted(array::Vector{T}) where {T}
    lacking = WARPSIZE*cld(length(array), WARPSIZE) - length(array)
    return vcat(array, Vector{T}(undef, lacking))
end

function size_adjusted(array::Matrix{T}, transpose=false) where {T}
    rows, cols = transpose ? (2, 1) : (1, 2)
    lacking = WARPSIZE*cld(size(array, cols), WARPSIZE) - size(array, cols)
    return hcat(transpose ? Matrix(array') : array, Matrix{T}(undef, size(array, rows), lacking))
end

@inline minimum_image(s) = s - round(s)
@inline to_vec3(vector, start) = Vec3(vector[start+1], vector[start+2], vector[start+3])

function compute_tiles!(forces, energies, virials, positions, L, tiles, model,
                        atoms::CUDA.CuDeviceArray{Atom,1},
                        ::Val{compute_energies}, ::Val{compute_virials}
                        ) where {Atom, compute_energies, compute_virials}
    @inbounds begin
        tid = CUDA.threadIdx().x
        bid = CUDA.blockIdx().x
        block_I, block_J = tiles[bid]
        is_diagonal_tile = block_J == block_I
        I = (block_I - 1)*WARPSIZE + tid
        J = (block_J - 1)*WARPSIZE + tid
        start_I = 3*(I - 1)
        start_J = 3*(J - 1)

        atoms_shmem = CUDA.@cuStaticSharedMem(Atom, WARPSIZE)
        atom_i = atoms[I]
        atoms_shmem[tid] = atoms[J]

        scaled_positions_shmem = CUDA.@cuStaticSharedMem(Vec3, WARPSIZE)
        scaled_position_i = to_vec3(positions, start_I)/L
        scaled_positions_shmem[tid] = to_vec3(positions, start_J)/L

        forces_shmem = CUDA.@cuStaticSharedMem(Vec3, WARPSIZE)
        force_i = forces_shmem[tid] = zeros(Vec3)

        if compute_energies
            energies_shmem = CUDA.@cuStaticSharedMem(Float32, WARPSIZE)
            energy_i = energies_shmem[tid] = 0.0f0
        end

        if compute_virials
            virials_shmem = CUDA.@cuStaticSharedMem(Float32, WARPSIZE)
            virial_i = virials_shmem[tid] = 0.0f0
        end

        for k = tid+1:tid+WARPSIZE-Int(is_diagonal_tile)
            j = (k - 1)%WARPSIZE + 1
            rᵥ = L*minimum_image.(scaled_positions_shmem[j] - scaled_position_i)
            r² = rᵥ⋅rᵥ
            E, r⁻¹E′ = interaction(r², model, atom_i, atoms_shmem[j])
            pair_force = r⁻¹E′*rᵥ
            force_i += pair_force
            forces_shmem[j] -= pair_force
            if compute_energies
                energy_i += E
                energies_shmem[j] += E
            end
            if compute_virials
                W = r⁻¹E′*r²
                virial_i += W
                virials_shmem[j] += W
            end
        end

        CUDA.atomic_add!(pointer(forces, start_I + 1), force_i[1])
        CUDA.atomic_add!(pointer(forces, start_I + 2), force_i[2])
        CUDA.atomic_add!(pointer(forces, start_I + 3), force_i[3])
        compute_energies && CUDA.atomic_add!(pointer(energies, I), 0.5f0*energy_i)
        compute_virials && CUDA.atomic_add!(pointer(virials, I), 0.5f0*virial_i)

        if !is_diagonal_tile
            CUDA.atomic_add!(pointer(forces, start_J + 1), forces_shmem[tid][1])
            CUDA.atomic_add!(pointer(forces, start_J + 2), forces_shmem[tid][2])
            CUDA.atomic_add!(pointer(forces, start_J + 3), forces_shmem[tid][3])
            compute_energies && CUDA.atomic_add!(pointer(energies, J), 0.5f0*energies_shmem[tid])
            compute_virials && CUDA.atomic_add!(pointer(virials, J), 0.5f0*virials_shmem[tid])
        end
    end
    return nothing
end

function compute_nonbonded_interactions!(forces, energies, virials, positions, L,
                                         tiles, model, atoms::CUDA.CuDeviceArray{Atom,1},
                                         compute_energies, compute_virials) where {Atom}
    forces .= 0.0f0
    compute_energies && (energies .= 0.0f0)
    compute_virials && (virials .= 0.0f0)
    bytes_per_thread = sizeof(Atom) + sizeof(Float32)*(6+Int(compute_energies)+Int(compute_virials))
    CUDA.@cuda(
        threads=WARPSIZE,
        blocks=length(tiles),
        shmem=WARPSIZE*bytes_per_thread,
        compute_tiles!(forces, energies, virials, positions, L, tiles, model, atoms,
                       Val(compute_energies), Val(compute_virials))
    )
end

function naively_compute_nonbonded_interaction!(positions, L, model, atoms)
    N = size(positions, 2)
    scaled_positions = map(x->x/L, positions)
    forces = zeros(3, N)
    energy = 0.0
    virial = 0.0
    for i = 1:N-1
        for j = i+1:N
            rᵥ = L*minimum_image.(scaled_positions[:,j] - scaled_positions[:,i])
            r² = rᵥ⋅rᵥ
            E, r⁻¹E′ = interaction(r², model, atoms[i], atoms[j])
            pair_force = r⁻¹E′*rᵥ
            forces[:,i] += pair_force
            forces[:,j] -= pair_force
            energy += E
            virial += r⁻¹E′*r²
        end
    end
    return energy, virial, forces
end
