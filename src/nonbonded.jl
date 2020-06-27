export FORCES, ENERGIES, VIRIALS
export nonbonded_computation_tiles, compute_nonbonded!, naively_compute_nonbonded!

import CUDA
using StaticArrays

CUDA.allowscalar(false)

const FORCES = 1 << 0
const ENERGIES = 1 << 1
const VIRIALS = 1 << 2

const WARPSIZE = 32
const Vec3 = StaticArrays.SVector{3,Float32}

function nonbonded_computation_tiles(N)
    n = Int32(cld(N, WARPSIZE))
    pairs = Vector{Tuple{Int32,Int32}}(undef, n*(n+1)÷2)
    k = 0
    for i=0:n-1, j=1:n-i
        pairs[k+=1] = (j, j+i)
    end
    return CUDA.cu(pairs)
end

function size_adjusted(array::Vector{T}) where {T}
    lacking = WARPSIZE*cld(length(array), WARPSIZE) - length(array)
    return vcat(array, Vector{T}(undef, lacking))
end

function size_adjusted(array::Matrix{T}, transpose=false) where {T}
    rows, cols = transpose ? (2, 1) : (1, 2)
    lacking = WARPSIZE*cld(size(array, cols), WARPSIZE) - size(array, cols)
    A = hcat(transpose ? Matrix(array') : array, Matrix{T}(undef, size(array, rows), lacking))
    return A
end

@inline minimum_image(s) = s - round(s)

@inline to_vec3(vector, offset) = Vec3(vector[offset+1], vector[offset+2], vector[offset+3])

@inline ⋅(x, y) = sum(x.*y)

function compute_tile!(forces, energies, virials, positions, L, tiles, model,
                       atoms::CUDA.CuDeviceArray{Atom,1}, ::Val{bitmask}
                       ) where {Atom, bitmask}
    @inbounds begin
        tid = CUDA.threadIdx().x
        bid = CUDA.blockIdx().x
        block_I, block_J = tiles[bid]
        is_diagonal_tile = block_J == block_I
        I = (block_I - 1)*WARPSIZE + tid
        J = (block_J - 1)*WARPSIZE + tid
        offset_I = 3*(I - 1)
        offset_J = 3*(J - 1)

        atoms_shmem = CUDA.@cuStaticSharedMem(Atom, WARPSIZE)
        atom_i = atoms[I]
        atoms_shmem[tid] = atoms[J]

        scaled_positions_shmem = CUDA.@cuStaticSharedMem(Vec3, WARPSIZE)
        scaled_position_i = to_vec3(positions, offset_I)/L
        scaled_positions_shmem[tid] = to_vec3(positions, offset_J)/L

        if FORCES & bitmask ≠ 0
            forces_shmem = CUDA.@cuStaticSharedMem(Vec3, WARPSIZE)
            force_i = forces_shmem[tid] = zeros(Vec3)
        end
        if ENERGIES & bitmask ≠ 0
            energies_shmem = CUDA.@cuStaticSharedMem(Float32, WARPSIZE)
            energy_i = energies_shmem[tid] = 0.0f0
        end
        if VIRIALS & bitmask ≠ 0
            virials_shmem = CUDA.@cuStaticSharedMem(Float32, WARPSIZE)
            virial_i = virials_shmem[tid] = 0.0f0
        end

        for k = tid+1:tid+WARPSIZE-Int(is_diagonal_tile)
            j = (k - 1)%WARPSIZE + 1
            rᵥ = L*minimum_image.(scaled_position_i - scaled_positions_shmem[j])
            r² = rᵥ⋅rᵥ
            E, minus_rE′ = interaction(r², model, atom_i, atoms_shmem[j])
            if FORCES & bitmask ≠ 0
                force_ij = minus_rE′/r²*rᵥ
                force_i += force_ij
                forces_shmem[j] -= force_ij
            end
            if ENERGIES & bitmask ≠ 0
                energy_i += E
                energies_shmem[j] += E
            end
            if VIRIALS & bitmask ≠ 0
                virial_i += minus_rE′
                virials_shmem[j] += minus_rE′
            end
        end

        if FORCES & bitmask ≠ 0
            CUDA.atomic_add!(pointer(forces, offset_I + 1), force_i[1])
            CUDA.atomic_add!(pointer(forces, offset_I + 2), force_i[2])
            CUDA.atomic_add!(pointer(forces, offset_I + 3), force_i[3])
        end
        ENERGIES & bitmask ≠ 0 && CUDA.atomic_add!(pointer(energies, I), 0.5f0*energy_i)
        VIRIALS & bitmask ≠ 0 && CUDA.atomic_add!(pointer(virials, I), 0.5f0*virial_i)

        if !is_diagonal_tile
            if FORCES & bitmask ≠ 0
                CUDA.atomic_add!(pointer(forces, offset_J + 1), forces_shmem[tid][1])
                CUDA.atomic_add!(pointer(forces, offset_J + 2), forces_shmem[tid][2])
                CUDA.atomic_add!(pointer(forces, offset_J + 3), forces_shmem[tid][3])
            end
            ENERGIES & bitmask ≠ 0 && CUDA.atomic_add!(pointer(energies, J), 0.5f0*energies_shmem[tid])
            VIRIALS & bitmask ≠ 0 && CUDA.atomic_add!(pointer(virials, J), 0.5f0*virials_shmem[tid])
        end
    end
    return nothing
end

function compute_nonbonded!(forces, energies, virials, positions, L,
                            tiles, model, atoms::CUDA.CuArray{Atom,1},
                            ::Val{bitmask}) where {Atom, bitmask}
    FORCES & bitmask ≠ 0 && (forces .= 0.0f0)
    ENERGIES & bitmask ≠ 0 && (energies .= 0.0f0)
    VIRIALS & bitmask ≠ 0 && (virials .= 0.0f0)
    bytes_per_thread = sizeof(Atom) + sizeof(Float32)*(3 + 3*count_ones(bitmask))
    CUDA.@cuda(
        threads=WARPSIZE,
        blocks=length(tiles),
        shmem=WARPSIZE*bytes_per_thread,
        compute_tile!(forces, energies, virials, positions, L, tiles, model, atoms, Val(bitmask))
    )
end

function naively_compute_nonbonded!(forces, energies, virials, positions, L, model, atoms)
    atoms_h = Array(atoms)
    scaled_positions =map(x->x/L, Array(positions))
    N = size(positions, 2)
    forces_h = zeros(Float32, 3, N)
    energies_h = zeros(Float32, N)
    virials_h = zeros(Float32, N)
    for i = 1:N-1
        atom_i = atoms_h[i]
        scaled_position_i = scaled_positions[:,i]
        energy_i = 0.0
        virial_i = 0.0
        force_i = zeros(3)
        for j = i+1:N
            rᵥ = L*minimum_image.(scaled_position_i - scaled_positions[:,j])
            r² = rᵥ⋅rᵥ
            E, minus_E′r = interaction(r², model, atom_i, atoms_h[j])
            f_ij = minus_E′r/r²*rᵥ
            force_i += f_ij
            forces_h[:,j] -= f_ij
            energy_i += E/2
            energies_h[j] += E/2
            virial_i += minus_E′r/2
            virials_h[j] += minus_E′r/2
        end
        forces_h[:,i] += force_i
        energies_h[i] += energy_i
        virials_h[i] += virial_i
    end
    unsafe_copyto!(pointer(forces), pointer(forces_h), 3*N)
    unsafe_copyto!(pointer(energies), pointer(energies_h), N)
    unsafe_copyto!(pointer(virials), pointer(virials_h), N)
    return nothing
end
