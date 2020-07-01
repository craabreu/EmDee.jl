export FORCES, ENERGIES, VIRIALS
export nonbonded_computation_tiles, compute_nonbonded!, naively_compute_nonbonded!

import CUDA

CUDA.allowscalar(false)

const FORCES = 1 << 0
const ENERGIES = 1 << 1
const VIRIALS = 1 << 2

const WARPSIZE = 32

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
        Ix3 = 3I
        Jx3 = 3J

        atom_i = atoms[I]
        atom_j = atoms[J]

        scaled_pos_i = Vec3(positions[Ix3-2], positions[Ix3-1], positions[Ix3])/L
        scaled_pos_j = Vec3(positions[Jx3-2], positions[Jx3-1], positions[Jx3])/L

        FORCES & bitmask ≠ 0 && (force_i = force_j = zeros(Vec3))
        ENERGIES & bitmask ≠ 0 && (energy_i = energy_j = 0.0f0)
        VIRIALS & bitmask ≠ 0 && (virial_i = virial_j = 0.0f0)

        for m = 1:WARPSIZE-Int(is_diagonal_tile)
            j = (tid + m - 1)%WARPSIZE + 1
            k = (tid + WARPSIZE - m - 1)%WARPSIZE + 1
            rᵥ = L*minimum_image.(scaled_pos_i - shfl_sync(CUDA.FULL_MASK, scaled_pos_j, j))
            r² = rᵥ⋅rᵥ
            E, minus_rE′ = interaction(r², model, atom_i, shfl_sync(CUDA.FULL_MASK, atom_j, j))
            if FORCES & bitmask ≠ 0
                force_ij = minus_rE′/r²*rᵥ
                force_i += force_ij
                force_j -= shfl_sync(CUDA.FULL_MASK, force_ij, k)
            end
            if ENERGIES & bitmask ≠ 0
                energy_i += E
                energy_j += CUDA.shfl_sync(CUDA.FULL_MASK, E, k)
            end
            if VIRIALS & bitmask ≠ 0
                virial_i += minus_rE′
                virial_j += CUDA.shfl_sync(CUDA.FULL_MASK, minus_rE′, k)
            end
        end

        if FORCES & bitmask ≠ 0
            CUDA.atomic_add!(pointer(forces, Ix3-2), force_i.x)
            CUDA.atomic_add!(pointer(forces, Ix3-1), force_i.y)
            CUDA.atomic_add!(pointer(forces, Ix3), force_i.z)
        end
        ENERGIES & bitmask ≠ 0 && CUDA.atomic_add!(pointer(energies, I), 0.5f0*energy_i)
        VIRIALS & bitmask ≠ 0 && CUDA.atomic_add!(pointer(virials, I), 0.5f0*virial_i)

        if !is_diagonal_tile
            if FORCES & bitmask ≠ 0
                CUDA.atomic_add!(pointer(forces, Jx3-2), force_j.x)
                CUDA.atomic_add!(pointer(forces, Jx3-1), force_j.y)
                CUDA.atomic_add!(pointer(forces, Jx3), force_j.z)
            end
            ENERGIES & bitmask ≠ 0 && CUDA.atomic_add!(pointer(energies, J), 0.5f0*energy_j)
            VIRIALS & bitmask ≠ 0 && CUDA.atomic_add!(pointer(virials, J), 0.5f0*virial_j)
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
    CUDA.@cuda(
        threads=WARPSIZE,
        blocks=length(tiles),
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
