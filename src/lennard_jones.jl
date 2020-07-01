export LennardJonesModel, LennardJonesAtom

import CUDA

struct LennardJonesModel
    rc²::Float32
    rs²::Float32
    δ⁻²::Float32
    LennardJonesModel(cutoff, switch) = new(cutoff^2, switch^2, 1/(cutoff^2 - switch^2))
end

LennardJonesAtom(ε, σ) = LJAtom(0.5σ, 2*sqrt(ε))

struct LJAtom
    half_σ::Float32
    twice_sqrt_ε::Float32
end

@inline shfl_sync(mask, val::LJAtom, src) = LJAtom(
    CUDA.shfl_sync(mask, val.half_σ, src),
    CUDA.shfl_sync(mask, val.twice_sqrt_ε, src)
)

@inline function interaction(r²::Float32,
                             model::LennardJonesModel,
                             atom_i::LJAtom,
                             atom_j::LJAtom)::Tuple{Float32,Float32}
    σ = atom_i.half_σ + atom_j.half_σ
    ε4 = atom_i.twice_sqrt_ε*atom_j.twice_sqrt_ε
    s⁻² = σ*σ/r²
    s⁻⁶ = s⁻²*s⁻²*s⁻²
    ε4s⁻⁶ = atom_i.twice_sqrt_ε*atom_j.twice_sqrt_ε*s⁻⁶
    E = ε4s⁻⁶*(s⁻⁶ - 1)
    minus_E′r = 6ε4s⁻⁶*(2s⁻⁶ - 1)
    x = (r² - model.rs²)*model.δ⁻²
    x *= 0.5f0(sign(x) - sign(x-1))
    x² = x*x
    g = 1 + x*x²*(15x - 6x² - 10)
    minus_g′r = 60x²*(1 - 2x + x²)*model.δ⁻²*r²
    return E*g, minus_E′r*g + E*minus_g′r
end
