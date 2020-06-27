export LennardJonesAtom, LennardJonesModel

struct LennardJonesAtom
    half_σ::Float32
    twice_sqrt_ε::Float32
    LennardJonesAtom(ε, σ) = new(0.5σ, 2*sqrt(ε))
end

struct LennardJonesModel
    rc²::Float32
    rs²::Float32
    δ⁻²::Float32
    LennardJonesModel(cutoff, switch) = new(cutoff^2, switch^2, 1/(cutoff^2 - switch^2))
end

@inline function interaction(r²::Float32,
                             model::LennardJonesModel,
                             atom_i::LennardJonesAtom,
                             atom_j::LennardJonesAtom)::Tuple{Float32,Float32}
    σ = atom_i.half_σ + atom_j.half_σ
    ε4 = atom_i.twice_sqrt_ε*atom_j.twice_sqrt_ε
    s⁻² = σ*σ/r²
    s⁻⁶ = s⁻²*s⁻²*s⁻²
    ε4s⁻⁶ = atom_i.twice_sqrt_ε*atom_j.twice_sqrt_ε*s⁻⁶
    E = ε4s⁻⁶*(s⁻⁶ - 1)
    minus_E′r = 6ε4s⁻⁶*(2s⁻⁶ - 1)
    theta = 0.5f0(1 + sign(model.rc² - r²))
    return theta*E, theta*minus_E′r
    # x = (r² - model.rs²)*model.δ⁻²
    # x *= 0.5f0(sign(x) - sign(x-1))
    # x² = x*x
    # g = 1 + x*x²*(15x - 6x² - 10)
    # minus_g′r = 60x²*(1 - 2x + x²)*model.δ⁻²*r²
    # return E*g, minus_E′r*g + E*minus_g′r
end
