# Shy voter model: mean-field analysis

Code accompanying

> C. H. Kim, S. Park, and B. Kahng, *A Shy Voter Model: Entropy Effect on Opinion Formation* (2026).

The scripts derive, with computer algebra, the mean-field quantities reported in the main text and the Supplemental Material (SM) for neighborhood size $k=8$:

- the expansion coefficients $G_n$ and $C_n$, including the full numerators $B_3$, $B_4$, $B_5$ that are too long to print in the SM;
- the tricritical point $(p_h^t, p_v^t, Q_0^{h,t})$;
- the Jacobian $J_T$, the determinant identity, and the Lyapunov solution for $\mathrm{Var}(m)$;
- the linear response $\chi_{\mathrm{MF}}$ to the opinion and concealment fields;
- the numerical location of the spinodal point $p_h^{**}$.

## Requirements

- Python ≥ 3.9
- [Wolfram Engine](https://www.wolfram.com/engine/) 14.x (free for non-commercial use) or Mathematica
- `wolframclient` (`pip install -r requirements.txt`)

The scripts look for the Wolfram kernel at the default Linux install path. To use another location, set

```bash
export WOLFRAM_KERNEL=/path/to/WolframKernel
```

## Usage

Run the scripts from the repository root, in order. Script 01 must run first, because scripts 03 and 04 read $C_1$ from its output.

```bash
python mean_field/01_expansion_coefficients.py      # G_n, C_n, steady-state equations
python mean_field/02_tricritical_point.py           # tricritical point
python mean_field/03_linear_response.py             # susceptibility and response matrix
python mean_field/04_jacobian_fluctuations.py       # Jacobian, det identity, Var(m)
python mean_field/04_jacobian_fluctuations.py --spinodal 0.1   # adds p_h^** at p_v = 0.1
```

All outputs are written to `results/` as JSON. Expressions are stored as Wolfram `InputForm` strings and can be read back with `ToExpression` or parsed with SymPy.

## Repository layout

| File | What it computes | Paper |
|---|---|---|
| `mean_field/01_expansion_coefficients.py` | $\Psi^h$, $w$, $y(m)=G_2m^2+G_4m^4$, $C_1\ldots C_5$ | Main text, mean-field solution; SM Sec. S1 |
| `mean_field/02_tricritical_point.py` | $C_1=C_3=0$, $Q_0^h=\Psi^h$ | Main text; SM Sec. S1 |
| `mean_field/03_linear_response.py` | response matrix, $\chi_{\mathrm{MF}}=K_\chi/\det A$, amplitude ratio | SM Sec. S3 |
| `mean_field/04_jacobian_fluctuations.py` | $J_T$, $\det\partial(F_1,F_2)/\partial(Q^h,m)=2(1-Q^h)\det A$, Lyapunov solution, spinodal | SM Sec. S2 |
| `mean_field/config.py` | kernel path and output folder | — |
| `extras/entropy_flux_check.py` | single-point check of the entropy-flux sum at the symmetric fixed point | SM Sec. S4 |
| `extras/polling_bias.py` | leading-order polling bias $Q_0^h\,|2m|$ on the ordered branch (not reported in the paper) | — |
| `results/shy_voter_results_zero_field.json` | output of 01 | |
| `results/shy_voter_chi_general.json` | output of 03 | |
| `results/shy_voter_jacobian.json` | output of 04 | |
| `results/shy_voter_polling_bias.json` | output of `extras/polling_bias.py` | |

## Notation

The code predates the final notation of the paper. The correspondence is:

| Code | Paper | Meaning |
|---|---|---|
| `Ph`, `Pv` | $p_h$, $p_v$ | concealment and visibility probabilities |
| `Qh`, `Psi` | $Q^h$ | hidden fraction |
| `Qh0` | $Q_0^h$ | hidden fraction at the symmetric fixed point |
| `y` | $y=Q^h-Q_0^h$ | |
| `x` | $m$ | order parameter, $-1/2\le m\le 1/2$ |
| `Phi`, `Fi` | $2m$ | order parameter rescaled to $[-1,1]$ |
| `PsiPrime` | $\Psi^h$ | post-update hidden probability |
| `mAvgPrime`, `Phi` (in script 01) | $w$ | post-update mean signed state |
| `G2`, `G4` | $G_2$, $G_4$ | coefficients of $y(m)$ |
| `C1`, `C3`, `C5` | $C_1$, $C_3$, $C_5$ | coefficients of $G(m)$ |
| `h0`, `h` | $h_0$, $h$ | concealment and opinion fields |
| `NeighborhoodNumber` | $k$ | |
| `DomainWall` | — | fixed to 0 (majority threshold $\lceil v/2\rceil$) |

Because scripts 03 and 04 work with `Phi` $=2m$, variances there are $\mathrm{Var}(2m)=4\,\mathrm{Var}(m)$, and the single-node noise $(1-\Phi^2)/(1-Q^h)$ equals $4\times(1-4m^2)/[4(1-Q^h)]$, the $V^*_{22}$ of SM Sec. S2.

## Reproduced values

With the scripts above,

- $C_1 = (Q_0^h-1)(A_1+7p_hA_2)/8$ and $G_2$ agree term by term with the expressions in SM Sec. S1;
- $B_3$, $B_4$, $B_5$ have degrees $(13,2,1)$, $(19,2,2)$, $(26,4,3)$ in $(Q_0^h,p_h,p_v)$;
- the tricritical point is $(p_h^t,p_v^t,Q_0^{h,t})\simeq(0.610,\,0.240,\,0.714)$;
- `--spinodal 0.1` locates the spinodal used for the hybrid transition at $p_v=0.1$ ($p_h^{**}\simeq0.60$ in the paper).

## License

MIT; see `LICENSE`.

## Citation

If you use this code, please cite the paper above and this repository (see `CITATION.cff`).
