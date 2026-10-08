#!/usr/bin/env python3
"""
Shy Voter Model - divergence structure of the linear response (symbolic only)
=========================================================

At a general fixed point,

    F1 = Qh - Psi'(Qh, Phi; h0, h)
    F2 = (1 - Qh) Phi - <s'>(Qh, Phi; h0, h)

applying the implicit function theorem gives the response matrix

    d(Qh, Phi)/d(h0, h) = adj(dF) . S / det dF

        dF = [[ 1 - dPsi'/dQh ,        -dPsi'/dPhi        ],
              [ -Phi - d<s'>/dQh ,  (1 - Qh) - d<s'>/dPhi ]]
        S  = [[ dPsi'/dh0 , dPsi'/dh ],
              [ d<s'>/dh0 , d<s'>/dh ]]

All four components share the denominator det dF.  Moreover,

    det dF = (1 - Qh) det(I - J_T)

so the susceptibility diverges at the same point and in the same way as Var(Phi).

  * symmetric fixed point:  det dF = (1 - J11) C_1 / 2       -> chi ~ 1/C_1
  * ordered fixed point:    det dF -> 0 at the saddle-node   -> chi ~ (ph_sp - ph)^(-1/2)
  * in both cases N Var(Phi) / chi is finite (the denominators cancel)

Note: the free branch also responds to the field.  With the P_eff definition of the SM,
    the free-branch contribution to <s'> is (Pv/2)(e^h - e^{-h})/Zfree, which vanishes at h=0
    but has a finite h-derivative.  Zero-field results are unaffected.

STAGE 1  response matrix and common denominator
STAGE 2  det dF = (1 - Qh) det(I - J_T)
STAGE 3  reduction to Phi -> 0 and matching to C_1
STAGE 4  divergence exponents and amplitude ratio (abstract symbols)
"""

import json
import os
import time

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import KERNEL, results_path

A_RESULTS_FILE = results_path('shy_voter_results_zero_field.json')
OUT_FILE = results_path('shy_voter_chi_general.json')

NEIGHBOR = 8
DOMAIN_WALL = 0


def wl(session, code):
    res = session.evaluate(wlexpr(code))
    if hasattr(res, 'head') and str(getattr(res, 'head', '')) == 'Failure':
        raise RuntimeError(f"Wolfram evaluation failed:\n{code[:200]}\n-> {res}")
    return res


def wl_str(session, name):
    return session.evaluate(wlexpr(f'ToString[{name}, InputForm]'))


# ---------------------------------------------------------------- setup
def setup(session, verbose=True):
    if verbose:
        print("[setup] field-reweighted model (general Psi, Phi) ...", end=" ", flush=True)
    t0 = time.time()

    wl(session, f'NeighborhoodNumber = {NEIGHBOR}')
    wl(session, f'DomainWall = {DOMAIN_WALL}')

    wl(session, '''
        eh0e = Exp[h0];  ehpe = Exp[h];  ehme = Exp[-h];
        Zplus  = Ph*eh0e + (1 - Ph)*ehpe;
        Zminus = Ph*eh0e + (1 - Ph)*ehme;
        Zfree  = (1 - Pv)*eh0e + (Pv/2)*ehpe + (Pv/2)*ehme;
    ''')

    wl(session, '''
        SIPsiV[v_] :=
            Sum[
                With[{
                    prob = Binomial[v, m]*((1 + Phi)/2)^m*((1 - Phi)/2)^(v - m),
                    threshold = Ceiling[v/2 + DomainWall/2],
                    maj = Max[m, v - m]
                },
                    If[maj > threshold,
                        If[m >= v - m, (Ph*eh0e/Zplus)*prob,
                                       (Ph*eh0e/Zminus)*prob],
                        ((1 - Pv)*eh0e/Zfree)*prob]
                ], {m, 0, v}];
        PsiPrime = Sum[
            Binomial[NeighborhoodNumber, kh]*Psi^kh*
            (1 - Psi)^(NeighborhoodNumber - kh)*SIPsiV[NeighborhoodNumber - kh],
            {kh, 0, NeighborhoodNumber}];
    ''')

    wl(session, '''
        SIPhiV[v_] :=
            Sum[
                With[{
                    prob = Binomial[v, m]*((1 + Phi)/2)^m*((1 - Phi)/2)^(v - m),
                    threshold = Ceiling[v/2 + DomainWall/2],
                    maj = Max[m, v - m]
                },
                    If[maj > threshold,
                        If[m > v - m, ((1 - Ph)*ehpe/Zplus)*prob,
                            If[m < v - m, -((1 - Ph)*ehme/Zminus)*prob, 0]],
                        ((Pv/2)*(ehpe - ehme)/Zfree)*prob]
                ], {m, 0, v}];
        mAvgPrime = Sum[
            Binomial[NeighborhoodNumber, kh]*Psi^kh*
            (1 - Psi)^(NeighborhoodNumber - kh)*SIPhiV[NeighborhoodNumber - kh],
            {kh, 0, NeighborhoodNumber}];
    ''')

    if verbose:
        print(f"done ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- stage 1
def stage1_response(session, verbose=True, save_big=False):
    """Response matrix R = adj(dF).S / det dF.  Checks that there is a single denominator."""
    if verbose:
        print("\n[STAGE 1] response matrix and common denominator ...")
    t0 = time.time()

    wl(session, '''
        zf = {h -> 0, h0 -> 0};

        Smat = Simplify[{{D[PsiPrime,  h0], D[PsiPrime,  h]},
                         {D[mAvgPrime, h0], D[mAvgPrime, h]}} /. zf];

        PsQ = Simplify[D[PsiPrime,  Psi] /. zf];
        PsF = Simplify[D[PsiPrime,  Phi] /. zf];
        MsQ = Simplify[D[mAvgPrime, Psi] /. zf];
        MsF = Simplify[D[mAvgPrime, Phi] /. zf];

        dFmat = {{1 - PsQ, -PsF}, {-Phi - MsQ, (1 - Psi) - MsF}};
        detdF = Simplify[Det[dFmat]];
        adjdF = Simplify[detdF*Inverse[dFmat]];

        Rmat  = Simplify[Inverse[dFmat] . Smat];
        Rnum  = Simplify[adjdF . Smat];
        Rresidual = Simplify[Rmat - Rnum/detdF];
    ''')

    out = {'Rresidual': wl_str(session, 'Rresidual')}
    if save_big:
        for n in ['Smat', 'dFmat', 'detdF', 'adjdF', 'Rnum']:
            out[n] = wl_str(session, n)

    if verbose:
        print(f"   R - adj(dF).S/det dF = {out['Rresidual']}")
        print("   -> all four components share the single denominator det dF.")
        print(f"   ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------- stage 2
def stage2_det_identity(session, verbose=True):
    """det dF = (1 - Psi) det(I - J_T).  Checks that it equals the quantity in the fluctuation calculation."""
    if verbose:
        print("\n[STAGE 2] det dF = (1 - Psi) det(I - J_T) ...")

    wl(session, '''
        (* on the fixed point, 1 - Psi' = 1 - Psi *)
        J11g = PsQ;
        J12g = PsF;
        J21g = Together[(MsQ + Phi*PsQ)/(1 - Psi)];
        J22g = Together[(MsF + Phi*PsF)/(1 - Psi)];
        detIJ = Together[Det[IdentityMatrix[2] - {{J11g, J12g}, {J21g, J22g}}]];
        detIdentity = Simplify[detdF - (1 - Psi)*detIJ];
    ''')
    out = {'detIdentity': wl_str(session, 'detIdentity')}
    if verbose:
        print(f"   det dF - (1-Psi) det(I-J_T) = {out['detIdentity']}")
        print("   -> chi and Var(Phi) diverge at the same point.")
    return out


# ---------------------------------------------------------------- stage 3
def stage3_symmetric(session, c1_expr, verbose=True):
    """Phi -> 0: R becomes diagonal, chi = 2 S22 / C_1."""
    if verbose:
        print("\n[STAGE 3] reduction to Phi -> 0 and matching to C_1 ...")

    wl(session, '''
        sym = {Phi -> 0, Psi -> Qh0};
        SmatSym  = Simplify[Smat  /. sym];
        dFmatSym = Simplify[dFmat /. sym];
        RmatSym  = Simplify[Rmat  /. sym];

        S11 = Simplify[SmatSym[[1, 1]]];    (* dPsi'/dh0  -- hidden source *)
        S12 = SmatSym[[1, 2]];              (* dPsi'/dh   -> 0 (Z_2)  *)
        S21 = SmatSym[[2, 1]];              (* d<s'>/dh0  -> 0 (Z_2)  *)
        S22 = Simplify[SmatSym[[2, 2]]];    (* d<s'>/dh   -- opinion source *)

        (* factored form for the SM *)
        S11Factored = Factor[S11];
        S22Factored = Factor[S22];

        (* isolate the free-branch contribution: difference from the case with the free branch switched off *)
        S22PressureOnly = Simplify[S22 /. Pv -> 0];
        S22FreePart     = Simplify[S22 - S22PressureOnly];

        R12 = RmatSym[[1, 2]];              (* dQh/dh     -> 0 *)
        R21 = RmatSym[[2, 1]];              (* dPhi/dh0   -> 0 *)
        chiMF  = Simplify[RmatSym[[2, 2]]];
        chiHid = Simplify[RmatSym[[1, 1]]];
        detdFSym = Simplify[detdF /. sym];
    ''')

    out = {k: wl_str(session, k) for k in
           ['S11', 'S12', 'S21', 'S22',
            'S11Factored', 'S22Factored',
            'S22PressureOnly', 'S22FreePart',
            'R12', 'R21', 'chiMF', 'chiHid', 'detdFSym']}

    if verbose:
        print(f"   dPsi'/dh|sym  = {out['S12']}    (Z_2 -> 0)")
        print(f"   d<s'>/dh0|sym = {out['S21']}    (Z_2 -> 0)")
        print(f"   dQh/dh|sym    = {out['R12']}    (-> 0)")
        print(f"   dPhi/dh0|sym  = {out['R21']}    (-> 0)")
        print("   -> the response matrix is diagonal at the symmetric point.")
        print()
        print("   source terms (for the SM):")
        print(f"     S22 = d<s'>/dh   = {out['S22Factored']}")
        print(f"       pressure branch only     = {out['S22PressureOnly']}")
        print(f"       free-branch contribution = {out['S22FreePart']}")
        print(f"     S11 = dPsi'/dh0  = {out['S11Factored']}")

    if c1_expr is None:
        if verbose:
            print("   C_1 not provided -- matching check skipped")
        return out

    wl(session, f'C1FromA = {c1_expr};')
    wl(session, '''
        c1Diff       = Simplify[C1FromA - 2*dFmatSym[[2, 2]]];
        chiDiff      = Simplify[chiMF - 2*SmatSym[[2, 2]]/C1FromA];
        detdFSymDiff = Simplify[detdFSym - dFmatSym[[1, 1]]*C1FromA/2];
    ''')
    out['c1Diff'] = wl_str(session, 'c1Diff')
    out['chiDiff'] = wl_str(session, 'chiDiff')
    out['detdFSymDiff'] = wl_str(session, 'detdFSymDiff')

    if verbose:
        print(f"   C_1 - 2[(1-Qh)-d<s'>/dPhi]  = {out['c1Diff']}")
        print(f"   chi - 2 d<s'>/dh / C_1      = {out['chiDiff']}")
        print(f"   det dF - (1-J11) C_1/2      = {out['detdFSymDiff']}")
        ok = out['c1Diff'].strip() == '0' and out['chiDiff'].strip() == '0'
        print("   " + ("OK  continuous transition: chi ~ 1/C_1" if ok else "FAIL"))
    return out


# ---------------------------------------------------------------- stage 4
def stage4_exponents(session, verbose=True):
    """Fix the divergence exponents and the amplitude ratio with abstract symbols."""
    if verbose:
        print("\n[STAGE 4] divergence exponents and amplitude ratio (abstract symbols) ...")

    wl(session, '''
        (* A = I - J_T,  V* = diag(q1, q2),  closed-form solution of the continuous Lyapunov equation *)
        Amat = {{a11, a12}, {a21, a22}};
        Qmat = {{q1, 0}, {0, q2}};
        dA = Det[Amat];  tA = Tr[Amat];
        adjA = Simplify[dA*Inverse[Amat]];
        Vsol = Simplify[(dA*Qmat + adjA . Qmat . Transpose[adjA])/(2*dA*tA)];
        lyapRes = Simplify[Amat . Vsol + Vsol . Transpose[Amat] - Qmat];

        (* leading term as det -> 0 *)
        V22Lead = Simplify[(adjA . Qmat . Transpose[adjA])[[2, 2]]/(2*dA*tA)];
        KV = Simplify[V22Lead*dA];          (* amplitude excluding det *)

        (* leading term of chi:  chi = [adj(dF).S]_22 / det dF,
              since det dF = (1 - Qh) dA *)
        chiLead = kchi/((1 - Qh)*dA);
        Kchi = Simplify[chiLead*dA];

        (* amplitude ratio: det cancels, so it is finite *)
        ratioNVarChi = Simplify[V22Lead/chiLead];
        ratioHasDet  = Simplify[D[ratioNVarChi, dA]];
    ''')

    out = {
        'lyapRes': wl_str(session, 'lyapRes'),
        'V22Lead': wl_str(session, 'V22Lead'),
        'KV': wl_str(session, 'KV'),
        'ratioNVarChi': wl_str(session, 'ratioNVarChi'),
    }

    if verbose:
        print(f"   Lyapunov residual = {out['lyapRes']}")
        print(f"   V22 * det A   = {out['KV']}")
        print("       = (a21^2 q1 + a11^2 q2)/(2 tr A),  a finite amplitude independent of det A")
        print(f"   N Var / chi   = {out['ratioNVarChi']}")
        print("   -> det A cancels, so the amplitude ratio is finite on both the continuous line and the spinodal.")
        print("\n   Conclusion:")
        print("     chi, Var  ~  1 / det dF")
        print("       symmetric point: det dF = (1-J11) C_1/2      -> ~ 1/C_1")
        print("       spinodal       : det dF ~ (ph_sp - ph)^(1/2) -> ~ (ph_sp-ph)^(-1/2)")
    return out


# ---------------------------------------------------------------- driver
def main(verbose=True, save_big=False):
    c1_expr = None
    if os.path.exists(A_RESULTS_FILE):
        with open(A_RESULTS_FILE) as fh:
            c1_expr = json.load(fh)['C_strs']['C1']
    elif verbose:
        print(f"Warning: {A_RESULTS_FILE} not found -- C_1 matching in STAGE 3 skipped")

    session = WolframLanguageSession(kernel=KERNEL)
    session.start(timeout=120)
    try:
        setup(session, verbose)
        res = {}
        res['stage1'] = stage1_response(session, verbose, save_big)
        res['stage2'] = stage2_det_identity(session, verbose)
        res['stage3'] = stage3_symmetric(session, c1_expr, verbose)
        res['stage4'] = stage4_exponents(session, verbose)
        return res
    finally:
        session.terminate()


if __name__ == "__main__":
    res = main(verbose=True)

    with open(OUT_FILE, 'w') as fh:
        json.dump(res, fh, indent=2)

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  STAGE 1  R - adj(dF)S/det dF  : {res['stage1']['Rresidual']}")
    print(f"  STAGE 2  det identity residual : {res['stage2']['detIdentity']}")
    print(f"  STAGE 3  chi - 2S22/C_1       : {res['stage3'].get('chiDiff', 'n/a')}")
    print(f"           R12, R21             : {res['stage3']['R12']}, "
          f"{res['stage3']['R21']}")
    print(f"           S22 (d<s'>/dh)       : {res['stage3']['S22Factored']}")
    print(f"  STAGE 4  Lyapunov residual     : {res['stage4']['lyapRes']}")
    print(f"\nSaved: {OUT_FILE}")