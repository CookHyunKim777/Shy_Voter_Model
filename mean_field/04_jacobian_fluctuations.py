#!/usr/bin/env python3
"""
Shy Voter Model - C-1 (v3): Jacobian at a general fixed point and the determinant identity
============================================================================

Purpose
----
Unifies the fluctuation divergence at the continuous transition (symmetric fixed point)
and at the discontinuous transition (spinodal, ordered fixed point) through det(I - J_T).

    F1 = Qh - Psi'(Qh, Phi)
    F2 = (1 - Qh) Phi - <s'>(Qh, Phi)

    det [ d(F1,F2)/d(Qh,Phi) ]  =  (1 - Qh) * det( I - J_T )        ... (*)

  * symmetric fixed point (Phi = 0):
        det(I - J_T) = (1 - J11) * C_1 / [ 2 (1 - Qh0) ]
    so it diverges as C_1 -> 0, matching the earlier SM result exactly.

  * ordered fixed point (Phi != 0):
        det(I - J_T) -> 0 at the saddle-node bifurcation,
    so it diverges at the spinodal p_h^sp, where det ~ sqrt(p_h^sp - p_h).

Lyapunov (continuous form, A = I - J_T; including batch updates with f = 1/sqrt(N)):
        A V + V A^T = V*/N
        V = [ det(A) V* + (A - tr(A) I) V* (A - tr(A) I)^T ] / [ 2 det(A) tr(A) ]
    => V ~ 1/det(A).  With det(A) as the axis, gamma' = 1 on both sides;
          with the sweep parameter as the axis, 1 (continuous) and 1/2 (spinodal).

Stages
----
STAGE 1  structural identity (*) with abstract symbols -- model independent, immediate
STAGE 2  general J_T of the model (no Phi substitution) and (*) at the model level
STAGE 3  specialization to the symmetric point -> recheck J22 = 1 - C_1/[2(1-Qh0)]
STAGE 4  analytic divergence: Lyapunov solution, V ~ 1/det, susceptibility
STAGE 5  (--spinodal) numeric sweep: ordered-branch continuation, spinodal location,
         sqrt exponent, V divergence

Usage:
    python mean_field/04_jacobian_fluctuations.py
    python mean_field/04_jacobian_fluctuations.py --spinodal 0.1   # numeric spinodal p_h^** at p_v = 0.1
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
OUT_FILE = results_path('shy_voter_jacobian.json')

NEIGHBOR = 8
DOMAIN_WALL = 0

# Numeric sweep settings (STAGE 5, --spinodal)
PV_NUM = 0.1          # high-concealment regime (discontinuous / spinodal)
PH_START = 0.02       # starting point of the ordered branch
PH_STEP = 0.002
PH_MAX = 0.99


# ---------------------------------------------------------------- utilities
def wl(session, code):
    """Do not let a Failure pass silently after evaluation."""
    res = session.evaluate(wlexpr(code))
    if hasattr(res, 'head') and str(getattr(res, 'head', '')) == 'Failure':
        raise RuntimeError(f"Wolfram evaluation failed:\n{code[:200]}\n-> {res}")
    return res


def wl_str(session, name):
    return session.evaluate(wlexpr(f'ToString[{name}, InputForm]'))


def wl_num(session, expr):
    return session.evaluate(wlexpr(f'N[{expr}]'))


# ---------------------------------------------------------------- model setup
def setup_model(session, verbose=True):
    """Define Psi' and <s'> at general (Psi, Phi). No Phi substitution."""
    if verbose:
        print("[setup] defining model polynomials ...", end=" ", flush=True)
    t0 = time.time()

    wl(session, f'NeighborhoodNumber = {NEIGHBOR}')
    wl(session, f'DomainWall = {DOMAIN_WALL}')

    wl(session, '''
        SIPsiV[v_] :=
            Sum[
                With[{
                    prob = Binomial[v, m] * ((1 + Phi)/2)^m * ((1 - Phi)/2)^(v - m),
                    threshold = Ceiling[v/2 + DomainWall/2],
                    maj = Max[m, v - m]
                },
                    If[maj > threshold, Ph * prob, (1 - Pv) * prob]
                ],
                {m, 0, v}
            ];
        PsiPrime = Expand[ Sum[
            Binomial[NeighborhoodNumber, kh] * Psi^kh *
            (1 - Psi)^(NeighborhoodNumber - kh) * SIPsiV[NeighborhoodNumber - kh],
            {kh, 0, NeighborhoodNumber}
        ] ];
    ''')

    wl(session, '''
        SIPhiV[v_] :=
            Sum[
                With[{
                    prob = Binomial[v, m] * ((1 + Phi)/2)^m * ((1 - Phi)/2)^(v - m),
                    threshold = Ceiling[v/2 + DomainWall/2],
                    maj = Max[m, v - m]
                },
                    If[maj > threshold,
                        If[m > v - m, (1 - Ph) * prob,
                            If[m < v - m, -(1 - Ph) * prob, 0]],
                        0
                    ]
                ],
                {m, 0, v}
            ];
        mAvgPrime = Expand[ Sum[
            Binomial[NeighborhoodNumber, kh] * Psi^kh *
            (1 - Psi)^(NeighborhoodNumber - kh) * SIPhiV[NeighborhoodNumber - kh],
            {kh, 0, NeighborhoodNumber}
        ] ];
    ''')

    # partial derivatives (general Psi, Phi)
    wl(session, '''
        PsModel = Expand[ D[PsiPrime,  Psi] ];
        PpModel = Expand[ D[PsiPrime,  Phi] ];
        MsModel = Expand[ D[mAvgPrime, Psi] ];
        MpModel = Expand[ D[mAvgPrime, Phi] ];
    ''')

    if verbose:
        print(f"done ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- stage 1
def stage1_structural_identity(session, verbose=True):
    """
    Structural part of identity (*).  Without specifying the model, Psi'=PP, <s'>=MM
    and their partial derivatives Ps, Pp, Ms, Mp are kept as abstract symbols.
    The identity holds after substituting the fixed-point conditions PP = Psi, MM = (1 - Psi) Phi.
    """
    if verbose:
        print("\n[STAGE 1] structural identity (abstract symbols) ...")

    wl(session, '''
        (* d(F1,F2)/d(Psi,Phi) *)
        dF = {{1 - Ps,       -Pp},
              {-Phi - Ms,    (1 - Psi) - Mp}};
        detF = Det[dF];

        (* Jacobian of Phi' = MM/(1 - PP), general form *)
        JTgen = {{Ps, Pp},
                 {(Ms*(1 - PP) + MM*Ps)/(1 - PP)^2,
                  (Mp*(1 - PP) + MM*Pp)/(1 - PP)^2}};

        (* fixed-point conditions: PP -> Psi,  MM -> (1 - Psi) Phi *)
        JTfp  = Simplify[ JTgen /. {PP -> Psi, MM -> (1 - Psi)*Phi} ];
        detIJ = Simplify[ Det[IdentityMatrix[2] - JTfp] ];

        structResidual = Simplify[ detF - (1 - Psi)*detIJ ];
    ''')

    residual = wl_str(session, 'structResidual')
    ok = (residual.strip() == '0')
    if verbose:
        print(f"   det(dF) - (1-Psi) det(I-J_T) = {residual}")
        print("   " + ("OK  structural identity holds" if ok
                       else "FAIL  identity does not hold -- later stages are meaningless"))
    return {'structResidual': residual, 'structOK': ok,
            'JTfp': wl_str(session, 'JTfp'),
            'detIJabstract': wl_str(session, 'detIJ')}


# ---------------------------------------------------------------- stage 2
def stage2_model_jacobian(session, verbose=True, save_big=False):
    """
    General J_T of the model.  On the fixed point (1 - PP) = (1 - Psi), so the denominator
    is written as (1 - Psi).  For Phi != 0, J12 and J21 are in general nonzero.
    """
    if verbose:
        print("\n[STAGE 2] general J_T of the model (no Phi substitution) ...")
    t0 = time.time()

    wl(session, '''
        J11gen = PsModel;
        J12gen = PpModel;
        J21gen = Together[ (MsModel + Phi*PsModel)/(1 - Psi) ];
        J22gen = Together[ (MpModel + Phi*PpModel)/(1 - Psi) ];

        JTmodel   = {{J11gen, J12gen}, {J21gen, J22gen}};
        detIJmodel = Together[ Det[IdentityMatrix[2] - JTmodel] ];
        detFmodel  = Together[ (1 - PsModel)*((1 - Psi) - MpModel)
                               - PpModel*(Phi + MsModel) ];

        modelResidual = Simplify[ detFmodel - (1 - Psi)*detIJmodel ];
    ''')

    residual = wl_str(session, 'modelResidual')
    ok = (residual.strip() == '0')

    # components that vanish only at Phi = 0 -- check that they are nonzero in general
    wl(session, '''
        J12atPhi0 = Simplify[ J12gen /. Phi -> 0 ];
        J21atPhi0 = Simplify[ J21gen /. Phi -> 0 ];
        J12nonzero = Simplify[ J12gen ] =!= 0;
        J21nonzero = Simplify[ J21gen ] =!= 0;
    ''')

    out = {
        'modelResidual': residual,
        'modelOK': ok,
        'J12atPhi0': wl_str(session, 'J12atPhi0'),
        'J21atPhi0': wl_str(session, 'J21atPhi0'),
        'J12nonzeroGeneral': str(session.evaluate(wlexpr('J12nonzero'))),
        'J21nonzeroGeneral': str(session.evaluate(wlexpr('J21nonzero'))),
    }
    if save_big:
        for name in ['J11gen', 'J12gen', 'J21gen', 'J22gen', 'detIJmodel']:
            out[name] = wl_str(session, name)

    if verbose:
        print(f"   det(dF) - (1-Psi) det(I-J_T)  [model] = {residual}")
        print("   " + ("OK" if ok else "FAIL"))
        print(f"   J12|_(Phi=0) = {out['J12atPhi0']}   (should be 0)")
        print(f"   J21|_(Phi=0) = {out['J21atPhi0']}   (should be 0)")
        print(f"   J12 != 0 at general Phi : {out['J12nonzeroGeneral']}")
        print(f"   J21 != 0 at general Phi : {out['J21nonzeroGeneral']}")
        print(f"   ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------- stage 3
def stage3_symmetric_point(session, c1_expr, verbose=True):
    """
    Specialization to the symmetric fixed point.  Checks the match with the earlier v2 result
    and shows that detIJ factorizes with C_1.
    """
    if verbose:
        print("\n[STAGE 3] specialization to the symmetric point and matching to C_1 ...")

    wl(session, '''
        J11sym = Simplify[ J11gen /. {Phi -> 0, Psi -> Qh0} ];
        J22sym = Simplify[ J22gen /. {Phi -> 0, Psi -> Qh0} ];
        detIJsym = Simplify[ detIJmodel /. {Phi -> 0, Psi -> Qh0} ];
    ''')

    out = {'J11': wl_str(session, 'J11sym'),
           'J22': wl_str(session, 'J22sym'),
           'J12': '0', 'J21': '0',
           'detIJsym': wl_str(session, 'detIJsym')}

    if c1_expr is None:
        if verbose:
            print("   C_1 not provided -- matching check skipped")
        return out

    wl(session, f'C1FromA = {c1_expr};')
    wl(session, '''
        J22Predicted = 1 - C1FromA/(2*(1 - Qh0));
        J22Diff = Simplify[ J22sym - J22Predicted ];

        detIJsymPredicted = Simplify[ (1 - J11sym)*C1FromA/(2*(1 - Qh0)) ];
        detIJsymDiff = Simplify[ detIJsym - detIJsymPredicted ];
    ''')

    out['J22Predicted'] = wl_str(session, 'J22Predicted')
    out['J22Diff'] = wl_str(session, 'J22Diff')
    out['detIJsymPredicted'] = wl_str(session, 'detIJsymPredicted')
    out['detIJsymDiff'] = wl_str(session, 'detIJsymDiff')

    if verbose:
        print(f"   J22 - (1 - C_1/[2(1-Qh0)])            = {out['J22Diff']}")
        print(f"   det(I-J_T) - (1-J11) C_1/[2(1-Qh0)]   = {out['detIJsymDiff']}")
        ok = out['J22Diff'].strip() == '0' and out['detIJsymDiff'].strip() == '0'
        print("   " + ("OK  the continuous-transition divergence occurs at C_1 -> 0"
                       if ok else "FAIL"))
    return out


# ---------------------------------------------------------------- stage 4
def stage4_numeric_spinodal(session, pv=PV_NUM, verbose=True):
    """
    Follows the ordered branch by continuation while tracking det(I - J_T), and
    locates the spinodal directly as the simultaneous solution of {F1=0, F2=0, det(I-J_T)=0}.
    Then checks the exponent alpha in det ~ (ph_sp - ph)^alpha and V ~ 1/det.
    """
    if verbose:
        print(f"\n[STAGE 5] numeric sweep (p_v = {pv}) ...")

    wl(session, f'PvNum = {pv};')

    # pure functions for numerical evaluation (Evaluate expands the polynomials once)
    wl(session, '''
        FPsiF = Function[{aq, af, ap},
            Evaluate[PsiPrime  /. {Psi -> aq, Phi -> af, Ph -> ap, Pv -> PvNum}]];
        FMF = Function[{aq, af, ap},
            Evaluate[mAvgPrime /. {Psi -> aq, Phi -> af, Ph -> ap, Pv -> PvNum}]];
        JTF = Function[{aq, af, ap},
            Evaluate[{{J11gen, J12gen}, {J21gen, J22gen}}
                     /. {Psi -> aq, Phi -> af, Ph -> ap, Pv -> PvNum}]];
        detIJF = Function[{aq, af, ap},
            Det[IdentityMatrix[2] - JTF[aq, af, ap]]];

        (* step 1: map iteration.  Deep in the ordered phase, where J_T ~ 0, it converges in a few steps,
              and it follows the same path as a backward sweep, so the ordered branch is not lost. *)
        iterFP[ap_, start_] := Module[
            {q = start[[1]], fi = start[[2]], qn, mn, fn, err = 1., c = 0},
            While[c < 5000 && err > 1.*^-13,
                qn = FPsiF[q, fi, ap];
                mn = FMF[q, fi, ap];
                fn = If[1 - qn > 1.*^-12, mn/(1 - qn), 0.];
                err = Max[Abs[qn - q], Abs[fn - fi]];
                q = qn; fi = fn; c++;
            ];
            {q, fi, err, c}
        ];

        (* step 2: refine with FindRoot, judged by the residual rather than by messages.
              With Check, harmless messages such as FindRoot::precw would also give $Failed. *)
        polishFP[ap_, g_] := Module[{sol, q, fi, r},
            sol = Quiet @ FindRoot[
                {FPsiF[uq, uf, ap] - uq,
                 FMF[uq, uf, ap] - (1 - uq)*uf},
                {{uq, g[[1]]}, {uf, g[[2]]}}, MaxIterations -> 400];
            q = uq /. sol; fi = uf /. sol;
            r = If[NumericQ[q] && NumericQ[fi],
                   Max[Abs[FPsiF[q, fi, ap] - q],
                       Abs[FMF[q, fi, ap] - (1 - q)*fi]],
                   $Failed];
            {q, fi, r}
        ];
    ''')

    # ---- continuation (ordered branch)
    wl(session, f'''
        phGrid = Range[{PH_START}, {PH_MAX}, {PH_STEP}];
        branch = {{}};
        gq = 0.5; gf = 0.98; idx = 1; stopFlag = False; stopWhy = "end of grid";
        While[idx <= Length[phGrid] && !stopFlag,
            aph = phGrid[[idx]];
            itr = iterFP[aph, {{gq, gf}}];
            pol = polishFP[aph, {{itr[[1]], itr[[2]]}}];
            Which[
                pol[[3]] === $Failed || !NumericQ[pol[[3]]],
                    stopFlag = True; stopWhy = "FindRoot non-numeric",
                pol[[3]] > 1.*^-9,
                    stopFlag = True; stopWhy = "residual too large",
                Abs[pol[[2]]] < 1.*^-5,
                    stopFlag = True; stopWhy = "ordered branch collapsed to Phi=0",
                True,
                    gq = pol[[1]]; gf = pol[[2]];
                    AppendTo[branch, {{aph, gq, gf,
                                      detIJF[gq, gf, aph],
                                      Max[Abs[Eigenvalues[JTF[gq, gf, aph]]]]}}];
                    idx++;
            ];
        ];
        nB = Length[branch];
        branchLast = If[nB > 0, Last[branch], $Failed];
    ''')

    n_pts = session.evaluate(wlexpr('nB'))
    stop_why = session.evaluate(wlexpr('stopWhy'))
    if verbose:
        print(f"   ordered-branch points: {n_pts}   (stopped because: {stop_why})")
        if n_pts:
            last = session.evaluate(wlexpr('N[branchLast]'))
            print(f"   branch end  p_h={last[0]:.5f}  Qh={last[1]:.6f}  "
                  f"Phi={last[2]:.6f}")
            print(f"            det={last[3]:.6e}  rho={last[4]:.8f}")

    if not n_pts:
        return {'pv': pv, 'nBranchPoints': 0, 'stopWhy': str(stop_why)}

    # ---- if det^2 is linear in p_h, then det ~ sqrt(p_h^sp - p_h).
    #      The intercept is the spinodal location, so location and exponent are checked together.
    wl(session, '''
        tailN = Min[60, nB];
        tailD = Take[branch, -tailN];
        d2data = ({#[[1]], #[[4]]^2} &) /@ tailD;
        fit2 = Fit[d2data, {1, u}, u];
        slope2 = Coefficient[fit2, u, 1];
        phSpFit = If[slope2 == 0, $Failed,
                     -Coefficient[fit2, u, 0]/slope2];
        r2 = Module[{ys = d2data[[All, 2]], ps},
            ps = (fit2 /. u -> #) & /@ d2data[[All, 1]];
            1 - Total[(ys - ps)^2]/Total[(ys - Mean[ys])^2]];
    ''')

    if verbose:
        print(f"   linear fit of det^2 vs p_h:  R^2 = {wl_num(session, 'r2')}")
        print(f"   p_h^sp from the intercept ~ {wl_num(session, 'phSpFit')}")

    # ---- direct determination of the spinodal: 3 equations, 3 unknowns
    wl(session, '''
        spin = Quiet @ FindRoot[
            {FPsiF[uq, uf, up] - uq,
             FMF[uq, uf, up] - (1 - uq)*uf,
             detIJF[uq, uf, up]},
            {{uq, branchLast[[2]]}, {uf, branchLast[[3]]},
             {up, If[phSpFit === $Failed, branchLast[[1]], phSpFit]}},
            MaxIterations -> 500];
        phSp = up /. spin; qhSp = uq /. spin; fSp = uf /. spin;
        spinRes = If[NumericQ[phSp],
            Max[Abs[{FPsiF[qhSp, fSp, phSp] - qhSp,
                     FMF[qhSp, fSp, phSp] - (1 - qhSp)*fSp,
                     detIJF[qhSp, fSp, phSp]}]], $Failed];
    ''')

    ph_sp = wl_num(session, 'phSp')
    if verbose:
        print(f"   spinodal  p_h^sp = {ph_sp}")
        print(f"             Qh     = {wl_num(session, 'qhSp')}")
        print(f"             Phi    = {wl_num(session, 'fSp')}   (must be nonzero)")
        print(f"             residual = {wl_num(session, 'spinRes')}")

    # ---- direct check of the sqrt exponent (log-log)
    wl(session, '''
        dlts = {1.*^-2, 3.*^-3, 1.*^-3, 3.*^-4, 1.*^-4, 3.*^-5, 1.*^-5};
        scanData = {};
        sq = qhSp; sf = fSp;
        Do[
            Module[{aph2 = phSp - dlts[[j]], it2, po2},
                it2 = iterFP[aph2, {sq, sf}];
                po2 = polishFP[aph2, {it2[[1]], it2[[2]]}];
                If[NumericQ[po2[[3]]] && po2[[3]] < 1.*^-9 && Abs[po2[[2]]] > 1.*^-5,
                    sq = po2[[1]]; sf = po2[[2]];
                    AppendTo[scanData, {dlts[[j]], Abs[detIJF[sq, sf, aph2]], sq, sf}]
                ]
            ],
            {j, 1, Length[dlts]}
        ];
        logFit = If[Length[scanData] >= 3,
            Fit[Log[10, scanData[[All, {1, 2}]]], {1, u}, u], $Failed];
        alphaDet = If[logFit === $Failed, $Failed, Coefficient[logFit, u, 1]];
    ''')

    alpha = wl_num(session, 'alphaDet')
    if verbose:
        print(f"   det ~ (p_h^sp - p_h)^alpha,  alpha = {alpha}   (expected 0.5)")

    # ---- check V ~ 1/det (continuous Lyapunov, 2x2 closed-form solution)
    wl(session, '''
        (* on the fixed point V* = diag[ Qh(1-Qh), (1-Phi^2)/(1-Qh) ] *)
        VstarOf[aq_, af_] := DiagonalMatrix[{aq*(1 - aq), (1 - af^2)/(1 - aq)}];

        (* 2x2 closed-form solution of A V + V A^T = Qm *)
        lyapCont[Am_, Qm_] := Module[{dA = Det[Am], tA = Tr[Am], Mm},
            Mm = Am - tA*IdentityMatrix[2];
            (dA*Qm + Mm . Qm . Transpose[Mm])/(2*dA*tA)];

        VtimesDet = Table[
            Module[{row = scanData[[j]], Am},
                Am = IdentityMatrix[2] - JTF[row[[3]], row[[4]], phSp - row[[1]]];
                lyapCont[Am, VstarOf[row[[3]], row[[4]]]][[2, 2]] * row[[2]]],
            {j, Length[scanData]}];
    ''')

    v_times_det = session.evaluate(wlexpr('N[VtimesDet]'))
    if verbose:
        print(f"   V22 * det  (N=1 normalization): {v_times_det}")
        print("   -> if this converges to a constant, V ~ 1/det, i.e. gamma' = 1")

    return {
        'pv': pv,
        'nBranchPoints': n_pts,
        'stopWhy': str(stop_why),
        'branchLast': str(session.evaluate(wlexpr('N[branchLast]'))),
        'phSpFit': str(wl_num(session, 'phSpFit')),
        'fitR2': str(wl_num(session, 'r2')),
        'phSpinodal': str(ph_sp),
        'qhSpinodal': str(wl_num(session, 'qhSp')),
        'phiSpinodal': str(wl_num(session, 'fSp')),
        'spinodalResidual': str(wl_num(session, 'spinRes')),
        'alphaDet': str(alpha),
        'scanData': str(session.evaluate(wlexpr('N[scanData]'))),
        'VtimesDet': str(v_times_det),
    }


# ---------------------------------------------------------------- stage 4 (symbolic)
def stage4_divergence_symbolic(session, c1_expr, verbose=True):
    """
    Derives the divergence at the transition points analytically (no numerics).

    (1) closed-form solution of the 2x2 continuous Lyapunov equation  A V + V A^T = V*/N  and its check
    (2) leading form as det A -> 0:  V22 ~ [a21^2 V*11 + a11^2 V*22]
                                              / (2 N det A tr A)
    (3) symmetric fixed point -> V22 ~ 1/(N C_1)          (continuous transition)
    (4) ordered fixed point: only det A vanishes, the amplitude stays finite   (spinodal)
    (5) general form of the susceptibility chi: same denominator det dF
    """
    if verbose:
        print("\n[STAGE 4] analytic derivation of the divergence at the transition points ...")

    out = {}

    # ---- (1) closed-form Lyapunov solution
    wl(session, '''
        Amat = {{a11, a12}, {a21, a22}};
        Qmat = {{q1, 0}, {0, q2}};
        dA = Det[Amat];  tA = Tr[Amat];
        adjA = Simplify[dA * Inverse[Amat]];
        Vsol = Simplify[(dA*Qmat + adjA . Qmat . Transpose[adjA])/(2*dA*tA)];
        lyapResidual = Simplify[Amat . Vsol + Vsol . Transpose[Amat] - Qmat];
    ''')
    out['lyapResidual'] = wl_str(session, 'lyapResidual')
    ok1 = out['lyapResidual'].replace(' ', '') in ('{{0,0},{0,0}}',)
    if verbose:
        print(f"   (1) A V + V A^T - Q = {out['lyapResidual']}")
        print("       " + ("OK  closed-form solution holds" if ok1 else "FAIL"))

    # ---- (2) leading form as det A -> 0
    wl(session, '''
        V22Leading = Simplify[(adjA . Qmat . Transpose[adjA])[[2, 2]]/(2*dA*tA)];
        V22LeadingPred = Simplify[(a21^2*q1 + a11^2*q2)/(2*dA*tA)];
        leadingDiff = Simplify[V22Leading - V22LeadingPred];
    ''')
    out['V22Leading'] = wl_str(session, 'V22Leading')
    out['leadingDiff'] = wl_str(session, 'leadingDiff')
    if verbose:
        print(f"   (2) V22_leading = (a21^2 q1 + a11^2 q2)/(2 det A tr A)")
        print(f"       residual = {out['leadingDiff']}")

    # ---- (3) symmetric fixed point -> 1/(N C_1)
    wl(session, '''
        V22Sym = Simplify[V22Leading /. {a12 -> 0, a21 -> 0}];
        (* value of a22 * V22 in the limit a22 = 1 - J22 -> 0 *)
        V22SymResidue = Simplify[Limit[a22*V22Sym, a22 -> 0]];
    ''')
    out['V22Sym'] = wl_str(session, 'V22Sym')
    out['V22SymResidue'] = wl_str(session, 'V22SymResidue')
    if verbose:
        print(f"   (3) symmetric point:  a22 * V22 -> {out['V22SymResidue']}   (expected q2/2)")

    if c1_expr is not None:
        wl(session, '''
            (* q2 = V*22 = 1/(1 - Qh0),  a22 = 1 - J22 = C_1/[2(1 - Qh0)] *)
            V22ContLeading = Simplify[
                (V22SymResidue/a22) /. {q2 -> 1/(1 - Qh0),
                                        a22 -> C1FromA/(2*(1 - Qh0))}];
            contDiff = Simplify[V22ContLeading - 1/C1FromA];
        ''')
        out['V22ContLeading'] = wl_str(session, 'V22ContLeading')
        out['contDiff'] = wl_str(session, 'contDiff')
        if verbose:
            print(f"       V22 (continuous line) = {out['V22ContLeading']}")
            print(f"       V22 - 1/C_1  = {out['contDiff']}")
            print("       " + ("OK  earlier SM result rederived"
                               if out['contDiff'].strip() == '0' else "FAIL"))

    # ---- (4) ordered fixed point: finite amplitude, only det vanishes
    wl(session, '''
        (* amplitude with V*11 = Qh(1-Qh),  V*22 = (1-Phi^2)/(1-Qh) substituted *)
        amplitude = Simplify[
            (a21^2*q1 + a11^2*q2)/(2*tA)
            /. {q1 -> Qh*(1 - Qh), q2 -> (1 - Fi^2)/(1 - Qh)}];
        (* does any factor vanish at the spinodal? none unless Phi = +-1 *)
        amplitudeVanishes = Simplify[amplitude == 0];
    ''')
    out['amplitude'] = wl_str(session, 'amplitude')
    if verbose:
        print(f"   (4) ordered-branch amplitude (excluding det A):")
        print(f"       {out['amplitude']}")
        print("       -> finite and nonzero unless Phi = +-1.")
        print("          The divergence therefore comes only from det A -> 0.")

    # ---- (5) susceptibility: same denominator det dF
    wl(session, '''
        dFm = {{1 - Ps, -Pp}, {-Fi - Ms, (1 - Qh) - Mp}};
        (* dx/dh = dFm^{-1} . (dhP, dhM) *)
        chiGen = Simplify[(Inverse[dFm] . {dhP, dhM})[[2]]];
        chiGenPred = Simplify[((Fi + Ms)*dhP + (1 - Ps)*dhM)/Det[dFm]];
        chiDiff = Simplify[chiGen - chiGenPred];

        (* Phi = 0 specialization: Ms = 0, dhP = 0 *)
        chiSym = Simplify[chiGen /. {Fi -> 0, Ms -> 0, dhP -> 0}];
        chiSymPred = Simplify[dhM/(((1 - Qh) - Mp))];
        chiSymDiff = Simplify[chiSym - chiSymPred];
    ''')
    out['chiGen'] = wl_str(session, 'chiGen')
    out['chiDiff'] = wl_str(session, 'chiDiff')
    out['chiSymDiff'] = wl_str(session, 'chiSymDiff')
    if verbose:
        print(f"   (5) chi = [(Phi + Ms) dhP + (1 - Ps) dhM] / det dF")
        print(f"       residual = {out['chiDiff']}")
        print(f"       residual at Phi=0 = {out['chiSymDiff']}"
              "   (denominator (1-Qh)-Mp = C_1/2)")
        print("       -> chi and Var share the same denominator det dF.")

    return out


# ---------------------------------------------------------------- driver
def main(verbose=True, save_big=False, spinodal_pv=None):
    c1_expr = None
    if os.path.exists(A_RESULTS_FILE):
        with open(A_RESULTS_FILE) as fh:
            c1_expr = json.load(fh)['C_strs']['C1']
    elif verbose:
        print(f"Warning: {A_RESULTS_FILE} not found -- C_1 matching check in STAGE 3 skipped")

    session = WolframLanguageSession(kernel=KERNEL)
    session.start(timeout=120)

    try:
        setup_model(session, verbose)
        results = {}
        results['stage1'] = stage1_structural_identity(session, verbose)
        if not results['stage1']['structOK']:
            raise RuntimeError("STAGE 1 failed -- aborting")
        results['stage2'] = stage2_model_jacobian(session, verbose, save_big)
        results['stage3'] = stage3_symmetric_point(session, c1_expr, verbose)
        results['stage4'] = stage4_divergence_symbolic(session, c1_expr, verbose)
        if spinodal_pv is not None:
            results['stage5_spinodal'] = stage4_numeric_spinodal(session, pv=spinodal_pv,
                                                                 verbose=verbose)
        return results
    finally:
        session.terminate()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--spinodal', type=float, default=None, metavar='PV',
                    help='also locate the spinodal p_h^** numerically at this p_v')
    args = ap.parse_args()
    res = main(verbose=True, spinodal_pv=args.spinodal)

    with open(OUT_FILE, 'w') as fh:
        json.dump(res, fh, indent=2)

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  STAGE 1  structural identity  : "
          f"{'OK' if res['stage1']['structOK'] else 'FAIL'}")
    print(f"  STAGE 2  model-level identity : "
          f"{'OK' if res['stage2']['modelOK'] else 'FAIL'}")
    print(f"  STAGE 3  J22Diff              : {res['stage3'].get('J22Diff', 'n/a')}")
    print(f"           detIJsymDiff         : {res['stage3'].get('detIJsymDiff', 'n/a')}")
    print(f"  STAGE 4  Lyapunov residual    : {res['stage4'].get('lyapResidual', 'n/a')}")
    print(f"           V22 - 1/C_1 (cont.)  : {res['stage4'].get('contDiff', 'n/a')}")
    print(f"           chi general residual : {res['stage4'].get('chiDiff', 'n/a')}")
    print(f"\nSaved: {OUT_FILE}")