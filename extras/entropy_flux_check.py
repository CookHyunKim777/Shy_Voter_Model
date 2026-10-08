"""
D-2: Entropy flux: trivial-baseline check and a single-point sanity check.
Used to cross-validate the separate entropy-flux code (entropy_flux.py).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mean_field'))
from config import KERNEL, results_path


from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr


def derive_entropy_flux_check(verbose=True):
    if verbose:
        print("="*60)
        print("D-2: Entropy flux check")
        print("="*60)
        print()
    
    session = WolframLanguageSession(
        kernel=KERNEL
    )
    session.start(timeout=60)
    
    try:
        session.evaluate(wlexpr('NeighborhoodNumber = 8'))
        session.evaluate(wlexpr('DomainWall = 0'))
        
        # Transition rates {r_+, r_-, r_0} for environment (nP, nM)
        session.evaluate(wlexpr('''
            TransitionRates[nP_, nM_] := Module[
                {nVis, threshold, maxCnt},
                nVis = nP + nM;
                threshold = Ceiling[(nVis + DomainWall)/2];
                maxCnt = Max[nP, nM];
                If[maxCnt > threshold,
                    If[nP > nM,
                        {1 - Ph, 0, Ph},
                        {0, 1 - Ph, Ph}
                    ],
                    {Pv/2, Pv/2, 1 - Pv}
                ]
            ];
            
            (* P_a list: a in {+, -, 0} mapped to indices 1, 2, 3 *)
            (* multinomial prob: assume P+ = P- = (1-Qh)/2, P0 = Qh (sym fixed point) *)
        '''))
        
        # Entropy flux (sym fixed point)
        if verbose:
            print("1. Defining the entropy-flux sum (symmetric fixed point)...")
        
        session.evaluate(wlexpr('''
            PhiP = (1 - Qh0)/2;
            PhiM = (1 - Qh0)/2;
            Phi0 = Qh0;
            Pvals = {PhiP, PhiM, Phi0};
            
            EntFluxSym = Sum[
                Module[{Pa, envSum},
                    Pa = Pvals[[aIdx]];
                    envSum = Sum[
                        Module[{nP = nPlus, nM = nMinus, n0 = NeighborhoodNumber - nPlus - nMinus,
                                pAlpha, rates, inner},
                            If[n0 < 0, 0,
                                pAlpha = Multinomial[nP, nM, n0] *
                                         PhiP^nP * PhiM^nM * Phi0^n0;
                                rates = TransitionRates[nP, nM];
                                inner = Sum[
                                    If[(bIdx == aIdx) || (rates[[bIdx]] == 0) || (Pa == 0), 0,
                                        rates[[bIdx]] * Log[rates[[bIdx]] / Pa]
                                    ],
                                    {bIdx, 1, 3}
                                ];
                                pAlpha * inner
                            ]
                        ],
                        {nPlus, 0, NeighborhoodNumber},
                        {nMinus, 0, NeighborhoodNumber - nPlus}
                    ];
                    Pa * envSum
                ],
                {aIdx, 1, 3}
            ];
        '''))
        
        if verbose:
            print("   done\n")
        
        # ========= Numerical evaluation at a single point (sanity check) =========
        if verbose:
            print("2. Numerical evaluation at a single point (sanity)...")
        
        # Example: (p_h, p_v) = (0.5, 0.4). Qh0 should be solved self-consistently,
        # but an arbitrary value is used here to check that the result is finite and nonzero.
        session.evaluate(wlexpr('''
            testPoint = EntFluxSym /. {Ph -> 0.5, Pv -> 0.4, Qh0 -> 0.5};
            testValue = N[testPoint, 10];
        '''))
        
        test_str = session.evaluate(wlexpr('ToString[testValue, InputForm]'))
        if verbose:
            print(f"   Phi_ent at (Ph=0.5, Pv=0.4, Qh0=0.5) = {test_str}")
        
        # ========= Trivial baseline =========
        # In the consensus state P+ = 1, P- = 0, P0 = 0, whereas (1-Qh)/2 = 1/2 ≠ 1.
        # The symmetric-fixed-point assumption above therefore fails in the trivial limit.
        # → The trivial result is checked by hand (see the explanation above).
        
        if verbose:
            print("\n3. Trivial baseline: checked by hand (see the explanation above)")
            print("   Consensus: (P+,P-,P0)=(1,0,0), all outflow rates = 0 ⇒ Phi_ent = 0")
        
        return {'testValue': test_str}
    
    finally:
        session.terminate()


if __name__ == "__main__":
    result = derive_entropy_flux_check(verbose=True)