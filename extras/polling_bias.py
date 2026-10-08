"""
Shy Voter Model - D-1: Polling bias (interpretation 1)

Bias = Q^h * Phi^*

1. Check that Qh0 = 0 at the trivial baseline (Ph=0, Pv=1)
2. Ordered branch leading form: Bias^2 ≈ Qh0^2 * (-4 C_1/C_3)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mean_field'))
from config import KERNEL, results_path


from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
import json


def derive_polling_bias(verbose=True):
    if verbose:
        print("="*60)
        print("Shy Voter Model - D-1: Polling bias")
        print("="*60)
        print()
    
    # Load C_1 and C_3 from the stage-A results
    with open(results_path('shy_voter_results_zero_field.json')) as f:
        a_data = json.load(f)
    c1_expr = a_data['C_strs']['C1']
    c3_expr = a_data['C_strs']['C3']
    
    session = WolframLanguageSession(
        kernel=KERNEL
    )
    session.start(timeout=60)
    
    try:
        session.evaluate(wlexpr('NeighborhoodNumber = 8'))
        session.evaluate(wlexpr('DomainWall = 0'))
        
        # ========= 1. Define PsiPrime =========
        if verbose:
            print("1. Defining PsiPrime ...")
        
        session.evaluate(wlexpr('''
            SIPsiVZF[v_] := 
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
            PsiPrime = Sum[
                Binomial[NeighborhoodNumber, kh] * Psi^kh * (1 - Psi)^(NeighborhoodNumber - kh) *
                SIPsiVZF[NeighborhoodNumber - kh],
                {kh, 0, NeighborhoodNumber}
            ];
        '''))
        
        # ========= 2. Self-consistency equation at the trivial baseline (Ph=0, Pv=1) =========
        if verbose:
            print("\n2. Checking the trivial baseline ...")
        
        session.evaluate(wlexpr('''
            (* RHS of the self-consistency equation at (Ph=0, Pv=1) *)
            PsiPrimeTrivial = Simplify[
                (PsiPrime /. {Phi -> 0, Psi -> Qh0}) /. {Ph -> 0, Pv -> 1}
            ];
            
            (* Solve Qh0 = PsiPrimeTrivial *)
            QhTrivialSolutions = Solve[Qh0 == PsiPrimeTrivial, Qh0];
        '''))
        
        ps_str = session.evaluate(wlexpr('ToString[PsiPrimeTrivial, InputForm]'))
        sols_str = session.evaluate(wlexpr('ToString[QhTrivialSolutions, InputForm]'))
        
        if verbose:
            print(f"   PsiPrime|(trivial baseline) = {ps_str}")
            print(f"   Qh0 solutions: {sols_str}")
        
        # ========= 3. Ordered branch leading form =========
        if verbose:
            print("\n3. Ordered branch leading form ...")
        
        session.evaluate(wlexpr(f'''
            C1FromA = {c1_expr};
            C3FromA = {c3_expr};
        '''))
        
        session.evaluate(wlexpr('''
            (* small-x ordered branch leading: x^2 = -C_1/C_3 *)
            xSqLeading = Simplify[-C1FromA / C3FromA];
            
            (* Phi = 2x ⇒ |Phi*|^2 = 4 x^2 = -4 C_1/C_3 *)
            PhiSqLeading = Simplify[4 * xSqLeading];
            
            (* Bias = Qh0 * Phi*, |Bias|^2 = Qh0^2 |Phi*|^2 *)
            BiasSqLeading = Simplify[Qh0^2 * PhiSqLeading];
        '''))
        
        # ========= 4. Extract results =========
        if verbose:
            print("\n4. Extracting results ...")
        
        results = {}
        names = ['PsiPrimeTrivial', 'QhTrivialSolutions',
                 'xSqLeading', 'PhiSqLeading', 'BiasSqLeading']
        for name in names:
            expr_str = session.evaluate(wlexpr(f'ToString[{name}, InputForm]'))
            results[name] = expr_str
            if verbose:
                print(f"   {name}: {len(expr_str)} chars")
        
        return results
    
    finally:
        session.terminate()
        if verbose:
            print("\nSession closed.")


def save_bias_results(results, filename=results_path('shy_voter_polling_bias.json')):
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {filename}")


def print_bias_result(results, name):
    if name in results:
        print(f"{name} =\n  {results[name]}\n")


if __name__ == "__main__":
    results = derive_polling_bias(verbose=True)
    save_bias_results(results)
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    for k in ['PsiPrimeTrivial', 'QhTrivialSolutions', 'BiasSqLeading']:
        print_bias_result(results, k)