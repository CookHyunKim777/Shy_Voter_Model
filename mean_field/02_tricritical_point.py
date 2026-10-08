"""
Tricritical point of the shy voter model (k = 8).
Solves C_1 = C_3 = 0 together with Q_0^h = Psi^h(Q_0^h) for (Ph, Pv, Qh0).
Output: results/tricritical_point.json
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import KERNEL, results_path

def find_tricritical_point(initial_guess=(0.6, 0.25, 0.5), 
                            precision=50, verbose=True):
    """
    Tricritical point: C_1 = C_3 = 0, Q_{h0} = Psi^h(Q_{h0})
    Three equations, three unknowns (Ph, Pv, Qh0).
    """
    from wolframclient.evaluation import WolframLanguageSession
    from wolframclient.language import wlexpr
    
    session = WolframLanguageSession(
        kernel=KERNEL
    )
    session.start(timeout=60)
    
    try:
        # ----- Setup: zero_field mode -----
        session.evaluate(wlexpr('NeighborhoodNumber = 8'))
        session.evaluate(wlexpr('DomainWall = 0'))
        
        # Define Psi (hidden)
        session.evaluate(wlexpr('''
            SuperiorInferiorFunctionHidden[v_] := 
                Total[Table[
                    If[Max[m, v - m] > Ceiling[v/2 + DomainWall/2], 
                        Ph * Binomial[v, m] (1/2 + x)^m (1/2 - x)^(v - m), 
                        (1 - Pv) * Binomial[v, m] (1/2 + x)^m (1/2 - x)^(v - m)
                    ], 
                    {m, 0, v, 1}
                ]]
        '''))
        session.evaluate(wlexpr('''
            Psi = Simplify[
                Total[Table[
                    Binomial[NeighborhoodNumber, h] * Qh^h * (1 - Qh)^(NeighborhoodNumber - h) *
                    SuperiorInferiorFunctionHidden[NeighborhoodNumber - h], 
                    {h, 0, NeighborhoodNumber, 1}
                ]]
            ]
        '''))
        
        # Phi (visible, simplified zero-field SI)
        session.evaluate(wlexpr('''
            SuperiorInferiorFunction[v_] := 
                Total[Table[
                    If[Max[m, v - m] > Ceiling[v/2 + DomainWall/2], 
                        If[m > v - m, 
                            (1 - Ph) * Binomial[v, m] (1/2 + x)^m (1/2 - x)^(v - m), 
                            If[m < v - m,
                                -(1 - Ph) * Binomial[v, m] (1/2 + x)^m (1/2 - x)^(v - m),
                                0
                            ]
                        ], 
                        0
                    ], 
                    {m, 0, v, 1}
                ]]
        '''))
        session.evaluate(wlexpr('''
            Phi = Total[
                Table[
                    Binomial[NeighborhoodNumber, h] * Qh^h * (1 - Qh)^(NeighborhoodNumber - h) *
                    SuperiorInferiorFunction[NeighborhoodNumber - h], 
                    {h, 0, NeighborhoodNumber, 1}
                ]
            ]
        '''))
        
        # ----- Three equations -----
        # (1) Self-consistent: Qh = Psi at x = 0, Qh = Qh0
        session.evaluate(wlexpr('''
            SelfConsistent = Simplify[
                (Qh - Psi) /. {Qh -> Qh0} /. {x -> 0}
            ]
        '''))
        
        # (2)(3) Solve for yx and extract C1, C3
        session.evaluate(wlexpr('Eqs = (Qh - Psi) /. {Qh -> Qh0 + y}'))
        session.evaluate(wlexpr('''
            EqsZero = SeriesCoefficient[
                SeriesCoefficient[Eqs, {y, 0, 0}], {x, 0, 0}
            ]
        '''))
        session.evaluate(wlexpr('EqsNew = Eqs - EqsZero'))
        session.evaluate(wlexpr('sol = AsymptoticSolve[EqsNew == 0, {y, 0}, {x, 0, 5}]'))
        session.evaluate(wlexpr('yx = sol[[1]][[1]][[2]]'))
        
        session.evaluate(wlexpr('EqsVisible = (1 - Qh) * (2*x) - Phi /. {Qh -> Qh0 + y}'))
        session.evaluate(wlexpr('EqsFinal = EqsVisible /. {y -> yx}'))
        
        session.evaluate(wlexpr('C1expr = Simplify[SeriesCoefficient[EqsFinal, {x, 0, 1}]]'))
        session.evaluate(wlexpr('C3expr = Simplify[SeriesCoefficient[EqsFinal, {x, 0, 3}]]'))
        
        if verbose:
            print("Setup done. Starting FindRoot...")
        
        # ----- FindRoot (high precision) -----
        ph0, pv0, qh00 = initial_guess
        result = session.evaluate(wlexpr(f'''
            FindRoot[
                {{C1expr == 0, C3expr == 0, SelfConsistent == 0}},
                {{{{Ph, {ph0}}}, {{Pv, {pv0}}}, {{Qh0, {qh00}}}}},
                WorkingPrecision -> {precision},
                MaxIterations -> 200
            ]
        '''))
        
        # Parse the result
        result_str = session.evaluate(wlexpr(f'''
            ToString[
                FindRoot[
                    {{C1expr == 0, C3expr == 0, SelfConsistent == 0}},
                    {{{{Ph, {ph0}}}, {{Pv, {pv0}}}, {{Qh0, {qh00}}}}},
                    WorkingPrecision -> {precision},
                    MaxIterations -> 200
                ],
                InputForm
            ]
        '''))
        
        if verbose:
            print(f"\nTricritical point:")
            print(f"  {result_str}")
        
        # Check: residuals of the three equations
        residuals = session.evaluate(wlexpr(f'''
            {{C1expr, C3expr, SelfConsistent}} /. 
                FindRoot[
                    {{C1expr == 0, C3expr == 0, SelfConsistent == 0}},
                    {{{{Ph, {ph0}}}, {{Pv, {pv0}}}, {{Qh0, {qh00}}}}},
                    WorkingPrecision -> {precision},
                    MaxIterations -> 200
                ]
        '''))
        
        if verbose:
            print(f"  Residuals: {residuals}")
        
        return {
            'result': result,
            'result_str': result_str,
            'residuals': residuals,
            'initial_guess': initial_guess,
            'precision': precision
        }
        
    finally:
        session.terminate()


if __name__ == "__main__":
    out = find_tricritical_point(initial_guess=(0.6, 0.25, 0.5),
                                 precision=50, verbose=True)

    import json
    with open(results_path('tricritical_point.json'), 'w') as fh:
        json.dump({'solution': out['result_str'],
                   'residuals': str(out['residuals']),
                   'initial_guess': list(out['initial_guess']),
                   'working_precision': out['precision']}, fh, indent=2)
    print(f"Saved: {results_path('tricritical_point.json')}")