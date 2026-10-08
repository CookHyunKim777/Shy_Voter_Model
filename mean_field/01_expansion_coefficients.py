"""
Shy Voter Model - Using Wolfram Engine via wolframclient
Original Mathematica code executed through Python

zero_field=True : stage A (external-field variables ε=0, f=1 are fixed when SI is defined)
zero_field=False: stage B (ε, f kept symbolic, for the external-field analysis)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import KERNEL, results_path


from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import time

def run_shy_voter_analysis(verbose=False, zero_field=True):
    if verbose:
        print("=" * 60)
        print("Shy Voter Model - Wolfram Engine")
        print(f"Mode: {'zero_field (stage A)' if zero_field else 'with external field (stage B)'}")
        print("=" * 60)
        print("\nConnecting to the Wolfram Engine...")
    
    session = WolframLanguageSession(
        kernel=KERNEL
    )
    session.start(timeout=60)
    
    if verbose:
        print("Connected.")
    
    try:
        # ============================================================
        # 0. Parameters
        # ============================================================
        if verbose:
            print("\n" + "=" * 60)
            print("0. Parameters")
            print("=" * 60)
        
        session.evaluate(wlexpr('NeighborhoodNumber = 8'))
        session.evaluate(wlexpr('DomainWall = 0'))   # Z2 symmetry assumed
        
        # Ph, Pv are phase-diagram variables, so they are kept symbolic
        session.evaluate(wlexpr('TempPvNumber = Pv'))
        session.evaluate(wlexpr('TempPhNumber = Ph'))
        
        # External-field variables: depend on the zero_field mode
        if zero_field:
            # Not used, but defined for compatibility
            session.evaluate(wlexpr('TempEpsilon = 0'))
            session.evaluate(wlexpr('TempF = 1'))
            SUBS = '{Pv -> TempPvNumber, Ph -> TempPhNumber}'
        else:
            session.evaluate(wlexpr('TempEpsilon = ε'))
            session.evaluate(wlexpr('TempF = f'))
            SUBS = '{Pv -> TempPvNumber, Ph -> TempPhNumber, ε -> TempEpsilon, f -> TempF}'
        
        if verbose:
            print("NeighborhoodNumber = 8")
            print("DomainWall = 0")
            if zero_field:
                print("ε=0, f=1 (fixed when SI is defined)")
            else:
                print("ε, f symbolic (external field mode)")
        
        # ============================================================
        # 1. Hidden State Mean-Field Equation (Psi)
        # ============================================================
        if verbose:
            print("\n" + "=" * 60)
            print("1. Hidden State Mean-Field Equation (Psi)")
            print("=" * 60)
        
        start_time = time.time()
        
        # Psi does not depend on ε, f, so it is the same in both modes
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
        
        if verbose:
            print(f"Psi computed ({time.time() - start_time:.2f}s)")
        
        # ============================================================
        # 2. Self-Consistent Equation and Asymptotic Solve
        # ============================================================
        if verbose:
            print("\n" + "=" * 60)
            print("2. Asymptotic Solve for y(x)")
            print("=" * 60)
        
        start_time = time.time()
        
        session.evaluate(wlexpr('Eqs = (Qh - Psi) /. {Qh -> Qh0 + y}'))
        session.evaluate(wlexpr(f'''
            EqsZero = SeriesCoefficient[
                SeriesCoefficient[Eqs, {{y, 0, 0}}], {{x, 0, 0}}
            ] /. {SUBS}
        '''))
        session.evaluate(wlexpr(f'''
            EqsNew = Eqs - EqsZero /. {SUBS}
        '''))
        
        session.evaluate(wlexpr('sol = AsymptoticSolve[EqsNew == 0, {y, 0}, {x, 0, 5}]'))
        session.evaluate(wlexpr('yx = sol[[1]][[1]][[2]]'))
        
        # Extract the G coefficients
        G_coeffs = {}
        G_strs = {}
        for i in range(6):
            result = session.evaluate(wlexpr(f'Simplify[SeriesCoefficient[yx, {{x, 0, {i}}}]]'))
            G_coeffs[f'G{i}'] = result
            result_str = session.evaluate(wlexpr(f'ToString[Simplify[SeriesCoefficient[yx, {{x, 0, {i}}}]], InputForm]'))
            G_strs[f'G{i}'] = result_str
            if verbose:
                print(f"\nG{i} = {result_str}")
        
        if verbose:
            print(f"\nDone ({time.time() - start_time:.2f}s)")
        
        # ============================================================
        # 3. Visible State Equation (Phi) — SI depends on zero_field
        # ============================================================
        if verbose:
            print("\n" + "=" * 60)
            print("3. Visible State Equation (Phi)")
            print("=" * 60)
        
        start_time = time.time()
        
        if zero_field:
            # Simplified form with ε=0, f=1 substituted in advance
            # (2f - 1 + 4xε(1-f))|_{f=1, ε=0} = 1
            # (1 - 2f + 4xεf)|_{f=1, ε=0}     = -1
            # free-regime contribution = Pv * 0 = 0
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
        else:
            # For the external-field analysis: ε, f symbolic
            session.evaluate(wlexpr('''
                SuperiorInferiorFunction[v_] := 
                    Total[Table[
                        If[Max[m, v - m] > Ceiling[v/2 + DomainWall/2], 
                            If[m > v - m, 
                                (1 - Ph) * (2*f - 1 + 2*x*2*ε*(1 - f)) * Binomial[v, m] (1/2 + x)^m (1/2 - x)^(v - m), 
                                (1 - Ph) * (1 - 2*f + 2*x*2*ε*f) * Binomial[v, m] (1/2 + x)^m (1/2 - x)^(v - m)
                            ], 
                            Pv * 0 * Binomial[v, m] (1/2 + x)^m (1/2 - x)^(v - m)
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
        
        if verbose:
            print(f"Phi computed ({time.time() - start_time:.2f}s)")
        
        # ============================================================
        # 4. C coefficients
        # ============================================================
        if verbose:
            print("\n" + "=" * 60)
            print("4. C Coefficients")
            print("=" * 60)
        
        start_time = time.time()
        
        session.evaluate(wlexpr('EqsVisible = (1 - Qh) * (2*x) - Phi /. {Qh -> Qh0 + y}'))
        session.evaluate(wlexpr(f'''
            EqsFinal = EqsVisible /. {{y -> yx}} /. {SUBS}
        '''))
        
        # Extract the C coefficients
        C_coeffs = {}
        C_strs = {}
        for i in range(6):
            result = session.evaluate(wlexpr(f'Simplify[SeriesCoefficient[EqsFinal, {{x, 0, {i}}}]]'))
            C_coeffs[f'C{i}'] = result
            result_str = session.evaluate(wlexpr(f'ToString[Simplify[SeriesCoefficient[EqsFinal, {{x, 0, {i}}}]], InputForm]'))
            C_strs[f'C{i}'] = result_str
            if verbose:
                print(f"\nC{i} = {result_str}")
        
        if verbose:
            print(f"\nDone ({time.time() - start_time:.2f}s)")
        
        # ============================================================
        # 5. Prime Equations (unexpanded form)
        # ============================================================
        if verbose:
            print("\n" + "=" * 60)
            print("5. Prime Equations")
            print("=" * 60)
        
        EqsPrime = session.evaluate(wlexpr(f'''
            Simplify[(Qh - Psi) /. {SUBS}]
        '''))
        EqsPrime_str = session.evaluate(wlexpr(f'''
            ToString[Simplify[(Qh - Psi) /. {SUBS}], InputForm]
        '''))
        
        if verbose:
            print(f"EqsPrime computed (length: {len(EqsPrime_str)} chars)")
        
        EqsVisiblePrime = session.evaluate(wlexpr(f'''
            Simplify[(1 - Qh) * 2*x - Phi /. {SUBS}]
        '''))
        EqsVisiblePrime_str = session.evaluate(wlexpr(f'''
            ToString[Simplify[(1 - Qh) * 2*x - Phi /. {SUBS}], InputForm]
        '''))
        
        if verbose:
            print(f"EqsVisiblePrime computed (length: {len(EqsVisiblePrime_str)} chars)")
        
        # ============================================================
        # Return results
        # ============================================================
        return {
            'zero_field': zero_field,
            'G_coeffs': G_coeffs,
            'G_strs': G_strs,
            'C_coeffs': C_coeffs,
            'C_strs': C_strs,
            'EqsPrime': EqsPrime,
            'EqsPrime_str': EqsPrime_str,
            'EqsVisiblePrime': EqsVisiblePrime,
            'EqsVisiblePrime_str': EqsVisiblePrime_str
        }
        
    finally:
        session.terminate()
        if verbose:
            print("\nSession closed")


def print_key_results(results):
    """Print a summary of the key results"""
    print("\n" + "=" * 60)
    print(f"Key results (zero_field={results.get('zero_field', '?')})")
    print("=" * 60)
    
    print("\n[G coefficients - series expansion of Qh]")
    for key, val in results['G_coeffs'].items():
        if val == 0:
            print(f"  {key} = 0")
        else:
            print(f"  {key} ≠ 0")
    
    print("\n[C coefficients - visible-state equation]")
    for key, val in results['C_coeffs'].items():
        if val == 0:
            print(f"  {key} = 0")
        else:
            print(f"  {key} ≠ 0")


def print_coeff(results, name):
    """Print a single coefficient
    
    Usage:
        print_coeff(results, 'G2')
        print_coeff(results, 'C1')
        print_coeff(results, 'EqsPrime')
    """
    if name.startswith('G'):
        print(f"{name} = {results['G_strs'].get(name, 'N/A')}")
    elif name.startswith('C'):
        print(f"{name} = {results['C_strs'].get(name, 'N/A')}")
    elif name == 'EqsPrime':
        print(f"EqsPrime = {results.get('EqsPrime_str', 'N/A')}")
    elif name == 'EqsVisiblePrime':
        print(f"EqsVisiblePrime = {results.get('EqsVisiblePrime_str', 'N/A')}")
    else:
        print(f"Unknown coefficient: {name}")

def save_results(results, filename):
    """Save the results to a JSON file"""
    import json
    
    save_data = {
        'zero_field': results.get('zero_field', True),
        'G_strs': results['G_strs'],
        'C_strs': results['C_strs'],
        'EqsPrime_str': results.get('EqsPrime_str', ''),
        'EqsVisiblePrime_str': results.get('EqsVisiblePrime_str', '')
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Results saved: {filename}")


def load_results(filename):
    """Load saved results"""
    import json
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    results = {
        'zero_field': data.get('zero_field', True),
        'G_strs': data['G_strs'],
        'C_strs': data['C_strs'],
        'G_coeffs': {k: (0 if v == '0' else 1) for k, v in data['G_strs'].items()},
        'C_coeffs': {k: (0 if v == '0' else 1) for k, v in data['C_strs'].items()},
        'EqsPrime_str': data.get('EqsPrime_str', ''),
        'EqsVisiblePrime_str': data.get('EqsVisiblePrime_str', '')
    }
    
    print(f"Results loaded: {filename}")
    return results


if __name__ == "__main__":
    # Stage A: zero-field MFT coefficients (fast)
    results = run_shy_voter_analysis(verbose=True, zero_field=True)
    
    suffix = 'zero_field' if results.get('zero_field', True) else 'with_field'
    save_results(results, results_path(f'shy_voter_results_{suffix}.json'))
    
    print_key_results(results)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)