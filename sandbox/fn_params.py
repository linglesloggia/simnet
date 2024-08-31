
# fn_params: shows different forms of parammeter passing

def fn(p1, p2, d1=1, d2=2, *pls, **pdc):
    '''Function with all types of parameters.

    The order of the different types of parameters must be respected.
    @param p1: positional parameter, must be given as an argument when calling the function.
    @param p2: second positional parameter.
    @param d1: default paramter; if not given as an argument the indicated value is taken. 
    @param d2: another default parameter.
    @param pls: arbitrary parameters, received in the function as a tuple.
    @param pdc: arbitrary default parameters, receeived in the function as a dictionary.
    '''
    print("  1st positional parameter:", p1)
    print("  2nd positional parameter:", p2)
    print("  a default parameter:", d1)
    print("  another default parameter:", d2)
    print("  arbitrary parameters:", pls)
    print("  arbitrary default parameters:", pdc)
    print()
    return

print("--- Call with all parameters:")
fn("First", "Second", 100, "a", "arb1", "arb2", key1="one", key2="two")

print("--- Order of default parameters may be changed if name is given:")
fn("First", "Second", d2="def2", d1="def1")

print("--- Naming parameters before arbitrary parameters is an error.")
print("This fails:")
print('fn("First", "Second", d2="a", "arb1", "arb2", key1="one", key2="two")')
#fn("First", "Second", d2="a", "arb1", "arb2", key1="one", key2="two")
print()

