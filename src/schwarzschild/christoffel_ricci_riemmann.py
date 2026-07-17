import sympy
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RicciTensor, RiemannCurvatureTensor

# initializing coordinates, we will be writing in cartesian coordinates here, see godelmetric.py for cylindrical coords
t, r, theta, phi = sympy.symbols('t r theta phi')
coords = [t, r, theta, phi]

G,M = sympy.symbols("G M")

# Define the Gödel metric components, written in cartesian like in the original paper
metric_matrix = [
    [1-(2*G*M/r),  0, 0, 0],
    [ 0,  1/(1-(2*G*M/r)),  0,0],
    [0,  0, r**2,  0],
    [ 0,  0,  0,  r**2 * sympy.sin(theta)**2]
]

# creating the metrictensor , g_mu_nu
# 'll' denotes covariant components (both indices lower)
schwarzschild_metric = MetricTensor(metric_matrix, coords, config='ll')

# Compute all christoffel symbols - Gamma^mn_alpha_beta
christoffel = ChristoffelSymbols.from_metric(schwarzschild_metric)

# Now let's print all of them
for coord in coords:
        print("--- Christoffel Symbols Matrix (Gamma^{}_alpha_beta) ---".format(coord))
        # Index convention is [upper, lower, lower] -> christoffel.tensor()[0] gives Gamma^t_mu_nu
        sympy.pprint(sympy.simplify(christoffel.tensor()[0]))

# Compute Ricci Tensor (R_mu_nu)
ricci = RicciTensor.from_metric(schwarzschild_metric)
print("\n--- Ricci Tensor Matrix ---")
sympy.pprint(sympy.simplify(ricci.tensor()))

# Computing scalar curvature
# Get the inverse metric matrix (g^mu_nu) with upper indices
inv_metric = schwarzschild_metric.inv().tensor()

# Get the standard Ricci tensor matrix (R_mu_nu) with lower indices
ricci_matrix = ricci.tensor()

# Compute the trace: sum up the diagonal elements of (inv_metric * ricci_matrix)
ricci_scalar = 0
for mu in range(4):
    for nu in range(4):
        ricci_scalar += inv_metric[mu, nu] * ricci_matrix[nu, mu]

print("\n--- Scalar Curvature  -> g^mu^nu R_mu_nu n ---")
print(sympy.simplify(ricci_scalar))
