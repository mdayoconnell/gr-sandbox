import sympy
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RicciTensor, RiemannCurvatureTensor

# initializing coordinates, we will be writing in cartesian coordinates here, see godelmetric.py for cylindrical coords
t, x, y, z = sympy.symbols('t x y z')
cart_coords = [t, x, y, z]

# Define the Gödel metric components, written in cartesian like in the original paper
metric_matrix_cart = [
    [-1,  0, -sympy.exp(x),                 0],
    [ 0,  1,  0,                            0],
    [-sympy.exp(x),  0, -0.5 * sympy.exp(2*x),  0],
    [ 0,  0,  0,                            1]
]

# creating the metrictensor , g_munu
# 'll' denotes covariant components (both indices lower)
godel_metric_cart = MetricTensor(metric_matrix_cart, cart_coords, config='ll')

# Compute all christoffel symbold - Gamma^mn_alpha_beta
christoffel = ChristoffelSymbols.from_metric(godel_metric_cart)

# Now let's print all of them
for coord in cart_coords:
        print("--- Christoffel Symbols Matrix (Gamma^{}_alpha_beta) ---".format(coord))
        # Index convention is [upper, lower, lower] -> christoffel.tensor()[0] gives Gamma^t
        sympy.pprint(sympy.simplify(christoffel.tensor()[0]))

# Compute Ricci Tensor (R_mu_nu)
ricci = RicciTensor.from_metric(godel_metric_cart)
print("\n--- Ricci Tensor Matrix ---")
sympy.pprint(sympy.simplify(ricci.tensor()))

# Compute Riemann Curvature Tensor (R^rho_sigma_mu_nu)
riemann = RiemannCurvatureTensor.from_metric(godel_metric_cart)
print("\n--- Riemann Tensor Component R^t_x_t_x ---")
# Index convention: [upper, lower, lower, lower]
sympy.pprint(sympy.simplify(riemann.tensor()[0, 1, 0, 1]))


# Computing scalar curvature
# Get the inverse metric matrix (g^mu_nu) with upper indices
inv_metric_cart = godel_metric_cart.inv().tensor()

# Get the standard Ricci tensor matrix (R_mu_nu) with lower indices
ricci_matrix_cart = ricci.tensor()

# Compute the trace: sum up the diagonal elements of (inv_metric * ricci_matrix)
ricci_scalar_cart = 0
for mu in range(4):
    for nu in range(4):
        ricci_scalar_cart += inv_metric_cart[mu, nu] * ricci_matrix_cart[nu, mu]

print("\n--- Scalar Curvature  -> g^mu^nu R_mu_nu n ---")
print(sympy.simplify(ricci_scalar_cart))
