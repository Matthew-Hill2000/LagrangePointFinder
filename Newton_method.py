import numpy as np
import matplotlib.pyplot as plt
from sympy import *

#  Function definitions
def potential(x, mu):
    return (-1/2)*x**2 - (1-mu)/(np.abs(x+mu)) - mu/(np.abs(x-1+mu))

def potential_deriv(x, mu):
    return -x + (1-mu)/((x+mu)*(np.abs(x+mu))) + mu/((x-1+mu)*(np.abs(x-1+mu)))

def potential_deriv_deriv(x, mu):
    return -1 - 2*(1-mu)/(((x+mu)**2)*(np.abs(x+mu))) - 2*mu/(((x-1+mu)**2)*np.abs(x-1+mu))


def next_x_function(previous_x, mu):
    """
    Returns next value for x according to algorithm
    previous_x (float)
    """
    return previous_x - potential_deriv(previous_x, mu) / potential_deriv_deriv(previous_x, mu)

def newton_raphson(mu, x_start, tolerance=0.00001,
                   next_x=next_x_function):
    """Iterates Newton Raphson algorithm until difference between succesive
    solutions is less than tolerance.
    Args:
        x_start: float, kwarg
        tolerance: float, kwarg
        next_x: function returning float, kwarg
    Returns:
        x_root: float
        counter: int
    """
    # set up parameters
    difference = 1
    counter = 0
    x_root = x_start

    # Repeatedly find x_n until the tolerance threshold is met.
    while difference > tolerance:
        counter += 1
        x_test = x_root
        x_root = next_x(x_root, mu)
        difference = abs(x_test - x_root)
    return x_root, counter


def lagrange_finder(mu, L1_start, L2_start, L3_start):
    L1_solution, L1_count = newton_raphson(mu, L1_start)
    L2_solution, L2_count = newton_raphson(mu, L2_start)
    L3_solution, L3_count = newton_raphson(mu, L3_start)
    L4_solution = [0.5-mu, np.sqrt(3)/2]
    L5_solution = [0.5-mu, -np.sqrt(3)/2]

    L1_solution = [L1_solution, 0]
    L2_solution = [L2_solution, 0]
    L3_solution = [L3_solution, 0]

    Solutions = [L1_solution, L2_solution, L3_solution, L4_solution, L5_solution]

    return Solutions

def hessian(coordinates, U):
    init_printing(use_unicode=True)
    x, y, z = symbols('x y z')
    U_xx = diff(U, x, x)
    U_yy = diff(U, y, y)
    U_xy = diff(U, x, y)
    hessians = [0,0,0,0,0]
    for i in range(len(hessians)):

        hessians[i] = U_xx.evalf(subs={x: coordinates[i][0], y: coordinates[i][1]})*U_yy.evalf(
        subs={x: coordinates[i][0], y: coordinates[i][1]}) - (U_xy.evalf(subs={x: coordinates[i][0], y: coordinates[i][1]}))**2

    return hessians

def lagrange_printer(solutions, U, hessians):
    # Print the results
    init_printing(use_unicode=True)
    x, y, z = symbols('x y z')
    U_xx = diff(U, x, x)

    print(f'L1 = ({solutions[0][0]}, 0)')
    U_xx = U_xx.evalf(subs={x: solutions[0][0], y: 0})
    hessian_analysis(hessians[0], U_xx)
    # print('This took {:d} iterations'.format(L1_count))

    print(f"L2 = ({solutions[1][0]}, 0)")
    U_xx = U_xx.evalf(subs={x: solutions[1][0], y: 0})
    hessian_analysis(hessians[1], U_xx)
    # print('This took {:d} iterations'.format(L2_count))

    print(f'L3 = ({solutions[2][0]}, 0)')
    U_xx = U_xx.evalf(subs={x: solutions[2][0], y: 0})
    hessian_analysis(hessians[2], U_xx)
    # print('This took {:d} iterations'.format(L3_count))

    print(f'L4 = ({solutions[3][0]}, {solutions[3][1]})')
    U_xx = U_xx.evalf(subs={x: solutions[3][0], y: solutions[3][1]})
    hessian_analysis(hessians[3], U_xx)

    print(f'L5 = ({solutions[4][0]}, {solutions[4][1]})')
    U_xx = U_xx.evalf(subs={x: solutions[4][0], y: solutions[4][1]})
    hessian_analysis(hessians[4], U_xx)

def hessian_analysis(H, f_xx):
    if H < 0:
        print(f"H={H} and therfore this point is a saddle")
    elif H > 0 and f_xx > 0:
        print(f"H={H} and U_xx={f_xx} and therefore this point is a local minimum")
    elif H > 0 and f_xx < 0:
        print(f"H={H} and U_xx={f_xx} and therefore this point is a local maximum")
    else:
        print("classification of startionary point unknown")


def lagrange_solver(m2, m1):
    # Initialise mass parameter
    mu = m2/(m1+m2)

    # Starting guesses for numerical solution
    L1_start = 0.9
    L2_start = 1.1
    L3_start = -1

    x, y, z = symbols('x y z')
    U = (-1/2)*(x**2 + y**2) - (1-mu) / \
        (sqrt((x+mu)**2 + y**2)) - mu/(sqrt((x-1+mu)**2 + y**2))

    # Numerical solutions to L1,L2,L3 and analytical solutions to L4,L5
    lagrange_points = lagrange_finder(mu, L1_start, L2_start, L3_start)
    hessians = hessian(lagrange_points, U)
    lagrange_printer(lagrange_points, U, hessians)
    return lagrange_points

def dynamics_solver(m2, m1, coordinate, velocity, final_time, steps):
    # Initialise mass parameter
    mu = m2/(m1+m2)

    p_0 = [0] * steps
    p_1 = [0] * steps
    p_2 = [0] * steps
    p_3 = [0] * steps

    phase_space = [p_0, p_1, p_2, p_3]
    phase_space[0][0] = coordinate[0]
    phase_space[1][0] = coordinate[1]
    phase_space[2][0] = velocity[0]
    phase_space[3][0] = velocity[1]

    x, y, v_x, v_y = symbols('x y v_x, v_y')
    U = (-1/2)*(x**2 + y**2) - (1-mu) / \
        (sqrt((x+mu)**2 + y**2)) - mu/(sqrt((x-1+mu)**2 + y**2))
    U_x = diff(U, x)
    U_y = diff(U, y)

    f_1 = (2*v_x)/2
    f_2 = (2*v_y)/2
    f_3 = 2*v_y - U_x
    f_4 = -2*v_x - U_y


    h = final_time/steps
    for i in range(steps-1):
        print(i)
        k_11=h*f_1.evalf(subs={x: phase_space[0][i], y: phase_space[1][i], v_x: phase_space[2][i], v_y: phase_space[3][i]})
        k_12=h*f_2.evalf(subs={x: phase_space[0][i], y: phase_space[1][i], v_x: phase_space[2][i], v_y: phase_space[3][i]})
        k_13=h*f_3.evalf(subs={x: phase_space[0][i], y: phase_space[1][i], v_x: phase_space[2][i], v_y: phase_space[3][i]})
        k_14=h*f_4.evalf(subs={x: phase_space[0][i], y: phase_space[1][i], v_x: phase_space[2][i], v_y: phase_space[3][i]})
        k_21=h*f_1.evalf(subs={x: phase_space[0][i]+ 1/2 * k_11, y: phase_space[1][i] + 1/2 * k_12, v_x: phase_space[2][i] + 1/2 * k_13, v_y: phase_space[3][i]+ 1/2 * k_14})
        k_22=h*f_2.evalf(subs={x: phase_space[0][i]+ 1/2 * k_11, y: phase_space[1][i] + 1/2 * k_12, v_x: phase_space[2][i] + 1/2 * k_13, v_y: phase_space[3][i]+ 1/2 * k_14})
        k_23=h*f_3.evalf(subs={x: phase_space[0][i]+ 1/2 * k_11, y: phase_space[1][i] + 1/2 * k_12, v_x: phase_space[2][i] + 1/2 * k_13, v_y: phase_space[3][i]+ 1/2 * k_14})
        k_24=h*f_4.evalf(subs={x: phase_space[0][i]+ 1/2 * k_11, y: phase_space[1][i] + 1/2 * k_12, v_x: phase_space[2][i] + 1/2 * k_13, v_y: phase_space[3][i]+ 1/2 * k_14})
        k_31=h*f_1.evalf(subs={x: phase_space[0][i]+ 1/2 * k_21, y: phase_space[1][i] + 1/2 * k_22, v_x: phase_space[2][i] + 1/2 * k_23, v_y: phase_space[3][i]+ 1/2 * k_24})
        k_32=h*f_2.evalf(subs={x: phase_space[0][i]+ 1/2 * k_21, y: phase_space[1][i] + 1/2 * k_22, v_x: phase_space[2][i] + 1/2 * k_23, v_y: phase_space[3][i]+ 1/2 * k_24})
        k_33=h*f_3.evalf(subs={x: phase_space[0][i]+ 1/2 * k_21, y: phase_space[1][i] + 1/2 * k_22, v_x: phase_space[2][i] + 1/2 * k_23, v_y: phase_space[3][i]+ 1/2 * k_24})
        k_34=h*f_4.evalf(subs={x: phase_space[0][i]+ 1/2 * k_21, y: phase_space[1][i] + 1/2 * k_22, v_x: phase_space[2][i] + 1/2 * k_23, v_y: phase_space[3][i]+ 1/2 * k_24})
        k_41=h*f_1.evalf(subs={x: phase_space[0][i]+ 1/2 * k_31, y: phase_space[1][i] + 1/2 * k_32, v_x: phase_space[2][i] + 1/2 * k_33, v_y: phase_space[3][i]+ 1/2 * k_34})
        k_42=h*f_2.evalf(subs={x: phase_space[0][i]+ 1/2 * k_31, y: phase_space[1][i] + 1/2 * k_32, v_x: phase_space[2][i] + 1/2 * k_33, v_y: phase_space[3][i]+ 1/2 * k_34})
        k_43=h*f_3.evalf(subs={x: phase_space[0][i]+ 1/2 * k_31, y: phase_space[1][i] + 1/2 * k_32, v_x: phase_space[2][i] + 1/2 * k_33, v_y: phase_space[3][i]+ 1/2 * k_34})
        k_44=h*f_4.evalf(subs={x: phase_space[0][i]+ 1/2 * k_31, y: phase_space[1][i] + 1/2 * k_32, v_x: phase_space[2][i] + 1/2 * k_33, v_y: phase_space[3][i]+ 1/2 * k_34})

        phase_space[0][i+1] = phase_space[0][i] + h/6*(k_11 + 2*k_21 + 2*k_31 + k_41)
        phase_space[1][i+1] = phase_space[1][i] + h/6*(k_12 + 2*k_22 + 2*k_31 + k_42)
        phase_space[2][i+1] = phase_space[2][i] + h/6*(k_13 + 2*k_23 + 2*k_31 + k_43)
        phase_space[3][i+1] = phase_space[3][i] + h/6*(k_14 + 2*k_24 + 2*k_31 + k_44)

    return phase_space

def main():
    # lagrange_points = lagrange_solver(1.89813e27, 1.98847e30)
    dynamics = dynamics_solver(1.89813e27, 1.98847e30, [0.93130000,0.0], [0,0], 4.0, 40)
    print(dynamics[0][30])

main()




# # Create a contour plot of the potential and plot the Lagrange points on top
# xlist = np.linspace(-1.5, 1.5, 1000)
# ylist = np.linspace(-1.5, 1.5, 1000)
# X, Y = np.meshgrid(xlist, ylist)
# Z = (-1/2)*(X**2 + Y**2) - (1-mu)/(np.sqrt((X+mu)**2 + Y**2)) - \
#     mu/(np.sqrt((X-1+mu)**2 + Y**2))

# fig, ax = plt.subplots(1, 1)
# plt.plot(L1_solution, 0, marker="x", markersize=10)
# plt.annotate("L1", (L1_solution, 0.03))
# plt.plot(L2_solution, 0, marker="x", markersize=10)
# plt.annotate("L2", (L2_solution, 0.03))
# plt.plot(L3_solution, 0, marker="x", markersize=10)
# plt.annotate("L3", (L3_solution, 0.03))
# plt.plot(L4_solution[0], L4_solution[1], marker="x", markersize=10)
# plt.annotate("L4", (L4_solution[0], L4_solution[1]+0.03))
# plt.plot(L5_solution[0], L5_solution[1], marker="x", markersize=10)
# plt.annotate("L5", (L5_solution[0], L5_solution[1]+0.03))

# cp = ax.contour(X, Y, Z, levels=np.linspace(-6, -1, 200),
#                 colors="black", linestyles="-")

# fig.colorbar(cp)  # Add a colorbar to a plot
# ax.set_title('Filled Contours Plot')
# #ax.set_xlabel('x (cm)')
# ax.set_ylabel('y (cm)')
# plt.show()
