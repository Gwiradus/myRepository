import pulp as p


def solve_mip(n_ev, charger, price, min_e, max_e):
    # Create a LP Minimization problem
    lp_prob = p.LpProblem('Problem', p.LpMinimize)

    # Create problem Variables
    u0 = p.LpVariable("u0", lowBound=0)  # Create a variable u0 >= 0
    u1 = p.LpVariable("u1", lowBound=0)  # Create a variable u1 >= 0
    u2 = p.LpVariable("u2", lowBound=0)  # Create a variable u2 >= 0
    u3 = p.LpVariable("u3", lowBound=0)  # Create a variable u3 >= 0
    u4 = p.LpVariable("u4", lowBound=0)  # Create a variable u4 >= 0
    u5 = p.LpVariable("u5", lowBound=0)  # Create a variable u5 >= 0
    u6 = p.LpVariable("u6", lowBound=0)  # Create a variable u6 >= 0
    u7 = p.LpVariable("u7", lowBound=0)  # Create a variable u7 >= 0
    u8 = p.LpVariable("u8", lowBound=0)  # Create a variable u8 >= 0
    u9 = p.LpVariable("u9", lowBound=0)  # Create a variable u9 >= 0
    u10 = p.LpVariable("u10", lowBound=0)  # Create a variable u10 >= 0
    u11 = p.LpVariable("u11", lowBound=0)  # Create a variable u11 >= 0

    # Objective Function
    lp_prob += n_ev * charger * (price[0] * u0 + price[1] * u1 + price[2] * u2 + price[3] * u3
                                 + price[4] * u4 + price[5] * u5 + price[6] * u6 + price[7] * u7
                                 + price[8] * u8 + price[9] * u9 + price[10] * u10 + price[11] * u11)

    # Constraints:
    lp_prob += n_ev * charger * u0 >= min_e[0]
    lp_prob += n_ev * charger * (u0 + u1) >= min_e[1]
    lp_prob += n_ev * charger * (u0 + u1 + u2) >= min_e[2]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3) >= min_e[3]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4) >= min_e[4]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5) >= min_e[5]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6) >= min_e[6]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7) >= min_e[7]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8) >= min_e[8]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8 + u9) >= min_e[9]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8 + u9 + u10) >= min_e[10]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8 + u9 + u10 + u11) >= min_e[11]

    lp_prob += n_ev * charger * u0 <= max_e[0]
    lp_prob += n_ev * charger * (u0 + u1) <= max_e[1]
    lp_prob += n_ev * charger * (u0 + u1 + u2) <= max_e[2]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3) <= max_e[3]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4) <= max_e[4]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5) <= max_e[5]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6) <= max_e[6]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7) <= max_e[7]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8) <= max_e[8]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8 + u9) <= max_e[9]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8 + u9 + u10) <= max_e[10]
    lp_prob += n_ev * charger * (u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8 + u9 + u10 + u11) <= max_e[11]

    lp_prob += u0 <= 1
    lp_prob += u1 <= 1
    lp_prob += u2 <= 1
    lp_prob += u3 <= 1
    lp_prob += u4 <= 1
    lp_prob += u5 <= 1
    lp_prob += u6 <= 1
    lp_prob += u7 <= 1
    lp_prob += u8 <= 1
    lp_prob += u9 <= 1
    lp_prob += u10 <= 1
    lp_prob += u11 <= 1

    # # Display the problem
    # print(lp_prob)

    lp_prob.solve()  # Solver
    # print(p.LpStatus[status])  # The solution status

    # # Printing the final solution
    # print(p.value(u0), p.value(u1), p.value(u2), p.value(u3), p.value(u4), p.value(u5),
    #       p.value(u6), p.value(u7), p.value(u8), p.value(u9), p.value(u10), p.value(u11), p.value(lp_prob.objective))
    #
    # print(p.value(lp_prob.objective))

    mip_opt = [p.value(u0), p.value(u1), p.value(u2), p.value(u3), p.value(u4), p.value(u5),
               p.value(u6), p.value(u7), p.value(u8), p.value(u9), p.value(u10), p.value(u11)]
    mip_opt[:] = [x * n_ev * charger for x in mip_opt]
    value = p.value(lp_prob.objective)
    return mip_opt, value
