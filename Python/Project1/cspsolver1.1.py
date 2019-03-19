import  random
import sys

class CSP():
    """The abstract class for a csp."""

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP . If variables is empty, it becomes domains.keys()."""

        variables = variables or list(domains.keys())

        self.variables = variables

        self.domains = domains

        self.neighbors = neighbors

        self.constraints = constraints

        self.initial = ()

        self.curr_domains = None

        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""

        assignment[var] = val

        self.nassigns += 1

        print(" ")

        for l in range(self.nassigns):
            print(" ", end="")
        print("Add variable and value: {", var, ":", val, "} to assignment. ")

        for l in range(self.nassigns):
            print(" ", end="")
        print("Now assignment:", assignment)


    def undoassign(self, var, assignment):
        """Remove {var: val} from assignment."""

        for l in range(self.nassigns + 1):
            print(" ", end="")

        self.nassigns -= 1

        if var in assignment:

            print("Remove {", var, ":", assignment[var], "} from assignment and backtrack to level", self.nassigns)

            del assignment[var]

    def num_conflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently

        def conflict(var2):
            return (var2 in assignment and

                    not self.constraints(var, val, var2, assignment[var2]))

        return sum(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""

        print('CSP:', self, 'assignment:', assignment)

    def init_curr_domains(self):

        """  """

        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def change_curr_domains(self, var, value):

        """Remove curr_domains to removals and change curr_domains of var from assuming var=value."""

        self.init_curr_domains()

        removals = [(var, a) for a in self.curr_domains[var] if a != value]

        self.curr_domains[var] = [value]

        return removals

    def prune(self, var, value, removals):

        """Delete var=value from curr_domains."""

        self.curr_domains[var].remove(value)

        if removals is not None:
            removals.append((var, value))

    def restore(self, removals):

        """Undo and restore curr_domains from removals."""

        for B, b in removals:
            self.curr_domains[B].append(b)

    def goal_test(self, state):

            """The goal is to assign all variables, with all constraints satisfied."""

            if state:

                assignment = dict(state)

                return (len(assignment) == len(self.variables)

                        and all(self.num_conflicts(variables, assignment[variables], assignment) == 0

                                for variables in self.variables))

            else:

                return False

# ______________________________________________________________________________

# Constraint Propagation with AC-3


def AC3(csp, queue=None, removals=None):

    """[Figure 6.3]"""

    num_revised = 0

    csp.display(own="inference-mac-AC3", curr_domains=csp.curr_domains, removals=removals)

    if queue is None:
        queue = [(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]]

    csp.init_curr_domains()

    while queue:

        (Xi, Xj) = queue.pop()

        num_revised = num_revised + revise(csp, Xi, Xj, removals)

        if num_revised:

            if not csp.curr_domains[Xi]:
                print("infernce False!")
                return False

            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.append((Xk, Xi))

    print(" ")
    for l in range(csp.nassigns + 1):
        print(" ", end="")

    if num_revised:

        print("Inference pruned:", removals[-num_revised:])

        for l in range(csp.nassigns + 1):
            print(" ", end="")
        print('After pruned curr_domains:')

        for key in csp.curr_domains:

            for l in range(csp.nassigns + 1):
                print(" ", end="")

            print(key, ":", csp.curr_domains[key])

    else:

        print("No pruned in this inference!")

    return True


def revise(csp, Xi, Xj, removals):

    """Return number of value we remove."""

    num_revised = 0

    for x in csp.curr_domains[Xi][:]:

        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x

        if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):

            csp.prune(Xi, x, removals)

            num_revised += 1

    return num_revised


# ______________________________________________________________________________

# CSP Backtracking Search


# Variable ordering


def first_unassigned_variable(assignment, csp):

    """The default variable order."""

    return [var for var in csp.variables if var not in assignment][0]


def mrv(assignment, csp):

    """Minimum-remaining-values heuristic."""

    mrvminvar = min([v for v in csp.variables if v not in assignment],

                           key=lambda var: num_legal_values(csp, var, assignment))

    csp.display(own="Variable select with mrv", assignment=assignment, curr_domains=csp.curr_domains)

    if csp.curr_domains:

        mrvmin = len(csp.curr_domains[mrvminvar])

    else:

        mrvmin = sum(csp.num_conflicts(mrvminvar, val, assignment) == 0
                                                for val in csp.domains[mrvminvar])

    print(" ")
    for l in range(len(assignment) + 1):
        print(" ", end="")

    print("Select mrv variable:", mrvminvar, "...mrv:", mrvmin)

    return mrvminvar


def num_legal_values(csp, var, assignment):

    if csp.curr_domains:

        return len(csp.curr_domains[var])

    else:

        return sum(csp.num_conflicts(var, val, assignment) == 0

                     for val in csp.domains[var])


# Value ordering


def unordered_domain_values(var, assignment, csp):

    """The default value order."""

    return (csp.curr_domains or csp.domains)[var]


def lcv(var, assignment, csp):

    """Least-constraining-values heuristic."""

    lcvorder= sorted((csp.curr_domains or csp.domains)[var],

                  key=lambda val: csp.num_conflicts(var, val, assignment))

    csp.display(own="select value in order_domain_values-lcv", assignment=assignment,  curr_domains=csp.curr_domains)

    print(" ")

    for l in range(len(assignment) + 1):
        print(" ", end="")

    print("lcv order of", "variable", var, ":", lcvorder)

    return lcvorder


# Inference


def no_inference(csp, var, value, assignment, removals):

    return True


def mac(csp, var, value, assignment, removals):

    """Maintain arc consistency."""

    return AC3(csp, [(X, var) for X in csp.neighbors[var]], removals)


# The search, proper


def backtracking_search(csp,

                        select_unassigned_variable=first_unassigned_variable,

                        order_domain_values=unordered_domain_values,

                        inference=no_inference):

    """[Figure 6.5]"""

    def backtrack(assignment):

        if len(assignment) == len(csp.variables):

            return assignment

        var = select_unassigned_variable(assignment, csp)

        for value in order_domain_values(var, assignment, csp):

            if 0 == csp.num_conflicts(var, value, assignment):

                csp.assign(var, value, assignment)

                removals = csp.change_curr_domains(var, value)

                if inference(csp, var, value, assignment, removals):

                    result = backtrack(assignment)

                    if result is not None:
                        return result

                csp.undoassign(var, assignment)

                csp.restore(removals)

                csp.display(own="backtrack", assignment=assignment, curr_domains=csp.curr_domains)

        return None

    result = backtrack({})

    if csp.goal_test(result):

        csp.display(own="Result, Goal reached! One of solution", assignment=result, curr_domains={})

    else:

        print(" ")
        print("NO such assignment is possible")

    return result


# -----------------------------------------------------------------------------------------

# Tasks schedule problem



class tasksCSP(CSP):
    """The subclass  for a tasks schedule."""

    def num_conflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        def conflict(var2):

            return (var2 in assignment  and
                        not self.constraints(var, val, var2, assignment[var2], assignment=assignment))

        if self.neighbors[var] == []:

            return not self.constraints(var, val, "deadline", "deadline", assignment=assignment)

        else:

            return sum(conflict(v) for v in self.neighbors[var])

    def display(self, own="revise", assignment={}, removals=[], curr_domains={}):
        """Show a human-readable representation of the CSP."""

        print(" ")

        for l in range(self.nassigns+1):
            print("-", end="")

        print('Search level', self.nassigns, '(num of assign:', self.nassigns, ")")

        for l in range(self.nassigns+1):
            print(" ", end="")

        print('Now going to', own, ":")

        if assignment:

            for l in range(self.nassigns + 1):
                print(" ", end="")

            print('assignment:', assignment)

        if assignment and curr_domains:

            for l in range(self.nassigns + 1):
                print(" ", end="")

            print('not assigned variable\'s curr_domains:')

            for key in curr_domains:

                if key not in assignment:

                    for l in range(self.nassigns + 1):
                        print(" ", end="")

                    print(key, ":", curr_domains[key])

        if not assignment and curr_domains:

            for l in range(self.nassigns + 1):
                print(" ", end="")

            print('curr_domains:')

            for key in curr_domains:

                for l in range(self.nassigns + 1):
                    print(" ", end="")

                print(key, ":", curr_domains[key])

        if assignment != {}:

            pfort = {}

            for p in list("pqrxyz"):
                pfort[p] = []

            for task in assignment.keys():
                pfort[assignment[task]].append(task)

            for l in range(self.nassigns + 1):
                print(" ", end="")

            print("processor for task: ", end="")

            for pkey in pfort.keys():
                print(pkey, ":", pfort[pkey],
                      sum(self.variables[t] for t in pfort[pkey] if pfort[pkey] != []),
                      "  ", end="")

            print("")
            for l in range(self.nassigns + 1):
                print(" ", end="")
            print("The total length of all tasks:",
                  sum(sum(self.variables[t] for t in pfort[pkey] if pfort[pkey] != []) for pkey in pfort.keys()))


def input_file(filename):

    processors = []

  #  inputString = input("Please input : search graphName.txt")
  #  inputFile = inputString.split()

    # inputFile = input("Please input task CSP input file Name(csp.txt): ")

    try:

        #f = open(inputFile, 'r')
        f = open(filename, 'r')

        fline = f.readlines()

        f.close()

        return fline

    except FileNotFoundError:

        print("File not found!")


def init_tasks_csgraph(fline):

    for i in range(len(fline)):
        fline[i] = fline[i].strip()

    vari = fline.index("##### - variables")
    valuei = fline.index("##### - values")
    deadlinei = fline.index("##### - deadline constraint")
    unaryini = fline.index("##### - unary inclusive")
    unaryexi = fline.index("##### - unary exclusive")
    bineqi = fline.index("##### - binary equals")
    binnoeqi = fline.index("##### - binary not equals")
    binnotsi = fline.index("##### - binary not simultaneous")

    tasks = {}
    for line in fline[vari+1:valuei]:
        task = line.split()
        tasks[task[0]] = int(task[1])

    processors = fline[valuei+1:deadlinei]

    deadline = int(fline[deadlinei+1])

    unaryincs = {}
    for u in fline[unaryini+1:unaryexi]:
        l = u.split()
        unaryincs[l[0]] = l[1:]

    unaryexcs = {}
    for u in fline[unaryexi+1:bineqi]:
        l = u.split()
        unaryexcs[l[0]] = l[1:]

    binaryeqcs = fline[bineqi+1:binnoeqi]
    binarynoeqcs = fline[binnoeqi+1:binnotsi]

    binarynotsl = fline[binnotsi+1:]
    binarynotscs = {}
    for bns in binarynotsl:
        l = bns.split()
        binarynotscs[(l[0],l[1])] = l[2:]

    neighbors = {}
    for vertex in tasks.keys():
        neighbors[vertex] = []
    for edge in binaryeqcs:
        l = edge.split()
        neighbors[l[0]].append(l[1])
        neighbors[l[1]].append(l[0])
    for edge in binarynoeqcs:
        l = edge.split()
        neighbors[l[0]].append(l[1])
        neighbors[l[1]].append(l[0])
    for edge in binarynotscs.keys():
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])

    domains = {}
    for var in tasks:
        domains[var] = processors

    print("tasks:", tasks)
    print("task variable:", list(tasks.keys()))
    print("processors:", processors)
    print("deadline:", deadline)

    print("unary inclusive:", unaryincs)
    print("unary exclusive:", unaryexcs)
    print("binary equals:", binaryeqcs)
    print("binary not equals:", binarynoeqcs)
    print("binary not simultaneous:", binarynotscs)
    print("neighbors:", neighbors)

    # cs_matrix = [[1, 1, 1, 1, 1, 1],
    #              [1, 1, 1, 1, 1, 1],
    #              [1, 1, 1, 1, 1, 1],
    #              [1, 1, 1, 1, 1, 1],
    #              [1, 1, 1, 1, 1, 1],
    #              [1, 1, 1, 1, 1, 1]]

    cs_dict ={}

    for b in binaryeqcs:
        cs_matrix = []
        for i in range(len(processors)):
            cs_matrix_row = []
            for j in range(len(processors)):
                cs_matrix_row.append(0)
                if i == j:
                    cs_matrix_row[j] = 1
            cs_matrix.append(cs_matrix_row)
        cs_dict[b[0]+b[2]] = cs_matrix

    for b in binarynoeqcs:
        cs_matrix = []
        for i in range(len(processors)):
            cs_matrix_row = []
            for j in range(len(processors)):
                cs_matrix_row.append(1)
                if i == j:
                    cs_matrix_row[j] = 0
            cs_matrix.append(cs_matrix_row)
        cs_dict[b[0]+b[2]] = cs_matrix

    for bnskey in binarynotscs.keys():
        cs_matrix = []
        for row in range(len(processors)):
            cs_matrix_row = []
            pi = processors.index(binarynotscs[bnskey][1])
            for col in range(len(processors)):
                cs_matrix_row.append(1)
                pj = processors.index(binarynotscs[bnskey][0])
                if row == pi and col == pj:
                    cs_matrix_row[col] = 0
            cs_matrix.append(cs_matrix_row)
        cs_dict[bnskey[0]+bnskey[1]] = cs_matrix

    for uinkey in unaryincs.keys():
        dp = processors[:]
        for p in unaryincs[uinkey]:
            dp.remove(p)
        unaryexcs[uinkey] = dp

    for uexkey in unaryexcs.keys():
        if not neighbors[uexkey]:
            for cskey in cs_dict.keys():
                if uexkey == cskey[0]:
                    for p in unaryexcs[uexkey]:
                        pi = processors.index(p)
                        for col in range(len(processors)):
                            cs_dict[cskey][pi][col] = 0
                elif uexkey == cskey[1]:
                    for p in unaryexcs[uexkey]:
                        pi = processors.index(p)
                        for row in range(len(processors)):
                            cs_dict[cskey][row][pi] = 0

    cs_dict["deadline"] = deadline

    for key in cs_dict.keys():
        print(key, ":")
        if key != "deadline":
            for i in range(len(processors)):
                print(cs_dict[key][i])
        else:
            print(cs_dict[key])
        print("")

    def pre_handle(unaryincs, unaryexcs, binaryeqcs):

        for cskey in cs_dict.keys():

            if cskey != "deadline":

                if all(cs_dict[cskey][row][col] == 0 for row in range(len(processors))
                            for col in range(len(processors))):

                    print("Prehandle domains find conflict, so it's no solution!")
                    print("Binary constraint", cskey, ":")
                    print("NO such assignment is possible")

                    for row in range(len(processors)):
                        print(cs_dict[cskey][row])

                    print("system exit!")
                    sys.exit(0)

        # return False

        for uinkey in unaryincs.keys():
            dp = processors[:]
            for p in unaryincs[uinkey]:
                dp.remove(p)
            unaryexcs[uinkey] = dp

        for uexkey in unaryexcs.keys():
            dp = processors[:]
            for p in unaryexcs[uexkey]:
                dp.remove(p)
            domains[uexkey] = dp

        if domains:
            print("After prehandled unary constraints domains:")
            for key in domains:
                print(key, ":", domains[key])

        for be in binaryeqcs:
            if set(domains[be[0]]) <= set(domains[be[2]]):
                domains[be[2]] = domains[be[0]]
            elif set(domains[be[2]]) < set(domains[be[0]]):
                domains[be[0]] = domains[be[2]]
            else:
                print(" ")
                print("Prehandle domains find binary equals constraints conflict, so it's no solution!")
                print("NO such assignment is possible")
                print("system exit!")
                sys.exit(0)

                # return False

        if domains:

            print("  ")
            print("After prehandled binary equal constraints domains:")

            for key in domains:
                print(key, ":", domains[key])

        return domains

    csgraph = {}

    csgraph["domains"] = pre_handle(unaryincs, unaryexcs, binaryeqcs)
    csgraph["vertex"] = tasks
    csgraph["neighbors"] = neighbors
    csgraph["processors"] = processors
    csgraph["cs_dict"] = cs_dict
    return csgraph


fline = input_file("input3.txt")

csgraph = init_tasks_csgraph(fline)


def tasks_constraints(T1, p1, T2, p2, assignment={}, csg=csgraph):
    """Constraint is satisfied return true"""

    tasks = csg["vertex"]
    processors = csg["processors"]
    cs_dict = csg["cs_dict"]

    pfort = {}

    if assignment != {} and T1 not in assignment:

        for p in processors:
            pfort[p] = []

        for task in assignment.keys():
            pfort[assignment[task]].append(task)

        pfort[p1].append(T1)

    def outofdeadline(pfort):

        if pfort != {}:

            for pkey in pfort.keys():

                tasklen= sum(tasks[t] for t in pfort[pkey] if pfort[pkey] != [])

                if tasklen > cs_dict["deadline"]:

                    for l in range(len(assignment) + 1):
                        print(" ", end="")

                    print("out of deadline! sum of", pkey, ":", pfort[pkey],"total len:", tasklen,end="")
                    print(".   So give up (", T1, ":", p1, "), select next value...")

                    return sum(tasks[t] for t in pfort[pkey] if pfort[pkey] != []) > cs_dict["deadline"]

        else:

            return False

    if T1+T2 in cs_dict.keys():

        row = processors.index(p1)
        col = processors.index(p2)

        return (bool(cs_dict[T1+T2][row][col])) and not outofdeadline(pfort)

    elif T2+T1 in cs_dict.keys():

        row = processors.index(p2)
        col = processors.index(p1)

        return (bool(cs_dict[T2+T1][row][col])) and not outofdeadline(pfort)

    else:

        return not outofdeadline(pfort)


taskscsp = tasksCSP(csgraph["vertex"], csgraph["domains"], csgraph["neighbors"], tasks_constraints)

print("  ")
print("Select:")

print("  1. no param backtracking_search.")
print("  2. backtracking_search with mrv lcv mac.")
print("  0. exit.")
print("  ")

inputs = input("Please input your select number(1 or 2):")

if inputs == "1":
    backtracking_search(taskscsp)
elif inputs == "2":
    backtracking_search(taskscsp, mrv, lcv, mac)
else:
    print(" ")
    print("system exit!")
    sys.exit(0)




