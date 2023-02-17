import matplotlib
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pulp import *
import csv
sys.path.append("C:\\Users\\boria\\PycharmProjects\\pythonProject3\\src")
from Plot.plot import plot_statistics


# Method for the model without solution
def model_mip(plate, n_circuits, x, y):
    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("vlsi", LpMinimize)
    # Height min
    height_min = min(y)
    # If there's a block higher than the sum of the heights divided by two
    if int(sum(y)/2) <= max(y):
        height_max = sum(y)
    else:
        height_max = int(sum(y)/2)
    # Height
    height = LpVariable("height", lowBound=height_min, upBound=height_max, cat=LpInteger)

    # Area
    area = height * plate
    area_min = sum(x[i]*y[i] for i in range(0, n_circuits))
    area_max = height_max * plate

    # Coordinate variables
    coord_x = [
        LpVariable(
            f"coord_x_{i}", lowBound=0, upBound=int(plate - min(x)), cat=LpInteger)
        for i in range(0, n_circuits)
    ]

    coord_y = [
        LpVariable(
            f"coord_y_{i}", lowBound=0, upBound=int(height_max - min(y)), cat=LpInteger)
        for i in range(0, n_circuits)
    ]

    big_m = LpVariable.dicts(
        "big_M",
        indices=(range(n_circuits), range(n_circuits), range(4)),
        cat=LpBinary,
        lowBound=0,
        upBound=1,
    )

    # CONSTRAINTS
    prob += height

    # Constraint for the width and height of the plate
    for i in range(0, n_circuits):
        prob += coord_x[i] + x[i] <= plate
    for i in range(0, n_circuits):
        prob += coord_y[i] + y[i] <= height

    # Constraint for overlapping
    for i in range(0, n_circuits):
        for j in range(0, n_circuits):
            if i < j:
                prob += coord_x[i] + x[i] <= coord_x[j] + (big_m[i][j][0]) * plate
                prob += coord_y[i] + y[i] <= coord_y[j] + (big_m[i][j][1]) * height_max
                prob += coord_x[j] + x[j] <= coord_x[i] + (big_m[i][j][2]) * plate
                prob += coord_y[j] + y[j] <= coord_y[i] + (big_m[i][j][3]) * height_max
                prob += (big_m[i][j][0] + big_m[i][j][1] + big_m[i][j][2] + big_m[i][j][3] <= 3)

    # Constraint to have less space unused
    prob += area >= area_min
    prob += area <= area_max

    # Solver (CPLEX, GUROBI, PULP_CMC_CMD)
    prob.solve(CPLEX(timeLimit=300))

    return prob


# Method for rotation model
def model_mip_rotation(plate, n_circuits, x, y):
    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("vlsi_rot", LpMinimize)
    # Height min
    height_min = min(y)
    # If there's a block higher than the sum of the heights divided by two
    if int(sum(y) / 2) <= max(y):
        height_max = sum(y)
    else:
        height_max = int(sum(y) / 2)
    # Height
    height = LpVariable("height", lowBound=height_min, upBound=height_max, cat=LpInteger)

    # Area
    area = height * plate
    area_min = sum(x[i] * y[i] for i in range(0, n_circuits))
    area_max = height_max * plate

    # Coordinate variables
    coord_x = [
        LpVariable(
            f"coord_x_{i}", lowBound=0, upBound=int(plate - min(min(y), min(x))), cat=LpInteger)
        for i in range(0, n_circuits)
    ]

    coord_y = [
        LpVariable(
            f"coord_y_{i}", lowBound=0, upBound=int(height_max - min(min(y), min(x))), cat=LpInteger)
        for i in range(0, n_circuits)
    ]

    rotation = LpVariable.dicts(
            f"rotation",
            indices=range(n_circuits), cat=LpBinary, lowBound=0, upBound=1,
    )

    big_m = LpVariable.dicts(
        "big_M",
        indices=(range(n_circuits), range(n_circuits), range(4)),
        cat=LpBinary,
        lowBound=0,
        upBound=1,
    )

    # CONSTRAINTS
    prob += height

    # Constraint for the rotation
    for i in range(0, n_circuits):
        if x[i] == y[i] or y[i] >= plate+1:
            rotation[i] = 0

    # Constraint for the width and height of the plate
    for i in range(0, n_circuits):
        prob += coord_x[i] + y[i] * rotation[i] + x[i] * (1-rotation[i]) <= plate
    for i in range(0, n_circuits):
        prob += coord_y[i] + x[i] * rotation[i] + y[i] * (1-rotation[i]) <= height

    # Constraint for overlapping and rotation
    for i in range(0, n_circuits):
        for j in range(0, n_circuits):
            if i < j:
                prob += coord_x[i] + x[i] * (1-rotation[i]) + y[i] * rotation[i] <= coord_x[j] + (big_m[i][j][0]) * plate
                prob += coord_y[i] + y[i] * (1-rotation[i]) + x[i] * rotation[i] <= coord_y[j] + (big_m[i][j][1]) * height_max
                prob += coord_x[j] + x[j] * (1-rotation[j]) + y[j] * rotation[j] <= coord_x[i] + (big_m[i][j][2]) * plate
                prob += coord_y[j] + y[j] * (1-rotation[j]) + x[j] * rotation[j] <= coord_y[i] + (big_m[i][j][3]) * height_max
                prob += (big_m[i][j][0] + big_m[i][j][1] + big_m[i][j][2] + big_m[i][j][3] <= 3)

    # Constraint to have less space unused
    prob += area >= area_min
    prob += area <= area_max

    # Solver (CPLEX, GUROBI, PULP_CMC_CMD)
    prob.solve(CPLEX(timeLimit=300))

    return prob


# Method to plot
def plot(height, coord_x, coord_y, circuits, n_ins, width, directory, PATH):
        file_name = f'MIP Solution{n_ins}'

        # Creating the image
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        plt.xlim([0, width])
        plt.ylim([0, height])

        # Randomizing the color for each rectangle
        for i in range(0, len(circuits)):
            r = random.random()
            g = random.random()
            # To avoid white color
            if r >= 0.9 and g >= 0.9:
                b = 0
            # To avoid black
            elif r <= 0.1 and g <= 0.1:
                b = 1
            else:
                b = random.random()

            color = (r, g, b)

            # Start coordinate
            start_point = (coord_x[i], coord_y[i])
            # add rectangle to plot
            ax.add_patch(Rectangle(start_point, circuits[i][0], circuits[i][1], facecolor=color, fill=True))

        if height <= 27:
            radius = 0.008 * height
        else:
            radius = 0.003 * height

        # Adding point of coordinates
        for i in range(0, len(circuits)):
            ax.add_patch(matplotlib.patches.Circle((coord_x[i], coord_y[i]), radius=radius, facecolor="black"))

        # Saving the image
        plt.savefig(f"{PATH}\\src\\MIP\\solution\\{directory}\\images\\{file_name}.jpg")
        plt.close(fig)


# Method to write the solution on txt file
def write_solution(coord_x, coord_y, circuits, directory, file_name, plate, height, n_circuits, PATH, rotation):
    f = open(f"{PATH}\\src\\MIP\\solution\\{directory}\\txt\\{file_name}.txt", "w")

    f.write(f"{plate} {height}\n{n_circuits}")

    if directory == "rotation":
        print(f"\nSolution with rotation: \n")
    else:
        print(f"\nSolution without rotation: \n")

    for i in range(0, len(coord_x)):
        if i == 0:
            if rotation[i] == 1:
                print(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]} R")
                f.write(f"\n{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]} R\n")
            else:
                print(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]}")
                f.write(f"\n{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]}\n")

        elif i == len(coord_x) - 1:
            if rotation[i] == 1:
                print(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]} R")
                f.write(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]} R\n\n")
            else:
                print(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]}")
                f.write(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]}\n\n")
        else:
            if rotation[i] == 1:
                print(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]} R")
                f.write(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]} R\n")
            else:
                print(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]}")
                f.write(f"{coord_x[i]}, {coord_y[i]}, {circuits[i][0]}, {circuits[i][1]}\n")

    f.close()


# Method that convert the solution into lists or int
def check_solution(prob, model, n_circuits):
    rotation = [0 for i in range(n_circuits)]
    coord_x = []
    coord_y = []
    height = 0

    for v in prob.variables():
        if "coord_x" in v.name:
            coord_x.insert(int(v.name[8:]), round(float(str(v.varValue))))
        elif "coord_y" in v.name:
            coord_y.insert(int(v.name[8:]), round(float(str(v.varValue))))
        elif "height" in v.name:
            height = int(v.varValue)
        elif "rotation" in v.name:
            rotation[int(v.name[9:])] = int(str(v.varValue).rsplit(".", 1)[0])

    return coord_x, coord_y, height, rotation


def check_circuits(rotation, circuits):
    for i in range(0, len(rotation)):
        if rotation[i] == 1:
            circuits[i][0], circuits[i][1] = circuits[i][1], circuits[i][0]
    return circuits


# Method that takes the instances from txts in lists
def creating_var(content):
    plate = int(content[0])
    n_circuits = int(content[1])
    circuit = []

    for i, j in enumerate(content[2:int(content[1]) + 2]):
        j = j.strip().split(" ")
        circuit.append([int(x) for x in j])

    x = [circuit[i][0] for i in range(0, n_circuits)]
    y = [circuit[i][1] for i in range(0, n_circuits)]

    return plate, n_circuits, x, y, circuit


# Method to check the solution status
def check_status(status):
    if status == -3:
        str = "Error"
    elif status == -2:
        str = "Unbounded"
    elif status == -1:
        str = "Infeasible"
    elif status == 0:
        str = "No solution found"
    elif status == 1:
        str = "Optimal"
    else:
        str = "Feasible"
    return str


def compiler():
    # Insert the path of your project
    PATH = "INSERT//YOUR//PATH//HERE"
    # Instances
    from_instance = 1
    to_instance = 41

    # Titles of CSV files
    data = ["N. Instance", "Solution Status", "Height", "Time"]
    with open(f'{PATH}\\src\\MIP\\solution\\model\\statistics\\statistics_CPLEX.csv', 'w') as csv_, \
         open(f'{PATH}\\src\\MIP\\solution\\rotation\\statistics\\statistics_CPLEX.csv', 'w') as csv_rot:

        # Writing the titles on CSVs files
        writer = csv.writer(csv_)
        writer.writerow(data)
        csv_.flush()

        # Writing the titles on CSVs files
        writer_rot = csv.writer(csv_rot)
        writer_rot.writerow(data)
        csv_rot.flush()

        # Reading all the instances
        for n_file in range(from_instance, to_instance):
            print(f"\nINSTANCE {n_file}")
            with open(f"{PATH}\\src\\Instances\\ins-{n_file}.txt") as f:
                content = f.readlines()
                plate, n_circuits, x, y, circuits = creating_var(content)
                # Starting the model without rotation
                print("------SEARCHING SOLUTION WITHOUT ROTATION------")
                prob = model_mip(plate, n_circuits, x, y)

                if prob.status == 2 or prob.status == 1:
                    # Inserting the solution on the correct variables
                    coord_x, coord_y, height, rotation = check_solution(prob, "model", n_circuits)
                    # Writing on the CSV all the statistics
                    writer.writerow([n_file, check_status(prob.status), height, round(prob.solutionTime, 4)])
                    csv_.flush()
                    # Plotting the solution found
                    plot(height, coord_x, coord_y, circuits, n_file, plate, "model", PATH)
                    # Writing on txt the solution
                    write_solution(coord_x, coord_y, circuits, "model", f"MIP Solution {n_file}", plate, height,
                                   n_circuits, PATH, rotation)
                else:
                    # Writing on the CSV all the statistics
                    writer.writerow([n_file, check_status(prob.status), " ", round(prob.solutionTime, 4)])
                    csv_.flush()

                # Starting the model with rotation
                print("\n------SEARCHING SOLUTION WITH ROTATION------")
                prob_rot = model_mip_rotation(plate, n_circuits, x, y)

                if prob_rot.status == 2 or prob_rot.status == 1:
                    # Inserting the solution on the correct variables
                    coord_x, coord_y, height, rotation = check_solution(prob_rot, "rotation", n_circuits)
                    # Writing on the CSV all the statistics
                    writer_rot.writerow([n_file, check_status(prob_rot.status), height, round(prob_rot.solutionTime, 4)])
                    csv_rot.flush()
                    # Rotating the circuits that the model decided to
                    circuits = check_circuits(rotation, circuits)
                    # Plotting the solution found
                    plot(height, coord_x, coord_y, circuits, n_file, plate, "rotation",  PATH)
                    # Writing on txt the solution
                    write_solution(coord_x, coord_y, circuits, "rotation", f"MIP Solution {n_file}", plate, height,
                                   n_circuits, PATH, rotation)
                else:
                    # Writing on the CSV all the statistics
                    writer_rot.writerow([n_file, check_status(prob_rot.status), " ", round(prob_rot.solutionTime, 4)])
                    csv_rot.flush()
            # Closing the file of instance
            f.close()
    # Closing CSVs files
    csv_.close()
    csv_rot.close()


if __name__ == "__main__":
    compiler()

    # Plot the comparisons between max 3 files statistics
    PATH = "INSERT//YOUR//PATH//HERE"
    # Type of model (model or rotation)
    model = "model"
    # Paths of csvs
    csvs_files = [f"{PATH}\\src\\MIP\\solution\\{model}\\statistics\\statistics_CPLEX.csv",
                  f"{PATH}\\src\\MIP\\solution\\{model}\\statistics\\statistics_PULP.csv",
                  f"{PATH}\\src\\MIP\\solution\\{model}\\statistics\\statistics_GUROBI.csv"]
    # Name of the image
    name_jpg = f"CPLEX_PULP_GUROBI"
    # Title of the image
    title_image = "Comparison between CPLEX, PULP and GUROBI"
    # Comparison in time or height
    type_stat = "time"
    # Legend of the graph
    legend = ["CPLEX", "PULP", "GUROBI"]
    
    # This method plots statistics of csvs. You can choose if you prefer to plot time or heights. It takes 6 parameters.
    # Plot_statistics(list of files that you want to plot (tested with max.3), string with name of jpg, string with the
    # title of the plot, list of what you want on the legend, string that is or height or time and string with the type
    # of model or model or rotation.
    plot_statistics(csvs_files, title_image, legend, type_stat, f"{PATH}\\src\\MIP\\solution\\{model}\\statistics\\plot\\"
                                                               f"{name_jpg}_{type_stat}.jpg")
