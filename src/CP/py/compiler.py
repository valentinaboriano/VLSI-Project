# Importing libraries
from minizinc import Instance, Model, Solver
from datetime import timedelta
import matplotlib
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import sys
sys.path.append("C:\\Users\\boria\\PycharmProjects\\pythonProject3\\src")
from Plot.plot import plot_statistics


# Method that runs the minizinc model
def model_mzn(content, file, solvers, PATH, free_search):
    # Load model from file
    model = Model(f"{PATH}\\src\\CP\\mzn\\{file}.mzn")
    # Find the MiniZinc solver configuration for Gecode
    solver = Solver.lookup(solvers)
    # Create an Instance of the model for Gecode
    instance = Instance(solver, model)
    # Assign instances
    instance["plate"] = int(content[0])
    instance["n_circuits"] = int(content[1])
    circuit = []
    for i, j in enumerate(content[2:int(content[1]) + 2]):
        j = j.strip().split(" ")
        circuit.append([int(x) for x in j])

    instance["circuits"] = circuit
    # Solving the instance
    result = instance.solve(timeout=timedelta(seconds=300), free_search=free_search)
    # Output
    return result, circuit, int(content[0]), int(content[1])


# Method to plot the solution
def plot(height, coord_x, coord_y, circuits, n_ins, width, directory, PATH):
    file_name = f'CP Solution{n_ins}'

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
    plt.savefig(f"{PATH}\\src\\CP\\solution\\{directory}\\images\\{file_name}.jpg")
    plt.close(fig)


# Method to write the solution on txt file
def write_solution(coord_x, coord_y, circuits, directory, file_name, plate, height, n_circuits, PATH, rotation):
    f = open(f"{PATH}\\src\\CP\\solution\\{directory}\\txt\\{file_name}.txt", "w")

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
def check_solution(result, directory):
    height = result.objective
    coord_x = result.solution.coord_x
    coord_y = result.solution.coord_y
    if directory == "rotation":
        rotation = result.solution.rotation
        return height, coord_x, coord_y, rotation
    return height, coord_x, coord_y, [0 for i in range(len(coord_x))]


# Method that convert the solution on the statistics that we need on the CSV
def statistics_csv(result, ins, status, height):
    statistic = [ins, str(status), height]
    if "initTime" in result.statistics:
        initTime = result.statistics["initTime"].total_seconds()
    elif "flatTime" in result.statistics:
        initTime = result.statistics["flatTime"].total_seconds()
    else:
        initTime = 0

    if "solveTime" in result.statistics:
        statistic.append(round(result.statistics["solveTime"].total_seconds() + initTime, 4))
    else:
        statistic.append(initTime)

    if "failures" in result.statistics:
        statistic.append(result.statistics["failures"])
    else:
        statistic.append("")

    return statistic


# Method to rotate the circuits that the model decided to
def check_circuits(rotation, circuits):
    for i in range(0, len(rotation)):
        if rotation[i]:
            circuits[i][0], circuits[i][1] = circuits[i][1], circuits[i][0]
    return circuits


def compiler():
    # Insert the path of your project
    PATH = "INSERT//YOUR//PATH//HERE"
    # Configuration for the solver
    solver = "chuffed"  # or "gecode"
    # Free Search
    free_search = True
    # Instances
    from_instance = 1
    to_instance = 41  # The last instance you want +1

    csv_name = f'statistics_{solver}_fs_{free_search}_best_ss.csv'

    # Titles of CSV files
    data = ["N. Instance", "Solution Status", "Height", "Time", "N. Failures"]
    with open(f'{PATH}\\src\\CP\\solution\\model\\statistics\\{csv_name}', 'w') as csv_, \
         open(f'{PATH}\\src\\CP\\solution\\rotation\statistics\\{csv_name}', 'w') as csv_rot:

        writer = csv.writer(csv_)
        writer_rot = csv.writer(csv_rot)

        # Writing the titles on CSVs files
        writer.writerow(data)
        writer_rot.writerow(data)

        # Reading all the instances
        for n_file in range(from_instance, to_instance):
            print(f"\nINSTANCE {n_file}")
            # Opening all the instances
            with open(f"{PATH}\\src\\Instances\\ins-{n_file}.txt") as f:
                content = f.readlines()
                # IF YOU WANT ONLY THE MODEL WITH ROTATION DELETE OR COMMENT FROM HERE
                # Starting the model without rotation
                print("------SEARCHING SOLUTION WITHOUT ROTATION------")
                result, circuits, width, n_circuits = model_mzn(content, "final_model", solver, PATH, free_search)

                if str(result.status) == "SATISFIED" or str(result.status) == "OPTIMAL_SOLUTION":
                    # Inserting the solution on the correct variables
                    height, coord_x, coord_y, rotation = check_solution(result, "model")
                    # Writing on the CSV all the statistics
                    stat = statistics_csv(result, n_file, result.status, height)
                    writer.writerow(stat)
                    # Plotting the solution found
                    plot(height, coord_x, coord_y, circuits, n_file, width, "model", PATH)
                    # Writing on txt the solution
                    write_solution(coord_x, coord_y, circuits, "model", f"CP Solution {n_file}", width, height,
                                   n_circuits, PATH, rotation)
                else:
                    # Writing on the CSV all the statistics
                    writer.writerow([n_file, result.status, "", "", ""])
                # TO HERE

                # IF YOU WANT ONLY THE MODEL WITHOUT ROTATION DELETE OR COMMENT FROM HERE
                # Starting the model with rotation
                print("\n------SEARCHING SOLUTION WITH ROTATION------")
                result, circuits, width, n_circuits = model_mzn(content, "rotation_model", solver, PATH, free_search)

                if str(result.status) == "SATISFIED" or str(result.status) == "OPTIMAL_SOLUTION":
                    # Inserting the solution on the correct variables
                    height, coord_x, coord_y, rotation = check_solution(result, "rotation")
                    # Writing on the CSV all the statistics
                    stat_rot = statistics_csv(result, n_file, result.status, height)
                    writer_rot.writerow(stat_rot)
                    # Rotating the circuits that the model decided to
                    circuits = check_circuits(rotation, circuits)
                    # Plotting the solution found
                    plot(height, coord_x, coord_y, circuits, n_file, width, "rotation", PATH)
                    # Writing on txt the solution
                    write_solution(coord_x, coord_y, circuits, "rotation", f"CP Solution {n_file}", width, height,
                                   n_circuits, PATH, rotation)
                else:
                    # Writing on the CSV all the statistics
                    writer_rot.writerow([n_file, result.status, "", "", ""])
                # TO HERE

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
    # Type of solver (chuffed or gecode)
    solver = "chuffed"
    type_stat = "time"
    # Paths of csvs
    csvs_files = [f"{PATH}\\src\\CP\\solution\\{model}\\statistics\\statistics_{solver}_fs_True_linear_500.csv",
                  f"{PATH}\\src\\CP\\solution\\{model}\\statistics\\statistics_{solver}_fs_True_constant_200.csv",
                  f"{PATH}\\src\\CP\\solution\\{model}\\statistics\\statistics_{solver}_fs_True_luby_200.csv"]
    # Name of the image
    name_jpg = f"{solver}_fs_True_linear_500-constant_200-luby_200"
    # Comparison in time or height
    title_image = f"Comparison between search strategies with {solver}"
    # Legend of the graph
    legend = [f"linear 500", "constant 200", "luby 200"]

    # This method plots statistics of csvs. You can choose if you prefer to plot time or heights. It takes 6 parameters.
    # Plot_statistics(list of files that you want to plot (tested with max.3), string with name of jpg, string with the
    # title of the plot, list of what you want on the legend, string that is or height or time and string with the type
    # of model or model or rotation.
    plot_statistics(csvs_files, title_image, legend, type_stat, f"{PATH}\\src\\CP\\solution\\{model}\\statistics\\plot\\"
                                                                f"{name_jpg}_{type_stat}.jpg")