import csv
import random
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from z3 import *

# Global variables
path = "INSERT//YOUR//PATH//HERE"
input_dir = os.path.join(path, "src\\instances\\")
output_dir = os.path.join(path, "src\\SMT\\solution\\")


def import_file(instance):
    with open(os.path.join(input_dir, "ins-{}.txt".format(instance)), 'r') as f:
        rows = f.read().splitlines()

        plate = int(rows[0])
        n_circuits = int(rows[1])

        x, y = [], []
        for i in range(2, int(n_circuits) + 2):
            divide = rows[i].split(' ')
            x.append(int(divide[0]))
            y.append(int(divide[1]))

        f.close()
        return plate, n_circuits, x, y


def output_file(instance, plate, n_circuits, x, y, coord_x, coord_y, rotation_sol, height_sol, total_time, rotation):
    if rotation:
        file = os.path.join(output_dir, "rotation\\txt\\SMT Solution{}.txt".format(instance))
    else:
        file = os.path.join(output_dir, "model\\txt\\SMT Solution{}.txt".format(instance))
    with open(file, 'w+') as f_out:
        f_out.write('{} {}\n'.format(plate, height_sol))
        f_out.write('{}\n'.format(n_circuits))

        for i in range(n_circuits):
            rotated_check = ""
            if rotation_sol[i]:
                rotated_check = "R"
            f_out.write('{} {} {} {} {}\n'.format(x[i], y[i], coord_x[i], coord_y[i], rotated_check))
        f_out.write(f'{total_time :.2f}')
        f_out.close()


def plot(instance, plate, n_circuits, x, y, coord_x, coord_y, rotation_sol, height, rotation_choice):
    # Array that will contain the solution
    height = int(height)
    print("width: ", x)
    print("height: ", y)
    print("circuits: ", n_circuits)
    print("coord_x: ", coord_x)
    print("coord_y: ", coord_y)

    # Creating the image
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.xlim([0, plate])
    plt.ylim([0, height])

    # Randomizing the color for each rectangle
    for i in range(0, n_circuits):
        r = random.random()
        g = random.random()
        # To avoid white color
        if r >= 0.9 and g >= 0.9:
            b = 0
        # To avoid black
        elif r <= 0.1 and g >= 0.1:
            b = 1
        else:
            b = random.random()

        color = (r, g, b)

        # Start coordinate
        start_point = (coord_x[i], coord_y[i])
        # add rectangle to plot
        if rotation_sol[i]:
            ax.add_patch(Rectangle(start_point, y[i], x[i], facecolor=color, fill=True))
        else:
            ax.add_patch(Rectangle(start_point, x[i], y[i], facecolor=color, fill=True))

    radius = 0.1
    if height <= 27:
        radius = 0.008 * height
    else:
        radius = 0.003 * height

    # Adding point of coordinates
    for i in range(0, n_circuits):
        ax.add_patch(matplotlib.patches.Circle((coord_x[i], coord_y[i]), radius=radius, facecolor="black"))

    # Saving the image
    if rotation_choice:
        plt.savefig(os.path.join(output_dir, "rotation\\images\\SMT Solution{}.png".format(instance)))
    else:
        plt.savefig(os.path.join(output_dir, "model\\images\\SMT Solution{}.png".format(instance)))
    plt.close(fig)


def max_custom(vector):
    max = 0
    for value in vector:
        max = If(value > max, value, max)
    return max


if __name__ == "__main__":
    data = ["N. Instance", "Solution Status", "Height", "Time"]
    path_statistics = "model\\statistics\\statistics.csv"
    with (open(os.path.join(output_dir, path_statistics), 'w') as csv_):
        # Setup writer for csv and csv_rot
        writer = csv.writer(csv_)
        writer.writerow(data)
        csv_.flush()

        # Loop through the instances to solve
        for instance in range(1, 41):
            print("------- Working on instance: " + str(instance) + " -------")

            optimizer = ()
            plate, n_circuits, x, y = import_file(instance)
            height_min = min(y)
            if int(sum(y) / 2) <= max(y):
                height_max = sum(y)
            else:
                height_max = int(sum(y) / 2)

            coord_x = [Int(f"coord_x{i}") for i in range(n_circuits)]
            coord_y = [Int(f"coord_y{i}") for i in range(n_circuits)]
            rotation = [Bool(f"rotation{i}") for i in range(n_circuits)]

            rot_x = [Int(f"rot_x{i}") for i in range(n_circuits)]
            rot_y = [Int(f"rot_y{i}") for i in range(n_circuits)]
            circuit_area = [x[i] * y[i] for i in range(n_circuits)]

            height = max_custom([y[i] + coord_y[i] for i in range(n_circuits)])

            # CONSTRAINT, the objective height must be inside the max and the min height
            optimizer.add(And(height >= height_min, height <= height_max))

            for i in range(0, n_circuits):
                # CONSTRAINT, the coordinates must be placed inside the plate
                optimizer.add(coord_x[i] >= 0)
                optimizer.add(coord_y[i] >= 0)

                # CONSTRAINT, the coordinates summed with their lenght must be inside the plate
                optimizer.add(x[i] + coord_x[i] <= plate)
                optimizer.add(y[i] + coord_y[i] <= height)

            # CONSTRAINT, remove overlap respect to height and width
            for i in range(0, n_circuits):
                for j in range(0, n_circuits):
                    if i != j:
                        optimizer.add(Or((coord_x[i] + x[i] <= coord_x[j]), (coord_x[j] + x[j] <= coord_x[i]),
                                         (coord_y[i] + y[i] <= coord_y[j]), (coord_y[j] + y[j] <= coord_y[i])))

            # CONSTRAINT, use the cumulative to remove blank spaces and reduce time
            cumulative = []
            for u in y:
                cumulative.append(sum([If(And(coord_y[i] <= u, u < coord_y[i] + y[i]), x[i], 0)
                                       for i in range(len(coord_y))]) <= plate)
            optimizer.add(cumulative)

            # Set the objective to minimize (height) and timer
            optimizer.minimize(height)
            optimizer.set(timeout=300000)  # Time limit is 5 minutes, in milliseconds

            start = time.time()
            solution = optimizer.check()
            total_time = time.time() - start

            coord_x_sol = []
            coord_y_sol = []
            height_sol = ""
            x_rot_sol = []
            y_rot_sol = []
            rot_sol = []

            if solution == sat:
                print(f'Total time: {total_time * 1000:.1f} ms')
                model = optimizer.model()

                for i in range(n_circuits):
                    coord_x_sol.append(model.evaluate(coord_x[i]).as_long())
                    coord_y_sol.append(model.evaluate(coord_y[i]).as_long())
                    if model.evaluate(rotation[i]) == True:
                        rot_sol.append(True)
                    else:
                        rot_sol.append(False)

                height_sol = model.evaluate(height).as_string()

                output_file(instance, plate, n_circuits, x, y, coord_x_sol, coord_y_sol, rot_sol, height_sol,
                            total_time, False)
                plot(instance, plate, n_circuits, x, y, coord_x_sol, coord_y_sol, rot_sol, height_sol,
                     False)
                writer.writerow([instance, solution, height_sol, round(total_time, 4)])
                csv_.flush()
                print(f"The best height is {height_sol}")
            else:
                writer.writerow([instance, "unknown", "NULL", round(total_time, 4)])
                print("#### !Unsatisfiable problem! ####")
    csv_.close()
