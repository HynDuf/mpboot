import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import os

def equilateral_triangle(side_length):
    height = np.sqrt(3) * side_length / 2

    vertex1 = [0, 0]
    vertex2 = [side_length, 0]
    vertex3 = [side_length / 2, height]

    return vertex1, vertex2, vertex3

def find_intersection(line1, line2):
    m1, b1 = line1
    m2, b2 = line2
    if m1 == m2:
        raise ValueError("Lines are parallel and do not intersect.")
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def distance_to_line(point, line_coefficients):
    x, y = point
    A, B, C = line_coefficients
    distance = abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
    return distance

def transform_to_points(p_nni, p_spr, p_tbr):
    assert(abs(p_nni + p_spr + p_tbr - 1) < 1e-9)
    line1 = (0, p_nni)
    line2 = (-np.sqrt(3), 2 - 2 * p_tbr)
    (x, y) = find_intersection(line1, line2)
    assert(y == p_nni)
    assert(np.abs(distance_to_line((x, y), (-np.sqrt(3), -1, 2)) - p_tbr) < 1e-9)
    return (x, y)

def draw_perpendicular_line(ax, point, line_slope, line_intercept):
    if line_slope != 0:
        perpendicular_slope = -1 / line_slope
        perpendicular_intercept = point[1] - perpendicular_slope * point[0]

        intersection_x = (line_intercept - perpendicular_intercept) / (perpendicular_slope - line_slope)
        intersection_y = perpendicular_slope * intersection_x + perpendicular_intercept

        ax.plot([point[0], intersection_x], [point[1], intersection_y], linestyle='--', color='black', linewidth=4)

        length = np.sqrt((point[0] - intersection_x)**2 + (point[1] - intersection_y)**2)
        text = ""
        midx = (point[0] + + intersection_x) / 2
        midy = (point[1] + + intersection_y) / 2
        if line_intercept == 2:
            text = f'{length:.2f}'
            midx -= 0.02
            midy -= 0.03
        else:
            text = f'{length:.2f}'
            midx -= 0.09
            midy -= 0.04
        ax.text(midx, midy, text, rotation=0, color='black', fontsize=15)

    else:
        ax.plot([point[0], point[0]], [point[1], 0], linestyle='--', color='black', linewidth=4)

        length = np.abs(point[1])
        ax.text(point[0] + 0.01, point[1] / 2, f'{length:.2f}', rotation=0, color='black', fontsize=15)

def draw_aco_triangle_path(prob_tuples, out_path):
    side_length = 2 / np.sqrt(3)

    vertex1, vertex2, vertex3 = equilateral_triangle(side_length)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.plot([vertex1[0], vertex2[0]], [vertex1[1], vertex2[1]], color='blue', linewidth=16)  # NNI
    ax.plot([vertex2[0], vertex3[0]], [vertex2[1], vertex3[1]], color='purple', linewidth=8)   # SPR
    ax.plot([vertex3[0], vertex1[0]], [vertex3[1], vertex1[1]], color='green', linewidth=8) # TBR

    ax.set_xlim(0, vertex2[0])
    ax.set_ylim(0, vertex3[1])

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('NNI', fontsize=20)
    ax.set_ylabel('Percentages', fontsize=20)
    ax.text(0.18, 0.5, "SPR", rotation=0, color='black', fontsize=20)
    ax.text(0.90, 0.5, "TBR", rotation=0, color='black', fontsize=20)
    ax.set_xticks([])

    points = []

    for prob_tuple in prob_tuples:
        point = transform_to_points(*prob_tuple)
        points.append(point)

    for point in points:
        ax.scatter(point[0], point[1], marker='o', color='r', s=60, linewidths=9.3)
    indices = np.arange(len(points))
    norm_indices = indices / (len(points) - 1)

    cmap = plt.get_cmap('OrRd')
    ax.scatter([point[0] for point in points], 
                     [point[1] for point in points], 
                     marker='o', 
                     c=norm_indices, 
                     cmap=cmap, 
                     s=60, 
                     linewidths=9)

    draw_perpendicular_line(ax, points[-1], 0, 0)
    draw_perpendicular_line(ax, points[-1], -np.sqrt(3), 2)
    draw_perpendicular_line(ax, points[-1], np.sqrt(3), 0)

    ax.grid(True, which='both', linestyle='--', linewidth=3)
    plt.savefig(out_path)
    plt.close()

def read_csv_and_draw_triangle(csv_file, output_path):
    prob_tuples = []
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            nnis = int(row[0])
            sprs = int(row[1])
            tbrs = int(row[2])
            sum = nnis + sprs + tbrs
            p_nni = nnis / sum
            p_spr = sprs / sum
            p_tbr = tbrs / sum
            prob_tuples.append((p_nni, p_spr, p_tbr))
    
    draw_aco_triangle_path(prob_tuples, output_path)


def main():
    parser = argparse.ArgumentParser(description="Draw a triangle from CSV points and save it as a PNG file.")
    parser.add_argument('--input', required=True, help="Input CSV file containing the triangle points.")
    parser.add_argument('--output', help="Output PNG file to save the triangle image.")

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".png"

    if not args.output.lower().endswith(".png"):
        args.output += ".png"

    read_csv_and_draw_triangle(args.input, args.output)

if __name__ == "__main__":
    main()
