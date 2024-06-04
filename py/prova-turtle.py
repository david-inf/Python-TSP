# -*- coding: utf-8 -*-

import turtle

from tsp import generate_cities, random_seq, create_city, circle_cities
from solvers import relax
from utils import diagnostic


# screen setup
s = turtle.Screen()
s.title("Traveling Salesman Problem")
s.setworldcoordinates(-11, -11, 11, 11)

# turtle for drawing
t = turtle.Turtle()
t.speed(0)
t.hideturtle()


def draw_city(x, y, name="City", col="blue"):

    t.penup()  # not drawing

    # go to the coordinates
    t.goto(x, y)
    # draw the node
    t.dot(20, col)

    # go near the node
    t.goto(x + 0.2, y)
    # wirte node name
    t.write(name, font=("Arial", 12, "normal"))

    s.update()


def draw_map(seq, cities):
    # seq: [2, 3, 1, 0, 6, 4, 5] 0 to ncity
    # cities: list of City

    start = cities[seq[0]]  # starting city
    end = cities[seq[-1]]  # ending sity
    other = cities[1:-1]  # middle cities

    for city in cities:
        draw_city(city.x, city.y, city.name)

    # draw starting city
    draw_city(start.x, start.y, start.name, "red")
    # draw ending city
    draw_city(end.x, end.y, end.name, "green")

    s.update()


def draw_path(seq, cities, col="black"):
    # seq: [2, 3, 1, 0, 6, 4, 5] np.arange(ncity)
    # cities: list of City

    # first city
    start = cities[seq[0]]

    t.penup()  # not drawing
    t.goto(start.x, start.y)

    t.pendown()  # drawing
    t.color(col)
    for i in range(1, len(seq)):
        # current city
        current = cities[seq[i]]
        t.goto(current.x, current.y)

    # return to first city
    # t.goto(start.x, start.y)


if __name__ == '__main__':
    ncities = 10
    # create random coordinates
    # _, C = generate_cities(ncities)
    D, C, _ = circle_cities(ncities)
    # create City object for each coordinate
    # returns a list of City
    cities = create_city(C)

    ### create a random cities sequence
    rand_seq = random_seq(ncities, [0, ncities-1])
    # print(f"Start with: {cities[rand_seq[0]]}")
    # print(f"End with: {cities[rand_seq[-1]]}")
    # print(f"Sequence: {rand_seq}")

    # # draw cities in the map
    draw_map(rand_seq, cities)
    # # draw path between cities
    draw_path(rand_seq, cities)

    ### solve the TSP
    res = relax(rand_seq, D)
    print(res)
    t.clear()
    draw_map(res.x, cities)
    # draw_path(rand_seq, cities, "red")
    draw_path(res.x, cities)
    # diagnostic(C, res.x, res.fun_seq)

    turtle.done()
