# -*- coding: utf-8 -*-

import turtle

from tsp import generate_cities, random_seq, create_city


# screen setup
s = turtle.getscreen()
s.title("Traveling Salesman Problem")
s.setworldcoordinates(-1, -1, 15, 15)

# turtle for drawing
t = turtle.Turtle()
t.speed(0)
t.hideturtle()


def draw_city(x, y, name="City"):
    t.penup()  # not drawing

    # draw the circle
    t.goto(x, y)
    t.dot(20, "blue")

    # write the name near
    t.goto(x + 0.2, y)
    t.write(name, font=("Arial", 12, "normal"))


def draw_path(seq, cities):
    # seq: [2, 3, 1, 6, 4, 5]
    # cities: list of City

    # first city
    start = cities[seq[0]]

    t.penup()  # not drawing
    t.goto(start.x, start.y)
    t.pendown()  # drawing

    for i in range(1, len(seq)):
        # current city
        current = cities[seq[i]]
        t.goto(current.x, current.y)

    # return to first city
    # t.goto(start.x, start.y)


if __name__ == '__main__':
    ncities = 10
    # create random coordinates
    _, C = generate_cities(ncities)
    # create City object for each coordinate
    # returns a list of City
    cities = create_city(C)

    for city in cities:
        # draw cities as dots and place their names
        draw_city(city.x, city.y, city.name)

    # create a random cities sequence
    rand_seq = random_seq(ncities)
    print(f"Start with: {cities[rand_seq[0]]}")
    print(f"End with: {cities[rand_seq[-1]]}")

    # draw path between cities
    draw_path(rand_seq, cities)

    turtle.done()


