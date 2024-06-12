# -*- coding: utf-8 -*-

import turtle
import tkinter as tk
from tkinter import ttk

from tsp_solvers.tsp import tsp_fun, circle_cities, create_city, generate_cities
# from tsp_solvers import solve_brute_force, solve_swap, solve_multi_start, multi_start_ls
from tsp_solvers import solve_tsp, solvers_list, multi_start_ls


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Travelling Salesman Problem")

        # self.cities = cities  # TSP nodes
        # self.seq = seq  # cities path sequence

        self.canvas = tk.Canvas(master)
        self.canvas.config(width=600, height=600)
        self.canvas.pack(side=tk.LEFT)

        self.screen = turtle.TurtleScreen(self.canvas)
        self.screen.bgcolor("white")

        ## turtle object for drawing in the canvas
        self.my_turtle = turtle.RawTurtle(self.screen, shape="turtle")
        self.my_turtle.color("black")
        self.my_turtle.speed(10)
        # self.my_turtle.hideturtle()

        self.cost = None  # distance matrix
        self.opt_result = None  # assign after optimization
        self.map = None  # assign after drawing the nodes

        # ****************************** #

        ## clear canvas
        self.erase_butt = self.create_button(
            "Clear", lambda: self.my_turtle.clear())
        ## slider for number of cities
        self.ncity_slide = self.create_slider(
            "N", from_=1, to=50, orient=tk.HORIZONTAL)
        ## drop down menu for map layout
        self.map_layout_menu = self.create_menu(
            ["circle", "random"], "Map layout")
        ## draw TSP nodes
        self.draw_nodes_butt = self.create_button(
            "Draw cities", self._draw_map)
        ## drop down menu for solver to use
        self.solver_menu = self.create_menu(
            solvers_list, "Solver")
        ## drop down menu for local search method
        self.multi_start_menu = self.create_menu(
            multi_start_ls, "Local search")
        ## slider for sim annealing cooling rate
        self.cooling_rate = self.create_slider(
            "alpha", from_=0.0001, to=0.95, orient=tk.HORIZONTAL,
            resolution=0.0001)
        ## button for solve TSP and draw solution path
        self.solve_butt = self.create_button(
            "Run solver", lambda: self.solve(
                self.solver_menu.get()))
        ## display objective function value
        self.fun_lab = self.create_label("f(x): N/A")

    # *************************************************** #
    # tkinter widgets

    def create_button(self, text, command):

        button = tk.Button(self.master, text=text, command=command)
        button.pack()

        return button


    def create_slider(self, label, **kwargs):

        slider = tk.Scale(self.master, label=label, variable=tk.DoubleVar(), **kwargs)
        slider.pack()

        return slider


    def create_menu(self, options, title):

        frame = tk.Frame(self.master)
        frame.pack()

        combo = ttk.Combobox(frame, values=options)
        combo.set(title)
        combo.pack(padx=5, pady=5)

        return combo


    def create_label(self, text):

        label = tk.Label(self.master, text=text)
        label.pack()

        return label

    # *************************************************** #

    def solve(self, solver):
        # update this with solver

        if self.cost is None:
            raise RuntimeError("Distance matrix is None")

        if solver == "multi-start":
            options = dict(local_search=self.multi_start_menu.get())
            res = solve_tsp(tsp_fun, self.cost, solver, options)

        elif solver == "simulated-annealing":
            options = dict(local_search=self.multi_start_menu.get(),
                           cooling_rate=self.cooling_rate.get())
            res = solve_tsp(tsp_fun, self.cost, solver, options)

        else:
            res = solve_tsp(tsp_fun, self.cost, solver)

        ## assign attributes
        self.opt_result = res
        self._draw_edges(res.x, self.map)

        ## display objective function value
        self.fun_lab.config(text=f"f(x): {res.fun:.2f}")


    # *************************************************** #

    def _draw_city(self, x, y, name="City", col="blue"):
        # draw a single nodes
        # x,y: cartesian coordinates of the city
        # name: name to be displayed next to the city
        # col: city node color

        self.my_turtle.penup()  # not drawing
        self.my_turtle.goto(x, y)  # go to the coordinates
        self.my_turtle.dot(20, col)  # draw the node

        self.my_turtle.goto(x + 0.2, y)  # go near the node and write the name
        self.my_turtle.write(name, font=("Arial", 10, "normal"))

        self.screen.update()


    def _draw_nodes(self, cities):
        # draw nodes
        # cities: list of City

        start_city = cities[0]  # starting city

        for city in cities:
            self._draw_city(city.x, city.y, city.name)

        # draw starting city with different color
        self._draw_city(start_city.x, start_city.y, start_city.name, "red")

        self.screen.update()


    def _draw_map(self):
        # ncity: number of city to generate
        # layout: map layout, circle or random

        self.my_turtle.clear()

        ncity = self.ncity_slide.get()
        layout = self.map_layout_menu.get()

        ## generate cities (nodes) coordinates
        if layout == "circle":
            D, C, _ = circle_cities(ncity, r=150)

        elif layout == "random":
            D, C = generate_cities(ncity, 300)

        # assign distance matrix
        self.cost = D
        # create objects
        cities = create_city(C)  # list of City
        self.map = cities

        # draw nodes
        self._draw_nodes(cities)

    # *************************************************** #

    def _draw_edges(self, seq, cities, col="black"):
        # seq: [0,...,0] array_like
        # cities: list of City
        # col: edge color

        start_city = cities[seq[0]]  # starting city

        self.my_turtle.penup()  # not drawing
        self.my_turtle.goto(start_city.x, start_city.y)  # go to starting city

        self.my_turtle.pendown()  # drawing
        self.my_turtle.pensize(3)
        self.my_turtle.color(col)  # edge color

        for i in range(1, seq.size):
            current = cities[seq[i]]  # current city
            self.my_turtle.goto(current.x, current.y)

        self.screen.update()


if __name__ == "__main__":

    root = tk.Tk()
    root.title("Travelling Salesman Problem")

    app = App(root)

    root.mainloop()  # tkinter event loop
