# -*- coding: utf-8 -*-

import turtle
import tkinter as tk


# def do_stuff():
#     for color in ["red", "yellow", "green"]:
#         my_lovely_turtle.color(color)
#         my_lovely_turtle.right(120)


# def press():
#     do_stuff()


## turtle in standalone mode
# if __name__ == "__main__":
#     screen = turtle.Screen()
#     screen.bgcolor("cyan")

#     canvas = screen.getcanvas()
#     button = tk.Button(canvas.master, text="Press me", command=press)

#     canvas.create_window(-200, -200, window=button)

#     my_lovely_turtle = turtle.Turtle(shape="turtle")

#     turtle.done()


## turtle in embedded mode
# if __name__ == "__main__":
#     root = tk.Tk()

#     canvas = tk.Canvas(root)
#     canvas.config(width=600, height=200)
#     canvas.pack(side=tk.LEFT)

#     screen = turtle.TurtleScreen(canvas)
#     screen.bgcolor("cyan")

#     button = tk.Button(root, text="Press me", command=press)
#     button.pack()

#     my_lovely_turtle = turtle.RawTurtle(screen, shape="turtle")

#     root.mainloop()


## OOP version
class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Raw Turtle")
        self.canvas = tk.Canvas(master)
        self.canvas.config(width=600, height=200)
        self.canvas.pack(side=tk.LEFT)
        self.screen = turtle.TurtleScreen(self.canvas)
        self.screen.bgcolor("cyan")
        self.button = tk.Button(self.master, text="Press me", command=self.press)
        self.button.pack()
        self.my_lovely_turtle = turtle.RawTurtle(self.screen, shape="turtle")
        self.my_lovely_turtle.color("green")

    def do_stuff(self):
        for color in ["red", "yellow", "green"]:
            self.my_lovely_turtle.color(color)
            self.my_lovely_turtle.right(120)

    def press(self):
        self.do_stuff()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
