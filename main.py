#!/usr/bin/env python3

# Learning points:
# * Initial temperature and temperature schedule depends heavily on the problem (must make sure it is hot enough!)
# * Lots of potential in the way the algorithm proposes new solutions

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os

EPSILON=0.5
STEPS=10**6
INIT_TEMP=10**6
N_SOLVE=10

def f(x,y):
  return 2*x**6 - 12.2*x**5 + 21.2*x**4 + 6.2*x - 6.4*x**3 \
      - 4.7*x**2 + y**6 - 11*y**5 + 43.3*y**4 - 10*y \
      - 74.8*y**3 + 56.9*y**2 - 4.1*x*y - 0.1*(y**2)*(x**2) \
      + 0.4*y**2*x + 0.4*(x**2)*y

def sa(f):
  debug = os.getenv("DEBUG")
  s_x, s_y = 0, 0 # current solution
  for k in range(STEPS):
    t = INIT_TEMP/(1+math.log(1+k))
    if debug: print(f"Temperature Index = {k}, Temperature = {t:.2f}, Current solution = ({s_x:.4f}, {s_y:.4f}), f(solution) = {f(s_x,s_y)}")
    if t==0:
      return s_x, s_y
    new_s_x, new_s_y = s_x+t/(10**5)*random.random(), s_y+t/(10**5)*random.random()
    delta_energy = f(new_s_x,new_s_y) - f(s_x,s_y)
    if delta_energy<0 or random.random() > math.e**(delta_energy/t):
      s_x, s_y = new_s_x, new_s_y
  return s_x, s_y

def plot(x,y,f):
  X, Y = np.meshgrid(x,y)
  zs = np.array(f(np.ravel(X), np.ravel(Y)))
  Z = zs.reshape(X.shape)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.contour3D(X,Y,Z,50,cmap="autumn_r")
  fig.savefig("plot.png")
  plt.close(fig)

def main():
  x = np.arange(-1,3,0.01)
  y = np.arange(0,4,0.01)
  plot(x,y,f)
  for _ in range(N_SOLVE):
    solution = sa(f)
    print(f"Solution = {solution}, f(solution) = {f(solution[0],solution[1])}")

if __name__=="__main__":
  main()
