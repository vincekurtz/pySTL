#!/usr/bin/env python3

##
#
# A simple example of using our framework to
# evaluate performance on a simple reach-avoid problem
#
##

import numpy as np
from pySTL import STLFormula
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# The scenario we're interested in involves moving past an obstacle
# and eventually reaching a target region. First, we'll plot these regions for visualization
fig, ax = plt.subplots(1)
ax.set_xlim((0,12))
ax.set_ylim((0,12))

obstacle = Rectangle((3,4),2,2,color='red',alpha=0.5)
target = Rectangle((7,8),1,1, color='green',alpha=0.5)

ax.add_patch(obstacle)
ax.add_patch(target)

###########################################################################
# STL Specification                                                       #
###########################################################################

# Now we define the property we're interested in:
#   ALWAYS don't hit the obstacle and EVENTUALLY reach the goal in 20 steps.

# We'll assume the signal that our specification is over is a list of x,y coordinates
# at each timestep. We'll build up our specification from predicates, with the use of 
# this handy helper function:

def in_rectangle_formula(xmin,xmax,ymin,ymax):
    """
    Returns an STL Formula denoting that the signal is in
    the given rectangle.
    """
    # These are essentially predicates, since their robustnes function is
    # defined as (mu(s_t) - c) or (c - mu(s_t))
    above_xmin = STLFormula(lambda s, t : s[t,0] - xmin)
    below_xmax = STLFormula(lambda s, t : -s[t,0] + xmax)
    above_ymin = STLFormula(lambda s, t : s[t,1] - ymin)
    below_ymax = STLFormula(lambda s, t : -s[t,1] + ymax)

    # above xmin and below xmax ==> we're in the right x range
    in_x_range = above_xmin.conjunction(below_xmax)
    in_y_range = above_ymin.conjunction(below_ymax)

    # in the x range and in the y range ==> in the rectangle
    in_rectangle = in_x_range.conjunction(in_y_range)

    return in_rectangle

hit_obstacle = in_rectangle_formula(3,5,4,6)
at_goal = in_rectangle_formula(7,8,8,9)

obstacle_avoidance = hit_obstacle.negation().always(0,20)
reach_goal = at_goal.eventually(0,20)

full_specification = obstacle_avoidance.conjunction(reach_goal)

###########################################################################
# Evaluating Candiate Trajectories                                        #
###########################################################################

# Finally, we can test some possible trajectories 
# to see if they satisfy our specification

# Trajectory 1 reaches the goal, but fails to avoid the obstacle
t1 = np.array([[0.4*i,0.45*i] for i in range(25)])
p1 = ax.plot(t1[:,0],t1[:,1],label="$\\rho$ = %0.2f"%full_specification.robustness(t1,0))

print("TRAJECTORY 1:")
print("Obstacle Avoidance Robustness Score: %s" % obstacle_avoidance.robustness(t1,0))
print("Goal Reaching Robustness Score: %s" % reach_goal.robustness(t1,0))
print("Full Specification Robustness Score: %s" % full_specification.robustness(t1,0))
print("")

# Trajectory 2 avoids the obstacle, but doesn't reach the goal
t2 = np.array([[0.1*i,0.3*i] for i in range(25)])
p2 = ax.plot(t2[:,0],t2[:,1],label="$\\rho$ = %0.2f"%full_specification.robustness(t2,0))

print("TRAJECTORY 2:")
print("Obstacle Avoidance Robustness Score: %s" % obstacle_avoidance.robustness(t2,0))
print("Goal Reaching Robustness Score: %s" % reach_goal.robustness(t2,0))
print("Full Specification Robustness Score: %s" % full_specification.robustness(t2,0))
print("")

# Trajectory 3 avoids the obstacle, and reaches the goal
t3 = np.array([[0.43*i,0.0015*i**3] for i in range(25)])
p3 = ax.plot(t3[:,0],t3[:,1],label="$\\rho$ = %0.2f"%full_specification.robustness(t3,0))

print("TRAJECTORY 3:")
print("Obstacle Avoidance Robustness Score: %s" % obstacle_avoidance.robustness(t3,0))
print("Goal Reaching Robustness Score: %s" % reach_goal.robustness(t3,0))
print("Full Specification Robustness Score: %s" % full_specification.robustness(t3,0))
print("")

plt.legend()
plt.title("Specification: Always avoid red and eventually reach green")

plt.show()
