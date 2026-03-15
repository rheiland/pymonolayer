"""
Growing monolayer model in Python 
=================================

This initial version has no contact inhibition. The goal is to understand why PhysiCell seems
to differ from Chaste, in terms of overcrowding/overlapping cells in the center of the monolayer
when it gets too large.

Usage: <repulsion(e.g., 10)> <max cells, e.g., 500> <win size> <split type(0-2)>

where "split type" is:
  0 - like PhysiCell: keep 1 daughter at parent's position; other daughter randomly positioned so just touching
  1 - random equi-split from parent; no overlap
  2 - random equi-split from parent; slight overlap

* Cell size maintenance (parameter: target cell area A_0(t))
* start with a single cell at the origin, with area = A_0(t)
* Cell growth (parameters: linear growth rate α)
* Cell division: cells divide when A(t) = X * A_0(0) 
  (parameter:  X ∼ N(2, 0.4^2), truncated such that X > 0, 
   redraw X otherwise). 
   Observable for an isolated cell: cell cycle duration T = A0(0)/α)


This is the slower implementation of the model, to hopefully make it more obvious how
cells grow and divide. (The faster version is abm.py, uses numpy, and is a bit more challenging to see
the details.)

"""
import sys
import math
import random
# import uuid
import csv
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import mpld3

MAX_STEPS = 50000
MAX_STEPS = 500000

print("# args=",len(sys.argv))
if len(sys.argv) < 5:
    print("Usage: <repulsion(10)> <max cells> <win size> <split type(0-2)>\n")
    exit()
idx=1
# repulsion = float(sys.argv[idx])
cell_cell_repulsion = float(sys.argv[idx])
idx+=1
MAX_AGENTS = int(sys.argv[idx])
idx+=1
win_size = float(sys.argv[idx])
idx+=1
daughter_split = int(sys.argv[idx])

# dt_mechanics = 1.0
dt_mechanics = 0.1
dt_mechanics = 1.0
# grow_multiple = 6
dt_growth = 10 * dt_mechanics   # make integral multiple of dt_mechanics
dt_growth = 1
# max_relax_steps = int(1/dt_mechanics)
print(f"dt_mechanics= {dt_mechanics}")   # , max_relax_steps= {max_relax_steps}")
print(f"dt_growth= {dt_growth}") # ,  (multiple={grow_multiple})")

max_ID = 0

@dataclass
class Agent:
    """
    An agent with a 2D position, area, and constant growth rate.

    Attributes:
        x, y          : Position 
        vel_x, vel_y  : Velocity 
        prev_vel_x, prev_vel_y  : previous velocity 
        area        : Current area of the agent.
        division_area : area at which the agent divides.
        norm_rand     : X ∼ N(2, 0.4^2), truncated such that X > 0, redraw X otherwise
        gamma         : cell's free surface fraction (not overlapping with nbrs)
        ID            : unique integer identifier
        time_in_cycle : 
    """
    x: float
    y: float
    vel_x: float
    vel_y: float
    prev_vel_x: float
    prev_vel_y: float
    area: float = 78.54  # pi * 5^2
    division_area: float = 157.08        # division area=7.07
    norm_rand: float = 2.0
    gamma: float = 0.0
    ID: int = 0
    # time_step: int = 0
    time_in_cycle: float = 0.0

    def grow(self,time,dt) -> None:
        A_0 = 78.5
        r = 0.002265

        # A(t) = A₀ + r·t
        self.area = A_0 + r * self.time_in_cycle   # r = 0.002265

    def should_divide(self) -> bool:
        """Return True if this agent has reached or exceeded its division area."""
        return self.area >= self.division_area

    def divide(self, separation: float = 0.5, noise: float = 0.1) -> tuple["Agent", "Agent"]:
        global max_ID
        """
        Divide this agent into two daughter agents.

        The daughters have 1/2 the area of the parent. One daughter remains in the parent's position;
        the other daughter is placed at a random position around the other, so they just touch.

        Returns:
            A tuple of two new Agent instances (daughter_1, daughter_2).
        """

        # print(f"dividing cell ID {self.ID} has area {self.area}")
        daughter_area = self.area / 2
        daughter_radius = math.sqrt(daughter_area/math.pi)

        max_ID += 1

        theta = random.random() * 6.283185307179    # 2 * math.pi
        xvec = math.cos(theta) * daughter_radius * 2
        yvec = math.sin(theta) * daughter_radius * 2

        nr1 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4
        while nr1 < 0:
             nr1 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4

        nr2 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4
        while nr2 < 0:
             nr2 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4

        if daughter_split == 0:     # PhysiCell: daughter = parent x,y
            d1 = Agent(x=self.x,      y=self.y,      vel_x=0, vel_y=0, prev_vel_x=0, prev_vel_y=0,
                    area=daughter_area, 
                    division_area=78.54 * nr1, norm_rand=nr1, gamma=0, ID=self.ID, time_in_cycle=0.)
            d2 = Agent(x=self.x + xvec, y=self.y + yvec, vel_x=0, vel_y=0, prev_vel_x=0, prev_vel_y=0,
                    area=daughter_area, 
                    division_area=78.54 * nr2, norm_rand=nr2, gamma=0, ID=max_ID, time_in_cycle=0.)
        elif daughter_split == 1:   # equi-split
            xvec /= 2
            yvec /= 2
            d1 = Agent(x=self.x + xvec,  y=self.y + yvec,  vel_x=0, vel_y=0, prev_vel_x=0, prev_vel_y=0,
                    area=daughter_area, 
                    division_area=78.54 * nr1, norm_rand=nr1, gamma=0, ID=self.ID, time_in_cycle=0.)
            d2 = Agent(x=self.x - xvec, y=self.y - yvec, vel_x=0, vel_y=0, prev_vel_x=0, prev_vel_y=0,
                    area=daughter_area, 
                    division_area=78.54 * nr2, norm_rand=nr2, gamma=0, ID=max_ID, time_in_cycle=0.)
        elif daughter_split == 2:   # equi-split w/ overlap
            xvec /= 2.5
            yvec /= 2.5
            d1 = Agent(x=self.x + xvec,      y=self.y + yvec,      vel_x=0, vel_y=0, prev_vel_x=0, prev_vel_y=0,
                    area=daughter_area, 
                    division_area=78.54 * nr1, norm_rand=nr1, gamma=0, ID=self.ID, time_in_cycle=0.)
            d2 = Agent(x=self.x - xvec, y=self.y - yvec, vel_x=0, vel_y=0, prev_vel_x=0, prev_vel_y=0,
                    area=daughter_area, 
                    division_area=78.54 * nr2, norm_rand=nr2, gamma=0, ID=max_ID, time_in_cycle=0.)

        return d1, d2


class Simulation:
    """
    Manages a population of Agents over discrete time steps.

    At each step:
      1. Each cell grows
      2. Cells that exceed division_area are replaced by two daughters.
    """

    # Relevant parameter values will be passed in
    def __init__(
        self,
        n_seed: int = 0,
        division_area: float = 0.0,
        norm_rand: float = 0.0,
        initial_area: float = 0.0,
    ):
        self.agents: list[Agent] = [
            Agent(
                x=0,   # position
                y=0,
                vel_x=0,   # velocity
                vel_y=0,
                prev_vel_x=0,   # previous velocity
                prev_vel_y=0,
                area=initial_area,
                division_area=division_area,
                norm_rand=norm_rand,
                gamma=0,
                time_in_cycle=0,
            )
            for _ in range(n_seed)
        ]
        # self.time: int = 0
        self.time: float = 0.
        # self.history: list[dict] = []

    def update_position(self, dt) -> None:
        # Forward Euler: x_new = x + vel * dt
        for agent in self.agents:
            agent.x += agent.vel_x * dt
            agent.y += agent.vel_y * dt

            # velocity[0]=0; velocity[1]=0; velocity[2]=0;
            agent.vel_x = 0.0
            agent.vel_y = 0.0

    def update_position_AB(self, dt) -> None:
        # use Adams-Bashforth 
        # if( constants_defined == false )
        if True:
            d1 = dt; 
            d1 *= 1.5; 
            d2 = dt; 
            d2 *= -0.5; 
            # constants_defined = true; 

        for agent in self.agents:
            # axpy( &position , d1 , velocity );    # position[i] += velocity[i] * d1
            agent.x += agent.vel_x * d1
            agent.y += agent.vel_y * d1

            # axpy( &position , d2 , previous_velocity );  
            agent.x += agent.prev_vel_x * d2
            agent.y += agent.prev_vel_y * d2


            # previous_velocity = velocity; 
            agent.prev_vel_x = agent.vel_x
            agent.prev_vel_y = agent.vel_y

            # velocity[0]=0; velocity[1]=0; velocity[2]=0;
            agent.vel_x = 0.0
            agent.vel_y = 0.0

    def update_gamma(self) -> None:
        pi_inv = 1.0/math.pi
        for agent in self.agents:
            agent_r = math.sqrt(agent.area/math.pi)  # my radius; todo: speed up by adding radius to Agent
            nbrs = []
            sum_j = 0.0
            for agent2 in self.agents:
                if agent.ID != agent2.ID:
                    xdel = agent.x - agent2.x
                    ydel = agent.y - agent2.y
                    dist = math.sqrt(xdel*xdel + ydel*ydel)  # distance between me and all other cells
                    # distance = std::max(sqrt(distance), 0.00001); 
                    agent2_r = math.sqrt(agent2.area/math.pi)   # radius of other cell
                    R_sum = agent_r + agent2_r   # sum of radii
                    if dist < 1.05 * R_sum:      # qualify as a nbr
                        # print(f"update_gamma(): testing cells {agent.ID}:{agent2.ID}")
                        nbrs.append(agent2.ID)
                        phi_ij = ( dist*dist - agent2_r*agent2_r + agent_r*agent_r ) / (2 * dist * agent_r)
                        # sum_phi_ij += ( dist*dist - agent2_r*agent2_r + agent_r*agent_r ) / (2 * dist * agent_r)
                        phi_ij_2 = phi_ij * phi_ij
                        if phi_ij_2 < 1.0:
                            sum_j += math.sqrt(1.0 - phi_ij_2)
                        # else:
                        #     print("phi_ij = ",phi_ij)
                        # print("sum_j = ",sum_j)
            # free surface fraction of cell_i
            # agent.gamma = 1.0 - pi_inv * math.sqrt( 1.0 - sum_phi_ij )   # for all nbrs j:  1 - 1/pi * SUM_j (sqrt(1 - phi_ij)
            agent.gamma = 1.0 - pi_inv * sum_j   # for all nbrs j:  1 - 1/pi * SUM_j (sqrt(1 - phi_ij)
            # print(f"cell ID {agent.ID} has gamma {agent.gamma}")

            # if len(nbrs) > 0:
            #     print(f"cell ID {agent.ID} has nbrs {nbrs}")


    # involves the crucial assumptions related to "time"
    def step(self) -> int:
        global grow_count
        """
        Advance the simulation by one time step... or dt_mechanics time step?? (rwh)

        Returns:
            Number of division events that occurred this step.
        """
        survivors: list[Agent] = []
        divisions = 0

        #======================================================================
        # ========= cell growth (on one time scale) 

        # is it time to grow? is dt_growth
        # if self.time/dt_mechanics - int(self.time/dt_mechanics) < 0.01
        # print("grow_count=",grow_count )
        ratio = self.time / dt_growth
        grow_bool = abs(ratio - round(ratio)) < 1.e-6  # tolerance
        # print(f"----- check for doing growth: time={self.time}, time/dt_growth= {self.time/dt_growth} ")
        # if grow_count == 0:
        if self.time > 0 and grow_bool:
            # grow_count = grow_multiple
            if self.time < 5:
                print("----- doing growth at t=",self.time)
            for agent in self.agents:
                agent.grow(self.time,dt_growth)      # <---------------------- rwh: now use dt_mechanics
                if agent.ID == 0 and self.time < 5:
                    print(f"----- time={self.time}, ID={agent.ID}, time_in_cycle= {agent.time_in_cycle}")

                agent.time_in_cycle += dt_growth

                if agent.should_divide():
                    if len(self.agents) < 10:
                        print(f"----- cell ID {agent.ID} pre-dividing: area={agent.area}, time_in_cycle={agent.time_in_cycle}, norm_rand={agent.norm_rand:.3f}")

                    d1, d2 = agent.divide()

                    if len(self.agents) < 10:
                        print(f"       d1 ID {d1.ID} post-dividing: area={d1.area}, time_in_cycle={d1.time_in_cycle}, norm_rand={d1.norm_rand:.3f}")
                        print(f"       d2 ID {d2.ID} post-dividing: area={d2.area}, time_in_cycle={d2.time_in_cycle}, norm_rand={d2.norm_rand:.3f}")

                    survivors.extend([d1, d2])
                    divisions += 1
                else:
                    survivors.append(agent)
                # print("survivors=",survivors)

            self.agents = survivors
        # else :
            # grow_count -= 1

        # void standard_update_cell_velocity( Cell* pCell, Phenotype& phenotype, double dt)
        # for(neighbor = pCell->get_container()->agent_grid[pCell->get_current_mechanics_voxel_index()].begin(); neighbor != end; ++neighbor)
	    # {
		#     pCell->add_potentials(*neighbor);
	    # }


        #======================================================================
        # ======= cell movement (repulsion, possibly on a different time scale) 

        # ---- Update each agent's velocity vector:  like Cell::add_potentials(Cell* other_agent)
        # cell_cell_repulsion = repulsion

        for relax_steps in range(1):   # experimental...
            for agent in self.agents:
                agent_r = math.sqrt(agent.area/math.pi)

                # previous_velocity = velocity; 
                agent.prev_vel_x = agent.vel_x
                agent.prev_vel_y = agent.vel_y

                agent.vel_x = 0
                agent.vel_y = 0
                for agent2 in self.agents:
                    if agent.ID != agent2.ID:
                        # displacement[i] = position[i] - (*other_agent).position[i]; 
                        xdel = agent.x - agent2.x   # "displacement" vector in C++
                        ydel = agent.y - agent2.y
                        dist = math.sqrt(xdel*xdel + ydel*ydel)
                        # distance = std::max(sqrt(distance), 0.00001); 

                        agent2_r = math.sqrt(agent2.area/math.pi)
                        if dist < 1.25 * (agent_r + agent2_r):
                            R = agent_r + agent2_r
                            if dist > R:
                                temp_r = 0
                            else:
                                # print(f"update velocity of cell {agent.ID} due to nbr {agent.ID}")
                                temp_r = -dist  # -d
                                temp_r /= R # -d/R
                                temp_r += 1.0 # 1-d/R
                                temp_r *= temp_r # (1-d/R)^2 
                                
                                # add the relative pressure contribution 
                                # state.simple_pressure += ( temp_r / simple_pressure_scale ); // New July 2017 

                            # PhysiCell C++ code
                            # double effective_repulsion = sqrt( phenotype.mechanics.cell_cell_repulsion_strength * other_agent->phenotype.mechanics.cell_cell_repulsion_strength ); 
                            # effective_repulsion = math.sqrt(cell_cell_repulsion_strength * cell_cell_repulsion_strength) 
                            # effective_repulsion = cell_cell_repulsion_strength  
                            # temp_r *= effective_repulsion 
                            temp_r *= cell_cell_repulsion

                            # skip over adhesion, there is none

                            # if( fabs(temp_r) < 1e-16 )
                            # { return; }
                            temp_r /= dist

                            agent.vel_x += xdel * temp_r
                            agent.vel_y += ydel * temp_r


            # ----  Update each agent's position using its velocity
            # for idx_mech in range(1):   # experimental
            self.update_position_AB(dt_mechanics)

        self.update_gamma()

        # self.time += 1
        self.time += dt_mechanics
        # if self.time > 3:
        #     exit()

        return divisions

    def run(self, steps: int, max_agents: int = 500) -> None:
        """
        Run the simulation for a given number of steps or until max_agents is reached.

        Args:
            steps:      Maximum number of time steps to run.
            max_agents: Stop early if population exceeds this threshold.
        """

        for istep in range(steps):
            self.step()  # will now step by dt_mechanics
            if len(self.agents) >= max_agents:
                print(f"  Stopped early at t={self.time}: {len(self.agents)} agents.")
                break
            # if istep > 5670:
            #     print(f"  Stopped at step...")
            #     break


    def precompute(self, steps: int, max_agents: int = 500) -> None:
        """
        Run the full simulation up-front and store every frame's agent list.
        Required before calling interactive_viewer().

        Args:
            steps:      Number of time steps to pre-compute.
            max_agents: Stop early if population exceeds this.
        """
        import copy
        # print(f"Pre-computing {steps} steps...", end=" ", flush=True)
        print(f"Pre-computing {steps} steps (max)...")
        # Frame 0 = initial state (before any stepping)
        self._frames: list[list[Agent]] = [copy.deepcopy(self.agents)]

        # print("--> ",file_out)
        save_csv = False
        if save_csv:
            csv_filename = 'abm_AB.csv'
            csv_file = open(csv_filename, mode='w', newline='')
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow(['time','ID','x_pos','y_pos','radius','x_vec','y_vec','norm_rand_i','gamma'])

            #     writer.writerow(['x_pos','y_pos','radius_i','norm_rand_i'])
            #     for agent in self.agents:
            #         # writer.writerow([x_pos[jdx],y_pos[jdx],radius_i[jdx],f_i[jdx],a_i[jdx]])
            #         radius = math.sqrt(agent.area / math.pi)
            #         writer.writerow([agent.x, agent.y, radius, agent.norm_rand])
        for istep in range(steps):
            self.step()

            if istep % 10 == 0:   # rwh: modulo output of frames
                if save_csv:
                    for agent in self.agents:
                        csv_writer.writerow([self.time, agent.ID, agent.x,agent.y, math.sqrt(agent.area/math.pi),agent.prev_vel_x,agent.prev_vel_y,agent.norm_rand,agent.gamma])

                self._frames.append(copy.deepcopy(self.agents))
                if len(self.agents) >= max_agents:
                    print(f"(stopped early at t={self.time} — {len(self.agents)} agents)", end=" ")
                    break

            # if save_csv:
            #     csv_file.close()

        print(f"done. {len(self._frames)} frames stored.")

        if save_csv:
            csv_file.close()
            print(f"---> ",csv_filename)

        # save final results in .csv for post-processing, e.g., to compute f_i and a_i
        # file_out = f'py_monolayer_small.csv'
        # print("--> ",file_out)
        # with open(file_out, "w", newline="") as file:
        #     writer = csv.writer(file)

        #     writer.writerow(['x_pos','y_pos','radius_i','norm_rand_i'])
        #     for agent in self.agents:
        #         # writer.writerow([x_pos[jdx],y_pos[jdx],radius_i[jdx],f_i[jdx],a_i[jdx]])
        #         radius = math.sqrt(agent.area / math.pi)
        #         writer.writerow([agent.x, agent.y, radius, agent.norm_rand])


    def interactive_viewer(self, title: str = "Agent-Based Model", interval: int = 100) -> None:
        """
        Open an interactive matplotlib window with:
          ◀◀  Step back one frame
          ▶▶  Step forward one frame
          ▶ / ■  Play / Pause the full simulation
          A time-step slider for direct scrubbing

        Requires precompute() to have been called first.
        """
        import copy
        from matplotlib.widgets import Button, Slider
        from matplotlib.animation import FuncAnimation

        if not hasattr(self, "_frames"):
            raise RuntimeError("Call precompute() before interactive_viewer().")

        frames     = self._frames
        n_frames   = len(frames)
        cmap       = plt.cm.plasma

        fig = plt.figure(figsize=(6, 6), facecolor="#0d0d0d")
        fig.suptitle(title, fontsize=13, fontweight="bold", color="white")

        ax_space = fig.add_axes([0.1, 0.2, 0.8, 0.7])   # left: spatial: left, bottom, w, h)

        ax_space.set_facecolor("#111111")
        ax_space.tick_params(colors="white")
        for sp in ax_space.spines.values():
            sp.set_edgecolor("#444444")

        ax_space.set_aspect("equal")
        ax_space.set_xlabel("x", color="white")
        ax_space.set_ylabel("y", color="white")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 10))
        sm.set_array([])

        # Stable world bounds (union of all frame extents)
        all_x = [a.x for f in frames for a in f]
        all_y = [a.y for f in frames for a in f]
        # max_r = max((a.area for f in frames for a in f), default=1)  # now doing area, not radii
        # pad   = max_r * 2
        pad   = 20
        # ax_space.set_xlim(min(all_x) - pad, max(all_x) + pad)
        # ax_space.set_ylim(min(all_y) - pad, max(all_y) + pad)

        # win_size = 100  # rwh: hard-code for now
        # win_size = 150  # rwh: hard-code for now
        print("xlim (win_size): ",win_size)
        ax_space.set_xlim(-win_size, win_size)
        ax_space.set_ylim(-win_size, win_size)

        title_text = ax_space.set_title("", color="white", fontsize=10)

        # ------------------------------------------------------------------ #
        # Widget axes
        # ------------------------------------------------------------------ #
        btn_color      = "#1e2d3d"
        btn_hover      = "#2e4d6d"

        ax_slider = fig.add_axes([0.10, 0.10, 0.80, 0.03], facecolor="#1a1a2e")
        ax_back   = fig.add_axes([0.28, 0.02, 0.08, 0.055], facecolor=btn_color)
        ax_play   = fig.add_axes([0.38, 0.02, 0.10, 0.055], facecolor=btn_color)
        ax_fwd    = fig.add_axes([0.50, 0.02, 0.08, 0.055], facecolor=btn_color)

        slider   = Slider(ax_slider, "t", 0, n_frames - 1,
                          valinit=0, valstep=1, color="#00d4ff")
        slider.label.set_color("white")
        slider.valtext.set_color("white")

        btn_back = Button(ax_back, "◀◀", color=btn_color, hovercolor=btn_hover)
        btn_play = Button(ax_play, "▶ Play", color=btn_color, hovercolor=btn_hover)
        btn_fwd  = Button(ax_fwd,  "▶▶", color=btn_color, hovercolor=btn_hover)

        for btn in (btn_back, btn_play, btn_fwd):
            btn.label.set_color("white")
            btn.label.set_fontsize(10)

        # ------------------------------------------------------------------ #
        # Draw helpers
        # ------------------------------------------------------------------ #
        state = {"frame": 0, "playing": False}

        def _draw_frame(idx: int) -> None:
            idx = max(0, min(idx, n_frames - 1))
            state["frame"] = idx

            agent_list = frames[idx]
            # max_gen    = max((a.generation for a in agent_list), default=0) or 1

            for p in list(ax_space.patches):
                p.remove()
            for p in list(ax_space.texts):
                p.remove()

            # ax_space.clear()
            for agent in agent_list:
                # color = cmap(agent.generation / max(max_gen, 10))
                ax_space.add_patch(patches.Circle(
                    (agent.x, agent.y), 
                    radius=math.sqrt(agent.area / math.pi),
                    facecolor='black', alpha=1.0,
                    linewidth=1.0, edgecolor="white",))

                # ax_space.text(agent.x,agent.y,f'{agent.ID},{agent.gamma:.2f}',color='white', fontsize=7)
                ax_space.text(agent.x,agent.y,f'{agent.ID}',color='white', fontsize=7)
                if agent.gamma < 0.5:
                    ax_space.text(agent.x-3,agent.y-3,f'{agent.gamma:.2f}',color='red', fontsize=7)
                else:
                    ax_space.text(agent.x-3,agent.y-3,f'{agent.gamma:.2f}',color='white', fontsize=7)

            # print("self.time=",self.time)   # 2519.7999...
            # title_text.set_text(
            #     f"frame= {idx} | agents= {len(agent_list)} | dt_mech={dt_mechanics}"  #   |   max gen = {max_gen}"
            #     # f"t = {idx}   |   agents = {len(agent_list)}"  #   |   max gen = {max_gen}"
            # )
            title_text.set_text(f'ID={agent_list[0].ID}, t in cycle= {agent_list[0].time_in_cycle}, area cell 0= {agent_list[0].area}')

            # pop_line.set_data(all_times[:idx + 1], all_pops[:idx + 1])
            # time_marker.set_xdata([idx, idx])

            # Update slider without triggering its callback
            slider.eventson = False
            slider.set_val(idx)
            slider.eventson = True

            fig.canvas.draw_idle()

            # mpld3.save_html(fig, 'index.html') # Save directly to an HTML file



        # ------------------------------------------------------------------ #
        # Button / slider callbacks
        # ------------------------------------------------------------------ #
        def on_back(_event):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(state["frame"] - 1)

        def on_fwd(_event):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(state["frame"] + 1)

        def on_play(_event):
            state["playing"] = not state["playing"]
            btn_play.label.set_text("■ Pause" if state["playing"] else "▶ Play")
            fig.canvas.draw_idle()

        def on_slider(val):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(int(val))

        btn_back.on_clicked(on_back)
        btn_fwd.on_clicked(on_fwd)
        btn_play.on_clicked(on_play)
        slider.on_changed(on_slider)

        # ------------------------------------------------------------------ #
        # Timer-driven playback
        # ------------------------------------------------------------------ #
        def _tick(_frame):
            if state["playing"]:
                nxt = state["frame"] + 1
                if nxt >= n_frames:
                    state["playing"] = False
                    btn_play.label.set_text("▶ Play")
                else:
                    _draw_frame(nxt)

        anim = FuncAnimation(fig, _tick, interval=interval, cache_frame_data=False)

        # Draw initial frame
        _draw_frame(0)
        plt.show()

        return anim  # keep reference alive so GC doesn't kill the timer


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation  # ensure import available

    random.seed(42)

    sim = Simulation(
        n_seed= 1,              # Start with n agents
        division_area= 157.08,  #  vs. double area = 2 * 78.54 = 157.08
        norm_rand= 2.0,         #  X ~ N(2, 0.4^2)
        initial_area= 78.54,    #  area = pi * r^2 = pi * 25 = 78.54
    )

    # 1. Pre-compute all frames (enables backward scrubbing)
    sim.precompute(steps=MAX_STEPS, max_agents=MAX_AGENTS)  # rwh

    # 2. Open the interactive widget viewer
    sim.interactive_viewer(
        title="Growing Monolayer (serial)",
        interval=1,   # rwh: ms between auto-play frames
    )
