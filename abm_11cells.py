"""
11 overlapping horizontal cells relaxation model in Python 
=================================

- only mechanical repulsion/relaxation (no mechanical adhesion, no cell cycle/growth, ...nothing else)
- cell radius = 5
- ICs have nbr cells overlap half-way, i.e., from furthest left cell, their centers are: -25,-20,-15,..., 25
- print out time to reach 90% total relaxation (cells just touching), i.e., leftmost cell x= -45; rightmost x= 45

Usage: <repulsion(10)> <win size> <max_steps>

Note there is *nothing* related to cell growth in this script. It's sole purpose is to obtain the cell cycle duration (and cycle rate to double in size (area)).

e.g.,
$ python abm_11cells.py 10 50 2000   # repulsion=10 --> growth rate = 0.0885
$ python abm_11cells.py 20 50 2000   # repulsion=15 --> 0.1328
$ python abm_11cells.py 20 50 2000   # repulsion=20 --> 0.1772

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

print("# args=",len(sys.argv))
if len(sys.argv) < 4:
    print("Usage: <repulsion(10)> <win size> <max_steps>\n")
    exit()
idx=1
repulsion = float(sys.argv[idx])
print("repulsion=",repulsion)
idx+=1
win_size = float(sys.argv[idx])
idx+=1
max_steps = int(sys.argv[idx])

max_ID = 0

@dataclass
class Agent:
    """
    An agent with a 2D position, area, and constant growth rate.

    Attributes:
        x, y          : position 
        vel_x, vel_y  : velocity 
        prev_vel_x, prev_vel_y  : previous velocity 
        area          : current area of the agent.
        ID            : unique integer identifier
    """
    x: float
    y: float
    vel_x: float
    vel_y: float
    prev_vel_x: float
    prev_vel_y: float
    area: float
    ID: int = 0
    time_step: int = 0

class Simulation:
    """
    Manages a population of Agents over discrete time steps.

    At each step:
      * mechanics repulsion
    """

    # Relevant parameter values will be passed in
    def __init__(
        self,
        n_seed: int = 0,
        initial_area: float = 78.54,
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
            )
            for _ in range(n_seed)
            ]
        self.time: int = 0
        # self.history: list[dict] = []

    def load(self) -> None:
        for i, a in enumerate(self.agents):
            a.x             = i*5.0 - 25
            a.y             = 0.0
            a.vel_x         = 0.0
            a.vel_x         = 0.0
            a.prev_vel_x    = 0.0
            a.prev_vel_x    = 0.0
            a.area          = 78.54
            a.ID          = i
            print(i,a)

    def update_position(self, dt) -> None:
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


    # involves the crucial assumptions related to "time"
    def step(self) -> None:
        """
        Advance the simulation by one time step.
        """

        # l.614 of P*_standard_models.cpp
        # void standard_update_cell_velocity( Cell* pCell, Phenotype& phenotype, double dt)
        # for(neighbor = pCell->get_container()->agent_grid[pCell->get_current_mechanics_voxel_index()].begin(); neighbor != end; ++neighbor)
	    # {
		#     pCell->add_potentials(*neighbor);
	    # }

        # 2) Update each agent's velocity vector:  
        # like l.974: Cell::add_potentials(Cell* other_agent)

        # cell_cell_repulsion_strength = 10.0
        cell_cell_repulsion_strength = repulsion
        # cell_cell_repulsion_strength = 0.0

        max_relax_steps = 1   # 10 is equivalent to using dt_mechanics = 0.1 ??
        # print("----------------")
        for relax_step in range(max_relax_steps):   # isn't this violating our assumptions of time??
            # print("---- relax_step ",relax_step)

            for agent in self.agents:
                agent_r = math.sqrt(agent.area/math.pi)
                # agent.vel_x = 0
                # agent.vel_y = 0

                # for the case of 11 horizontal cells, we know exactly who the nbrs are of each cell
                nbr_cell_IDs = []
                if agent.ID == 0:    # left end
                    nbr_cell_IDs = [1]
                elif agent.ID == 10:    # right end
                    nbr_cell_IDs = [9]
                else:
                    nbr_cell_IDs = [agent.ID - 1, agent.ID + 1]

                for agent2 in self.agents:
                    # print(agent.ID, agent2.ID)
                    if agent2.ID in nbr_cell_IDs:
                        xdel = agent.x - agent2.x   # "displacement" vector in C++
                        # ydel = agent.y - agent2.y
                        # ydel = 0.0
                        # dist = math.sqrt(xdel*xdel + ydel*ydel)
                        dist = math.sqrt(xdel*xdel)
                        # distance = std::max(sqrt(distance), 0.00001); 
                        # dist = max(dist, 0.00001)

                        # //Repulsive
                        # double R = phenotype.geometry.radius+ (*other_agent).phenotype.geometry.radius; 

                        # agent2_r = math.sqrt(agent2.area/math.pi)
                        agent2_r = agent_r
                        # RMAD = 1.0   # Relative max adhesion distance
                        # # if dist < RMAD * (agent_r + agent2_r):    # RMAD = 1.25 leads to weird results
                        R = agent_r + agent2_r
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
                        temp_r *= cell_cell_repulsion_strength 

                        # skip over adhesion, there is none

                        # if( fabs(temp_r) < 1e-16 )
                        # { return; }
                        temp_r /= dist

                        # in add_potentials(), we also update the cell's velocity!
                        # axpy( &velocity , temp_r , displacement );    # velocity[i] += displacement[i] * temp_r;
                        # velocity[i] += displacement[i] * temp_r; 
                        # axpy( &velocity , temp_r , displacement ); 

                        agent.vel_x += xdel * temp_r

                        # agent.vel_y += ydel * temp_r
                        agent.vel_y = 0


            # 3) Update each agent's position with its velocity
            dt_mechanics = 0.1
            # dt_mechanics = 1.0  # results in bizarre position updates
            for idx_mech in range(1):  # 10 ?
                self.update_position(dt_mechanics)


        # print("type(self.agents))= ",type(self.agents))
        # print("dir(self.agents[0]))= ",dir(self.agents[0]))
        # tissue_width= self.agents[10].x - self.agents[0].x 
        # if tissue_width >= 90.:
        #     print(f"tissue width= {tissue_width} at time (#steps)={self.time}")
            # exit()

        self.time += 1   # confusing, just "step #"


    def run(self, steps: int) -> None:
        """
        Run the simulation for a given number of steps or until max_agents is reached.

        Args:
            steps:      Maximum number of time steps to run.
        """

        for _ in range(steps):
            self.step()


    def precompute(self, steps: int) -> None:
        """
        Run the full simulation up-front and store every frame's agent list.
        Required before calling interactive_viewer().

        Args:
            steps:      Number of time steps to pre-compute.
            max_agents: Stop early if population exceeds this.
        """
        import copy
        print(f"Pre-computing {steps} steps...", end=" ", flush=True)
        # Frame 0 = initial state (before any stepping)
        self._frames: list[list[Agent]] = [copy.deepcopy(self.agents)]

        # file_out = f'py_cells11_init.csv'
        # print("--> ",file_out)
        # with open(file_out, "w", newline="") as file:
        #     writer = csv.writer(file)

        #     writer.writerow(['x_pos','y_pos','radius_i','ID'])
        #     for agent in self.agents:
        #         # writer.writerow([x_pos[jdx],y_pos[jdx],radius_i[jdx],f_i[jdx],a_i[jdx]])
        #         radius = math.sqrt(agent.area / math.pi)
        #         writer.writerow([agent.x, agent.y, radius, agent.ID])

        for _ in range(steps):
            self.step()
            self._frames.append(copy.deepcopy(self.agents))

            tissue_width_90pct = self.agents[10].x - self.agents[0].x 
            if tissue_width_90pct >= 90.:
                print(f"\n\ntissue_width_90pct= {tissue_width_90pct} at time= {self.time}")
                print(f"   --> cell cycle duration")
                cell_cycle_rate = 1.0/self.time
                print(f"   --> cell cycle rate (1/duration)= {cell_cycle_rate}")
                print(f"   --> growth rate of cell (circular area): (2*A - A)*cycle rate= 78.5*{cell_cycle_rate}= {78.5*cell_cycle_rate}")
                break

        print(f"done. {len(self._frames)} frames stored.")

        # save final results in .csv for post-processing
        # file_out = f'py_cells11.csv'
        # print("--> ",file_out)
        # with open(file_out, "w", newline="") as file:
        #     writer = csv.writer(file)

        #     writer.writerow(['x_pos','y_pos','radius_i','ID'])
        #     for agent in self.agents:
        #         # writer.writerow([x_pos[jdx],y_pos[jdx],radius_i[jdx],f_i[jdx],a_i[jdx]])
        #         radius = math.sqrt(agent.area / math.pi)
        #         writer.writerow([agent.x, agent.y, radius, agent.ID])


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
        ax_space.set_xlim(-win_size, win_size)
        ax_space.set_ylim(-20, 20)

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
            for agent in agent_list:
                # color = cmap(agent.generation / max(max_gen, 10))
                ax_space.add_patch(patches.Circle(
                    (agent.x, agent.y), 
                    radius=math.sqrt(agent.area / math.pi),
                    facecolor='gray', alpha=0.72,
                    linewidth=1.0, edgecolor="white",
                ))

            title_text.set_text(
                f"t = {idx}   |   width = ?"  
            )

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
        n_seed= 11,              # Start with n agents
        initial_area= 78.54,    #  area = pi * r^2 = pi * 25 = 78.54
    )

    sim.load()

    # 1. Pre-compute all frames (enables backward scrubbing)
    # sim.precompute(steps=50)  # rwh
    sim.precompute(steps=max_steps)  # rwh

    # print("ID for agent[0]= ",sim.agents[0].ID)
    # print("ID for agent[10]= ",sim.agents[10].ID)
    print("tissue width= ",sim.agents[10].x - sim.agents[0].x)

    # 2. Open the interactive widget viewer
    sim.interactive_viewer(
        title="Relax 11 compressed cells",
        interval=1,   # rwh: ms between auto-play frames
    )
