import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import time

# Enhanced function evaluation with caching
class CachedFunction:
    def __init__(self, func, cache_size=10000):
        self.func = func
        self.cache = {}
        self.cache_size = cache_size

    def __call__(self, x):
        key = tuple(round(xi, 8) for xi in x)
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = self.func(x)
        return self.cache[key]


class PlotCache:
    def __init__(self):
        self.grid_cache = {}

    def get_grid(self, func_name, domain_min, domain_max, resolution=50):
        cache_key = (func_name, domain_min, domain_max, resolution)
        if cache_key not in self.grid_cache:
            x = np.linspace(domain_min, domain_max, resolution)
            y = np.linspace(domain_min, domain_max, resolution)
            X, Y = np.meshgrid(x, y)
            func = TEST_FUNCTIONS[func_name]
            Z = np.array([func([X[i, j], Y[i, j]]) for i in range(resolution) for j in range(resolution)]).reshape(
                resolution, resolution)
            self.grid_cache[cache_key] = (X, Y, Z)
        return self.grid_cache[cache_key]


# TestFunction
def sphere(x): return np.sum(np.array(x) ** 2)
def rosenbrock(x): a,b=1,100; x1,x2=x[0],x[1]; return (a-x1)**2 + b*(x2-x1**2)**2
def rastrigin(x): A=10; n=len(x); return A*n+sum([(xi**2-A*np.cos(2*np.pi*xi)) for xi in x])
def ackley(x): a,b,c=20,0.2,2*np.pi; n=len(x); s1=-a*np.exp(-b*np.sqrt(sum(xi**2 for xi in x)/n)); s2=-np.exp(sum(np.cos(c*xi) for xi in x)/n); return s1+s2+a+np.exp(1)
def himmelblau(x): x1,x2=x[0],x[1]; return (x1**2+x2-11)**2+(x1+x2**2-7)**2
def easom(x): x1,x2=x[0],x[1]; return -np.cos(x1)*np.cos(x2)*np.exp(-((x1-np.pi)**2+(x2-np.pi)**2))
def testing_function(x): x1,x2=x[0],x[1]; return -((x1)**2+(x2)**2)

TEST_FUNCTIONS = {'Sphere': sphere, 'Rosenbrock': rosenbrock, 'Rastrigin': rastrigin, 'Ackley': ackley, 'Himmelblau': himmelblau, 'Easom': easom, 'Testing Function': testing_function}
GLOBAL_OPTIMA = {'Sphere': {'position': [0,0], 'value': 0}, 'Rosenbrock': {'position': [1,1], 'value': 0}, 'Rastrigin': {'position': [0,0], 'value': 0}, 'Ackley': {'position': [0,0], 'value': 0}, 'Himmelblau': {'position': [3,2], 'value': 0}, 'Easom': {'position': [np.pi,np.pi], 'value': -1}, 'Testing Function': {'position': [3.14,2.72], 'value': testing_function([3.14,2.72])}}
FUNCTION_DOMAINS = {'Sphere': (-5.0, 5.0), 'Rosenbrock': (-5.0, 5.0), 'Rastrigin': (-5.12, 5.12), 'Ackley': (-5.0, 5.0), 'Himmelblau': (-6.0, 6.0), 'Easom': (0.0, 2 * np.pi), 'Testing Function': (-5.0, 10.0)}

# --- PSO Algorithm ---
def precompute_pso(obj_func, n_iterations=100, n_particles=30, domain_min=-5.0, domain_max=5.0):
    positions=np.random.uniform(domain_min,domain_max,(n_particles,2)); velocities=np.random.uniform(-1,1,(n_particles,2))
    pbest_values=np.array([obj_func(p) for p in positions]); pbest_positions=positions.copy()
    gbest_idx=np.argmin(pbest_values); gbest_position=pbest_positions[gbest_idx].copy(); gbest_value=pbest_values[gbest_idx]
    #Parameters
    w = 0.9
    c1 = 1.5
    c2 = 1.5
    iterations_data=[{'positions':positions.copy(),'velocities':velocities.copy(),'pbest_positions':pbest_positions.copy(),'pbest_values':pbest_values.copy(),'gbest_position':gbest_position.copy(),'gbest_value':gbest_value}]
    for _ in range(n_iterations):
        r1,r2=np.random.random((n_particles,2)),np.random.random((n_particles,2))
        velocities = w*velocities + c1*r1*(pbest_positions-positions) + c2*r2*(gbest_position-positions)
        velocities = np.clip(velocities,-1,1)
        positions=np.clip(positions+velocities,domain_min,domain_max)
        current_values=np.array([obj_func(p) for p in positions])
        improved=current_values<pbest_values; pbest_values[improved]=current_values[improved]; pbest_positions[improved]=positions[improved].copy()
        min_idx=np.argmin(current_values)
        if current_values[min_idx]<gbest_value: gbest_position=positions[min_idx].copy(); gbest_value=current_values[min_idx]
        iterations_data.append({'positions':positions.copy(),'velocities':velocities.copy(),'pbest_positions':pbest_positions.copy(),'pbest_values':pbest_values.copy(),'gbest_position':gbest_position.copy(),'gbest_value':gbest_value})
    return iterations_data

class PSOViewer:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced PSO Visualization")
        master.geometry("1200x750")
        master.minsize(900, 600)

        # Define background colors as instance variables
        self.main_bg = "#f0f0f0"
        self.control_bg = "#e8e8e8"
        self.plot_bg = "#ffffff"
        self.button_active_bg = '#c0c0c0' # Slightly darker active bg

        master.configure(bg=self.main_bg)

        self._configure_styles() # Configure styles using instance colors

        # --- Core Parameters & State ---
        self.current_iter = 0; self.is_playing = False; self.anim_id = None
        self.swarm_data = []; self.fig_width, self.fig_height, self.fig_dpi = 12, 5, 100
        self.RES = 50
        self.plot_cache = PlotCache()


        self.n_iterations_var=tk.IntVar(value=100); self.n_particles_var=tk.IntVar(value=30)
        self.anim_delay_var=tk.IntVar(value=100); self.func_var=tk.StringVar(value='Sphere')
        self.domain_label_var=tk.StringVar(value=""); self.pso_status_var=tk.StringVar(value="Initializing...")
        self.gbest_pos_var=tk.StringVar(value="N/A"); self.gbest_val_var=tk.StringVar(value="N/A")
        self.true_opt_pos_var=tk.StringVar(value="N/A"); self.true_opt_val_var=tk.StringVar(value="N/A")
        self.pos_error_var=tk.StringVar(value="N/A"); self.val_error_var=tk.StringVar(value="N/A")

        # Set initial function info
        self.obj_func_name=self.func_var.get(); self.obj_func=TEST_FUNCTIONS[self.obj_func_name]
        # Wrap function with caching if not already cached
        if not isinstance(self.obj_func, CachedFunction):
            self.obj_func = CachedFunction(self.obj_func)
        self.domain_min,self.domain_max=FUNCTION_DOMAINS[self.obj_func_name]

        self._create_ui(master)
        self._create_figure_and_axes()
        self._create_canvas()
        self._update_domain_label()
        self.recalculate_pso()
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.pso_status_var.set("Ready")


    def _configure_styles(self):
        style = ttk.Style()
        try: style.theme_use('clam') # Clam or aqua(mac) often look better
        except tk.TclError: print("Clam theme not found, using default."); style.theme_use('default')

        # Define fonts
        button_font = ("Segoe UI", 10, "bold") # Bolder buttons
        label_font = ("Segoe UI", 9)
        header_font = ("Segoe UI", 11, "bold")
        result_font = ("Segoe UI", 10) # Slightly larger result text
        result_font_bold = (result_font[0], result_font[1], "bold")

        style.configure("TFrame", background=self.main_bg)
        style.configure("Control.TFrame", background=self.control_bg, borderwidth=1, relief=tk.FLAT)
        style.configure("PlotArea.TFrame", background=self.plot_bg, borderwidth=1, relief=tk.SUNKEN)

        style.configure("Control.TButton", font=button_font, padding=(8, 6), borderwidth=1, relief=tk.RAISED) # More padding
        style.map("Control.TButton",
                  background=[('active', self.button_active_bg)],
                  relief=[('pressed', tk.SUNKEN), ('!pressed', tk.RAISED)]) # Ensure it raises again


        style.configure("Run.TButton", foreground="white", background="#28a745") # Green
        style.configure("Stop.TButton", foreground="white", background="#dc3545") # Red
        style.configure("Recalculate.TButton", foreground="white", background="#007bff") # Blue
        style.configure("Quit.TButton", foreground="white", background="#6c757d") # Grey


        style.configure("Run.TButton", font=button_font, padding=(8, 6), borderwidth=1, relief=tk.RAISED)
        style.configure("Stop.TButton", font=button_font, padding=(8, 6), borderwidth=1, relief=tk.RAISED)
        style.configure("Recalculate.TButton", font=button_font, padding=(8, 6), borderwidth=1, relief=tk.RAISED)
        style.configure("Quit.TButton", font=button_font, padding=(8, 6), borderwidth=1, relief=tk.RAISED)

        # Mapping for active/pressed states
        style.map("Run.TButton", background=[('active', '#218838')], relief=[('pressed', tk.SUNKEN), ('!pressed', tk.RAISED)])
        style.map("Stop.TButton", background=[('active', '#c82333')], relief=[('pressed', tk.SUNKEN), ('!pressed', tk.RAISED)])
        style.map("Recalculate.TButton", background=[('active', '#0056b3')], relief=[('pressed', tk.SUNKEN), ('!pressed', tk.RAISED)])
        style.map("Quit.TButton", background=[('active', '#5a6268')], relief=[('pressed', tk.SUNKEN), ('!pressed', tk.RAISED)])



        style.configure("TLabel", background=self.control_bg, font=label_font) # Default label in control panel
        style.configure("Header.TLabel", font=header_font, background=self.control_bg)
        style.configure("Status.TLabel", font=label_font, background=self.control_bg, foreground="#555555")
        style.configure("ControlPanelValue.TLabel", font=result_font_bold, background=self.control_bg) # Values in control panel
        style.configure("ResultValue.TLabel", font=result_font_bold, background=self.main_bg) # Values in results area
        style.configure("ResultLabel.TLabel", font=result_font, background=self.main_bg) # Non-bold labels in results area


        style.configure("TLabelframe", background=self.control_bg)
        style.configure("TLabelframe.Label", background=self.control_bg, font=("Segoe UI", 10, "bold"), padding=(5, 3))
        style.configure("TScale", background=self.control_bg, troughcolor="#b0b0b0", sliderrelief=tk.FLAT, borderwidth=1) # Slightly darker trough, added border
        style.configure("TCombobox", font=("Segoe UI", 9))


    def _create_ui(self, master):

        self.control_panel=ttk.Frame(master,width=260,style="Control.TFrame",padding=10); self.control_panel.pack(side=tk.LEFT,fill=tk.Y,padx=(5,2),pady=5); self.control_panel.pack_propagate(False) # Wider panel
        self.main_area=ttk.Frame(master,padding=(2,5,5,5)); self.main_area.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)


        ttk.Label(self.control_panel,text="PSO Configuration",style="Header.TLabel").pack(pady=(0,10),anchor=tk.W)
        func_frame=ttk.Frame(self.control_panel, style="Control.TFrame"); ttk.Label(func_frame,text="Function:").pack(side=tk.LEFT,anchor=tk.W, padx=(0,5))
        self.func_menu=ttk.Combobox(func_frame,textvariable=self.func_var,values=list(TEST_FUNCTIONS.keys()),state="readonly",width=18,font=("Segoe UI",9)); self.func_menu.pack(side=tk.LEFT,padx=0); self.func_menu.bind("<<ComboboxSelected>>",self.on_function_change); func_frame.pack(fill=tk.X,pady=4)
        ttk.Label(self.control_panel,text="Domain:",anchor=tk.W).pack(fill=tk.X, padx=(0,0))
        self.domain_label_widget=ttk.Label(self.control_panel,textvariable=self.domain_label_var,style="ControlPanelValue.TLabel",anchor=tk.W); self.domain_label_widget.pack(fill=tk.X,pady=(0,8))

        param_frame=ttk.LabelFrame(self.control_panel,text="Parameters"); param_frame.pack(fill=tk.X,pady=4)
        ttk.Label(param_frame,text="N Particles:").grid(row=0,column=0,padx=5,pady=4,sticky=tk.W)
        self.n_particles_scale=ttk.Scale(param_frame,from_=5,to=200,orient=tk.HORIZONTAL,variable=self.n_particles_var,length=110,command=lambda v:self.n_particles_var.set(int(float(v)))); self.n_particles_scale.grid(row=0,column=1,padx=5,pady=4,sticky=tk.EW)
        self.n_particles_label=ttk.Label(param_frame,textvariable=self.n_particles_var,width=4, style="ControlPanelValue.TLabel"); self.n_particles_label.grid(row=0,column=2,padx=5,pady=4) # Use value style
        ttk.Label(param_frame,text="N Iterations:").grid(row=1,column=0,padx=5,pady=4,sticky=tk.W)
        self.n_iterations_scale=ttk.Scale(param_frame,from_=10,to=500,orient=tk.HORIZONTAL,variable=self.n_iterations_var,length=110,command=lambda v:self.n_iterations_var.set(int(float(v)))); self.n_iterations_scale.grid(row=1,column=1,padx=5,pady=4,sticky=tk.EW)
        self.n_iterations_label=ttk.Label(param_frame,textvariable=self.n_iterations_var,width=4, style="ControlPanelValue.TLabel"); self.n_iterations_label.grid(row=1,column=2,padx=5,pady=4); param_frame.columnconfigure(1,weight=1)


        self.recalc_button=ttk.Button(self.control_panel,text="Recalculate PSO",command=self.recalculate_pso, style="Recalculate.TButton"); self.recalc_button.pack(pady=(15,8),fill=tk.X)

        anim_frame=ttk.LabelFrame(self.control_panel,text="Animation"); anim_frame.pack(fill=tk.X,pady=4)
        self.play_button=ttk.Button(anim_frame,text="Play",command=self.toggle_play,width=10, style="Run.TButton"); self.play_button.grid(row=0,column=0,columnspan=3,padx=5,pady=5,sticky=tk.EW)
        ttk.Label(anim_frame,text="Delay (ms):").grid(row=1,column=0,padx=5,pady=4,sticky=tk.W)
        self.anim_delay_scale=ttk.Scale(anim_frame,from_=10,to=500,orient=tk.HORIZONTAL,variable=self.anim_delay_var,length=110,command=lambda v:self.anim_delay_var.set(int(float(v)))); self.anim_delay_scale.grid(row=1,column=1,padx=5,pady=4,sticky=tk.EW)
        self.anim_delay_label=ttk.Label(anim_frame,textvariable=self.anim_delay_var,width=4, style="ControlPanelValue.TLabel"); self.anim_delay_label.grid(row=1,column=2,padx=5,pady=4); anim_frame.columnconfigure(1,weight=1)

        ttk.Label(self.control_panel,text="Status:",anchor=tk.W).pack(pady=(15,0),fill=tk.X)
        self.status_label=ttk.Label(self.control_panel,textvariable=self.pso_status_var,style="Status.TLabel",anchor=tk.W); self.status_label.pack(fill=tk.X)
        ttk.Separator(self.control_panel).pack(fill=tk.X,pady=10)
        self.quit_button=ttk.Button(self.control_panel,text="Quit",command=self.on_closing, style="Quit.TButton"); self.quit_button.pack(pady=5,fill=tk.X)

        # --- Main Area UI ---
        top_frame=ttk.Frame(self.main_area); top_frame.pack(side=tk.TOP,fill=tk.X,pady=(0,5))
        self.iter_scale=ttk.Scale(top_frame,from_=0,to=self.n_iterations_var.get(),orient=tk.HORIZONTAL,command=self.on_slider_change, length=400); self.iter_scale.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=5,pady=5) # Increase length
        self.iter_label=ttk.Label(top_frame,text="Iteration: 0 / 0",font=("Segoe UI",10,"bold"),width=16,anchor=tk.E, background=self.main_bg); self.iter_label.pack(side=tk.RIGHT,padx=(0,5)) # Slightly larger iteration label
        self.canvas_frame=ttk.Frame(self.main_area,style="PlotArea.TFrame"); self.canvas_frame.pack(fill=tk.BOTH,expand=True)

        # --- Results Area (Increased padding and font size) ---
        results_frame=ttk.LabelFrame(self.main_area,text="Results", padding=(10, 5)); results_frame.pack(fill=tk.X,pady=(5,0)) # Add padding to frame
        # Use specific style for labels here
        pady_results = 5 # Increase vertical padding
        ttk.Label(results_frame, text="PSO Best Pos:", style="ResultLabel.TLabel").grid(row=0, column=0, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.gbest_pos_var, style="ResultValue.TLabel").grid(row=0, column=1, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, text="PSO Value:", style="ResultLabel.TLabel").grid(row=0, column=2, padx=5,pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.gbest_val_var, style="ResultValue.TLabel").grid(row=0, column=3, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, text="Pos Error:", style="ResultLabel.TLabel").grid(row=0, column=4, padx=(20, 5), pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.pos_error_var, style="ResultValue.TLabel").grid(row=0, column=5, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, text="True Optimum:", style="ResultLabel.TLabel").grid(row=1, column=0, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.true_opt_pos_var, style="ResultValue.TLabel").grid(row=1, column=1, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, text="True Value:", style="ResultLabel.TLabel").grid(row=1, column=2, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.true_opt_val_var, style="ResultValue.TLabel").grid(row=1, column=3, padx=5, pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, text="Val Error:", style="ResultLabel.TLabel").grid(row=1, column=4, padx=(20, 5), pady=pady_results, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.val_error_var, style="ResultValue.TLabel").grid(row=1, column=5, padx=5, pady=pady_results, sticky=tk.W)
        results_frame.columnconfigure(1, weight=1); results_frame.columnconfigure(3, weight=1); results_frame.columnconfigure(5, weight=1)

        # --- Define Controls List ---
        self.interactive_controls = [
            self.func_menu, self.n_particles_scale, self.n_iterations_scale,
            self.recalc_button, self.play_button, self.anim_delay_scale,
            self.iter_scale, self.quit_button ]


    def _create_figure_and_axes(self):
        plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams.update({'font.size':9})
        self.fig = plt.figure(figsize=(self.fig_width,self.fig_height),dpi=self.fig_dpi)
        left_pos, right_pos, cbar_pos = [0.06,0.1,0.37,0.8], [0.57,0.1,0.37,0.8], [0.44,0.1,0.015,0.8]
        self.ax2d=self.fig.add_axes(left_pos); self.ax3d=self.fig.add_axes(right_pos,projection='3d'); self.cbar_ax=self.fig.add_axes(cbar_pos)
        self.ax2d_pos, self.ax3d_pos = self.ax2d.get_position(), self.ax3d.get_position()
        self._nullify_artists() # Ensure artists start as None


    def _create_canvas(self):
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.canvas_frame); self.canvas_widget=self.canvas.get_tk_widget(); self.canvas_widget.pack(fill=tk.BOTH,expand=True); self.canvas.draw()


    def _update_domain_label(self):
        self.domain_label_var.set(f"[{self.domain_min:.2f}, {self.domain_max:.2f}]")


    def _toggle_controls(self, state=tk.NORMAL):
        actual_state = state if state == tk.NORMAL else 'disabled'
        for widget in self.interactive_controls:
            if widget:
                try: widget.configure(state=actual_state)
                except tk.TclError as e:
                    try:
                        if isinstance(widget, ttk.Combobox): widget.state(['readonly'] if state == tk.NORMAL else ['disabled'])
                        # Apply specific state handling for styled buttons too
                        elif isinstance(widget, ttk.Button) or isinstance(widget, ttk.Scale):
                             widget.state(['!disabled'] if state == tk.NORMAL else ['disabled'])
                    except tk.TclError as e2: print(f"Warn: configure&state failed {widget}: {e},{e2}")


    def recalculate_pso(self):
        if self.is_playing: self.toggle_play()
        self._toggle_controls(tk.DISABLED); self.pso_status_var.set("Calculating PSO...")
        self.master.update_idletasks()
        n_iter=self.n_iterations_var.get(); n_part=self.n_particles_var.get()
        self.obj_func_name=self.func_var.get(); self.obj_func=TEST_FUNCTIONS[self.obj_func_name]
        self.domain_min,self.domain_max=FUNCTION_DOMAINS[self.obj_func_name]; self._update_domain_label()
        self.swarm_data=[]; pso_success=False
        if n_iter<1 or n_part<1: messagebox.showerror("Error","N Iter/Particles must be >=1"); self.pso_status_var.set("Err:Invalid Params")
        else:
            try: self.swarm_data=precompute_pso(self.obj_func,n_iter,n_part,self.domain_min,self.domain_max); pso_success=True; self.pso_status_var.set("Calculating... Plotting..."); self.master.update_idletasks()
            except Exception as e: messagebox.showerror("PSO Error",f"PSO calculation error:\n{e}"); print(f"PSO Err:{e}"); traceback.print_exc(); self.pso_status_var.set("Err:PSO Failed"); self.swarm_data=[]
        self.current_iter=0; iter_count=len(self.swarm_data); slider_max=iter_count-1 if iter_count>0 else 0
        self.iter_scale.config(to=slider_max); self.iter_scale.set(0)
        self.iter_label.config(text=f"Iteration: 0 / {slider_max}")
        if pso_success and self.swarm_data:
            try:
                self.setup_plots(); initial_data=self.swarm_data[0]; self.update_results_display(initial_data['gbest_position'],initial_data['gbest_value'])
                self.pso_status_var.set("Ready")
            except Exception as plot_e: messagebox.showerror("Plot Error",f"Plot setup error:\n{plot_e}"); print(f"Plot Err:{plot_e}"); traceback.print_exc(); self.pso_status_var.set("Err:Plot Failed!"); self._clear_plots_on_error()
        elif not pso_success: self.pso_status_var.set("Error state - Recalculate"); self._clear_plots_on_error()
        self._toggle_controls(tk.NORMAL)


    def _clear_plots_on_error(self):
        try:
             print("Attempting to clear plots due to error...")
             self.ax2d.clear(); self.ax3d.clear()
             if self.colorbar: self.colorbar.remove(); self.colorbar = None
             self.update_results_display(np.array([np.nan, np.nan]), np.nan)
             self._nullify_artists()
             self.canvas.draw_idle()
        except Exception as clear_e: print(f"Error during _clear_plots_on_error: {clear_e}")


    def _nullify_artists(self):
         self.contour, self.colorbar, self.surface = None, None, None
         self.scatter2d, self.gbest2d = None, None
         self.scatter3d, self.gbest3d = None, None
         self.true_opt2d, self.true_opt3d = None, None


    def on_function_change(self, event=None):
        self.obj_func_name=self.func_var.get(); self.obj_func=TEST_FUNCTIONS[self.obj_func_name]
        self.domain_min,self.domain_max=FUNCTION_DOMAINS[self.obj_func_name]; self._update_domain_label()
        self.plot_cache.grid_cache.clear()
        self.pso_status_var.set(f"Func->'{self.obj_func_name}'. Press Recalculate.")


    def setup_plots(self):
        """Clears axes, recreates static background plots, and
           initializes dynamic elements with data from iteration 0."""

        if not self.swarm_data:
            print("Error: setup_plots called with no swarm data.")
            self._clear_plots_on_error() # Attempt cleanup
            return

        # Get initial data safely
        initial_data = self.swarm_data[0]
        positions = initial_data['positions']
        velocities = initial_data['velocities']
        gbest_position = initial_data['gbest_position']
        gbest_value = initial_data['gbest_value']
        try:
            z_positions = np.array([self.obj_func(p) for p in positions])
        except Exception as e:
            print(f"Error calculating initial Z positions in setup: {e}")
            z_positions = np.zeros(len(positions)) # Fallback

        # --- Prepare background grid ---
        try:
            X, Y, Z = self.plot_cache.get_grid(self.obj_func_name, self.domain_min, self.domain_max, self.RES)

        except Exception as e:
             print(f"Error calculating Z grid in setup: {e}")
             messagebox.showerror("Grid Error", f"Could not evaluate function for plotting grid:\n{e}")
             Z = np.zeros((self.RES, self.RES)) # Fallback grid



        self.ax2d.clear()
        self.ax3d.clear()


        if self.colorbar:
            try:
                self.colorbar.remove()
            except Exception as e:
                print(f"Minor err removing cbar: {e}")

            self.colorbar = None

        self._nullify_artists()


        self.ax2d.set_xlabel('x'); self.ax2d.set_ylabel('y'); self.ax2d.set_title(f'{self.obj_func_name}(Contour)',fontsize=10,fontweight='bold'); self.ax2d.set_xlim([self.domain_min,self.domain_max]); self.ax2d.set_ylim([self.domain_min,self.domain_max]); self.ax2d.set_aspect('equal',adjustable='box')
        self.ax3d.set_xlabel('x'); self.ax3d.set_ylabel('y'); self.ax3d.set_zlabel('f(x,y)'); self.ax3d.set_title(f'{self.obj_func_name}(Surface)',fontsize=10,fontweight='bold'); self.ax3d.set_xlim([self.domain_min,self.domain_max]); self.ax3d.set_ylim([self.domain_min,self.domain_max])

        try:
            self.contour = self.ax2d.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
            if hasattr(self,'cbar_ax') and self.cbar_ax is not None and self.contour:
                 self.colorbar = self.fig.colorbar(self.contour, cax=self.cbar_ax)
                 self.colorbar.ax.tick_params(labelsize=8)
            else:
                 print("Warn: cbar_ax not found or contour failed.")
        except Exception as e:
            print(f"Err contour/cbar: {e}")

            if self.colorbar:
                try:
                    self.colorbar.remove()
                except Exception as e_rem:
                    print(f"Err removing potentially failed colorbar: {e_rem}")
                self.colorbar = None

        try:
             self.surface = self.ax3d.plot_surface(
                 X, Y, Z, cmap='viridis', alpha=0.7, rstride=1, cstride=1, linewidth=0, antialiased=True
             )
        except Exception as e:
             print(f"Err surface: {e}")



        try:
            self.scatter2d = self.ax2d.scatter(positions[:, 0], positions[:, 1], c='red', s=25, alpha=0.9, zorder=5, label='Particles')
            self.gbest2d = self.ax2d.scatter([gbest_position[0]], [gbest_position[1]], c='gold', s=150, marker='*', edgecolors='black', linewidths=0.5, zorder=6, label='PSO Best')
            self.quiver2d = self.ax2d.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], color='black', scale=15, width=0.004, zorder=4)

            self.scatter3d = self.ax3d.scatter(positions[:, 0], positions[:, 1], z_positions, c='red', s=25, alpha=0.8)
            self.gbest3d = self.ax3d.scatter([gbest_position[0]], [gbest_position[1]], [gbest_value], c='gold', s=150, marker='*', edgecolors='black', linewidths=0.5)
            self.quiver3d = self.ax3d.quiver(positions[:, 0], positions[:, 1], z_positions, velocities[:, 0], velocities[:, 1], np.zeros_like(velocities[:, 0]), color='black', length=0.3, normalize=False, arrow_length_ratio=0.3, linewidth=0.8)

        except Exception as e:
             print(f"Error creating initial dynamic plot elements: {e}")
             messagebox.showerror("Plot Init Error", f"Failed to create initial particle plots:\n{e}")



        if self.obj_func_name in GLOBAL_OPTIMA:
            true_opt = GLOBAL_OPTIMA[self.obj_func_name]['position']
            true_val = GLOBAL_OPTIMA[self.obj_func_name]['value']
            if (self.domain_min <= true_opt[0] <= self.domain_max and
                self.domain_min <= true_opt[1] <= self.domain_max):
                try:
                    self.true_opt2d = self.ax2d.scatter([true_opt[0]], [true_opt[1]], c='lime', s=100, marker='X', edgecolors='black', linewidths=1, zorder=7, label='True Optimum')
                    self.true_opt3d = self.ax3d.scatter([true_opt[0]], [true_opt[1]], [true_val], c='lime', s=100, marker='X', edgecolors='black', linewidths=1)
                except Exception as e:
                     print(f"Err true opt: {e}")



        handles, labels = self.ax2d.get_legend_handles_labels()
        if handles:
            self.ax2d.legend(loc='upper right', fontsize='small', frameon=True, framealpha=0.8)


        try:
            self.canvas.draw_idle()
        except Exception as draw_e:
            print(f"Error during canvas draw at end of setup_plots: {draw_e}")
            traceback.print_exc()

    def update_plot(self, iteration):
        if not self.swarm_data or not(0<=iteration<len(self.swarm_data)):return
        self.current_iter = iteration; data=self.swarm_data[iteration]; positions=data['positions']; velocities=data['velocities']; gbest_position=data['gbest_position']; gbest_value=data['gbest_value']
        self.iter_label.config(text=f"Iter: {iteration} / {len(self.swarm_data)-1}")
        if not all([self.scatter2d, self.gbest2d, self.quiver2d, self.scatter3d, self.gbest3d]): print("Warn:Core plot elem missing in update"); return
        self.update_results_display(gbest_position, gbest_value)
        self.scatter2d.set_offsets(positions); self.gbest2d.set_offsets([gbest_position]); self.quiver2d.set_offsets(positions); self.quiver2d.set_UVC(velocities[:,0],velocities[:,1])
        try: z_positions=np.array([self.obj_func(p) for p in positions]); z_gbest=gbest_value
        except Exception as e: print(f"Err Z upd:{e}"); z_positions=np.zeros(len(positions)); z_gbest=0
        if hasattr(self.scatter3d,'_offsets3d'): self.scatter3d._offsets3d=(positions[:,0],positions[:,1],z_positions)
        if hasattr(self.gbest3d,'_offsets3d'): self.gbest3d._offsets3d=([gbest_position[0]],[gbest_position[1]],[z_gbest])
        if self.quiver3d:
             try: self.quiver3d.remove()
             except ValueError: pass # Ignore if already gone
             self.quiver3d=None # Ensure None after attempt
        # Avoid error if ax3d cleared between remove and recreate
        if self.ax3d.axes: # Check if axes still valid
             try:
                 self.quiver3d = self.ax3d.quiver(positions[:,0],positions[:,1],z_positions, velocities[:,0],velocities[:,1],np.zeros_like(velocities[:,0]), color='black',length=0.3,normalize=False,arrow_length_ratio=0.3,linewidth=0.8)
             except Exception as e: print(f"Error recreating quiver: {e}")
        if int(self.iter_scale.get())!=iteration: self.iter_scale.set(iteration)
        try: self.canvas.draw_idle()
        except Exception as draw_e: print(f"Err draw_idle upd:{draw_e}"); traceback.print_exc()


    def update_results_display(self, gbest_position, gbest_value):
        pos_str,val_str,true_pos_str,true_val_str,pos_err_str,val_err_str="N/A","N/A","N/A","N/A","N/A","N/A"
        if gbest_position is not None and not any(np.isnan(gbest_position)) and not np.isnan(gbest_value):
             pos_str=f"{gbest_position[0]:.4f},{gbest_position[1]:.4f}"; val_str=f"{gbest_value:.6f}"
             if self.obj_func_name in GLOBAL_OPTIMA:
                 true_opt=GLOBAL_OPTIMA[self.obj_func_name]; true_pos_str=f"{true_opt['position'][0]:.4f},{true_opt['position'][1]:.4f}"; true_val_str=f"{true_opt['value']:.6f}"
                 try: pos_error=np.linalg.norm(np.array(gbest_position)-np.array(true_opt['position'])); val_error=abs(gbest_value-true_opt['value']); pos_err_str=f"{pos_error:.6f}"; val_err_str=f"{val_error:.6f}"
                 except Exception: pass
        self.gbest_pos_var.set(pos_str); self.gbest_val_var.set(val_str); self.true_opt_pos_var.set(true_pos_str); self.true_opt_val_var.set(true_val_str); self.pos_error_var.set(pos_err_str); self.val_error_var.set(val_err_str)


    def on_slider_change(self, value):
        iter_val = int(float(value))
        if self.swarm_data and iter_val != self.current_iter and not self.is_playing: self.update_plot(iter_val)


    def toggle_play(self):
        if self.is_playing:
            self.is_playing=False; self.play_button.config(text="Play",style="Run.TButton")
            if self.anim_id is not None: self.master.after_cancel(self.anim_id); self.anim_id = None
            self.pso_status_var.set("Paused"); self._toggle_controls(tk.NORMAL)
        else:
            if not self.swarm_data: messagebox.showwarning("No Data","Please calculate PSO first."); return
            self.is_playing=True; self.play_button.config(text="Pause",style="Stop.TButton")
            self.pso_status_var.set("Playing...")
            # More refined disabling: Keep Quit, Anim Delay enabled
            self.recalc_button.config(state=tk.DISABLED); self.n_particles_scale.config(state='disabled')
            self.n_iterations_scale.config(state='disabled'); self.iter_scale.config(state='disabled')
            self.func_menu.configure(state='disabled')

            if self.current_iter>=len(self.swarm_data)-1: self.current_iter = -1
            self.animate()

    def animate(self):
        if not self.is_playing or not self.swarm_data: return

        start_time = time.time()
        next_iter = self.current_iter + 1

        if next_iter >= len(self.swarm_data):
            self.toggle_play()
            self.pso_status_var.set("Finished")
            return

        self.update_plot(next_iter)

        # Adaptive delay to maintain smooth framerate
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        target_delay = self.anim_delay_var.get()
        actual_delay = max(10, target_delay - int(processing_time))  # Minimum 10ms

        self.anim_id = self.master.after(actual_delay, self.animate)

    def on_closing(self):
        if self.is_playing:
             if self.anim_id: self.master.after_cancel(self.anim_id)
        plt.close(self.fig); self.master.quit(); self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    viewer = PSOViewer(root)
    root.mainloop()
