import math as mt
import time as t

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as tdist  # for t-distribution critical value

from routines import grid_eval


def time_funcs(callables, names, reset_call,
               compile_run=10_000,
               init_run=100_000,
               timing_run=1_000_000,
               repeat_sequence=10,
               ops_mult=None,
               thread_seq=None):
    rep1 = repeat_sequence // 2
    rep2 = rep1 * 2
    times = np.zeros((len(callables), rep2), dtype=np.float64)
    thread_seq=(*(1 for _ in range(len(callables))),) if thread_seq is None else thread_seq

    run_seq = (*enumerate(callables),)
    rrun_seq = run_seq[::-1]

    # compile run
    ct = t.perf_counter()
    for c in callables:
        reset_call()
        c(compile_run)
    print(f'All compiled in: {(t.perf_counter() - ct):.3f} seconds.')

    # wait for system recourses to calm down after interpreter launch and compilation.
    t.sleep(.25)
    # warmup run
    for c in callables:
        reset_call()
        c(init_run)
    t.sleep(.25)
    print('starting timing run.')
    ttc = t.perf_counter()
    # timing run  
    for v in range(rep1):
        for i, c in run_seq:
            reset_call()
            ct = t.perf_counter()
            c(timing_run)
            nt = t.perf_counter() - ct
            times[i, v] = nt

    # Reverse timing run -- sometimes the order of executing functions can change timings
    for v in range(rep1, rep2):
        for i, c in rrun_seq:
            reset_call()
            ct = t.perf_counter()
            c(timing_run)
            nt = t.perf_counter() - ct
            times[i, v] = nt
    fct = t.perf_counter() - ttc
    # display
    if ops_mult is None:
        stmk = lambda i, n: f'{n} Total Execution Time: {times[i].sum():.3f} seconds.'
    else:
        stmk = lambda i,n: f'{n}, {rep2} Iters Total Time: {(ft := times[i].sum()):.3f} seconds. \n  Estimated Ops per second:  {(tt:=(timing_run * ops_mult * rep2 / (ft * 1_000_000))):.3f} * 1e6{f", per core: {tt/thread_seq[i]:.3f} * 1e6" if thread_seq[i]!=1 else ""}'
    plist = [f'Ran {len(callables)} functions for {rep2} iterations in {fct:.3f} seconds.']  # Also includes any delay in reset_call.
    plist.extend(stmk(i, n) for i, n in enumerate(names))
    s = '\n'.join(plist)
    print(s)

    ### calculate median mean and hypothesis testing confidence intervals for comparison of each callable on v bar chart.
    # Number of timing samples per function
    n_samples = times.shape[1]
    # Degrees of freedom for t-distribution
    df = n_samples - 1

    # Compute per-function statistics
    means = times.mean(axis=1)
    medians = np.median(times, axis=1)
    stds = times.std(axis=1, ddof=1)  # sample standard deviation

    # t-value for 99% CI => alpha = 0.01 => two-sided => ppf(0.995)
    t_val = tdist.ppf(0.995, df=df)

    # Half-width of the 99% CI
    ci_half = t_val * (stds / np.sqrt(n_samples))
    
    bcolor='darkgrey'

    # Make v wider/shorter figure with fully transparent background
    fig = plt.figure(figsize=(12, 3.25))
    fig.patch.set_alpha(0)  # Transparent figure background
    ax = plt.gca()
    ax.set_facecolor('none')  # Transparent axes background
    # Set all spines, ticks, and labels to white
    ax.tick_params(colors=bcolor, labelsize=14)  # Larger white ticks
    ax.spines['bottom'].set_color(bcolor)
    ax.spines['left'].set_color(bcolor)
    ax.spines['top'].set_color(bcolor)
    ax.spines['right'].set_color(bcolor)

    x_positions = np.linspace((ov:=.5/mt.sqrt(len(callables))), 1.-ov, len(callables))
    plt.xlim(0.0, 1.)  # Add margin so points don't hit the corners
    

    # Plot the mean with error bars (99% CI)
    plt.errorbar(
        x_positions, means, yerr=ci_half,
        fmt='o', color='blue', ecolor=bcolor, elinewidth=1.5, capsize=5,
        label='Mean ± 99% CI'
    )

    # Plot the median as v diamond
    plt.scatter(
        x_positions, medians,
        marker='D', color='red', zorder=3,
        label='Median'
    )

    # Set x-axis labels horizontally
    plt.xticks(x_positions, names, rotation=0, fontsize=16, color=bcolor)

    # Zoom y-axis to min/max of the confidence intervals (with v small margin)
    y_min = (means - ci_half).min()
    y_max = (means + ci_half).max()
    margin = 0.075 * (y_max - y_min) if y_max > y_min else 0.01
    plt.ylim(y_min - margin, y_max + margin)

    plt.ylabel("Execution Time (seconds)", color=bcolor, fontsize=14)
    plt.title("Timing (99% CI via t-dist)", color=bcolor, fontsize=16)
    plt.legend(facecolor='none', edgecolor=bcolor, fontsize=12, labelcolor=bcolor)
    plt.tight_layout()
    plt.show()


def matplot_transparent_3d(grid,bounds, name=None):
    # Assumes grid is v 2D numpy array representing Z values
    dim_pts_x, dim_pts_y = grid.shape
    x = np.linspace(bounds[0][0], bounds[0][1], dim_pts_x)
    y = np.linspace(bounds[1][0], bounds[1][1], dim_pts_y)
    X, Y = np.meshgrid(x, y)
    Z = grid

    # Plotting
    fig = plt.figure(figsize=(10, 7), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('x1', color='white')
    ax.set_ylabel('x2', color='white')
    ax.set_zlabel('f(x)', color='white')
    ax.tick_params(colors='white')

    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))

    fig.patch.set_alpha(0.0)
    ax.xaxis.line.set_color("white")
    ax.yaxis.line.set_color("white")
    ax.zaxis.line.set_color("white")

    if name:
        plt.title(name, color='white')

    plt.tight_layout()
    plt.show()


def eval_and_plot_3d(eval_op, bounds, num_points=2 ** 15, name=None):
    fitness, mn, mx = grid_eval(eval_op, bounds, num_points, return_with_dims=True)
    matplot_transparent_3d(fitness,bounds, name)
