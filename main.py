import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import matplotlib.animation as animation
import time

x_vals = []
y_vals = []
f_interp = None
search_point = None
anim_running = False
search_generator = None
best_point = None
min_text = None
counter = 0
iteration_counter = 0
start_time = None
elapsed = 0

fig, ax = plt.subplots()
plt.subplots_adjust(bottom = 0.2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Klikni misem da crtas funkciju. Klikni enter kada zavrsis')

# --- CRTANEJ FUNKCIJE ---
def onclick(event):
    if event.inaxes == ax and event.button == 1 and event.xdata is not None and event.ydata is not None:
        x_vals.append(event.xdata)
        y_vals.append(event.ydata)
        ax.plot(event.xdata, event.ydata, 'ro')
        fig.canvas.draw()

def on_key(event):
    if event.key == 'enter':
        prepare_f_plot()

# --- RESET ---
def reset(event):
    global x_vals, y_vals, f_interp, search_point, anim_running, min_text, best_point, counter
    x_vals.clear()
    y_vals.clear()
    f_interp = None
    counter = 0
    anim_running = False
    min_text = None
    search_point = None
    best_point = None
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Klikni da crtas funkciju. ENTER za kraj crtanja")
    fig.canvas.draw()


# --- PRIPREMA FUNKCIJE ---
def prepare_f_plot():
    global f_interp, search_point, best_point, min_text
    if len(x_vals) < 2:
        print("Prvo nacrtaj funkciju!")
        return

    sorted_points = sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])
    x_sorted, y_sorted = zip(*sorted_points)
    x_sorted = np.array(x_sorted)
    y_sorted = np.array(y_sorted)
    f_interp = interp1d(x_sorted, y_sorted, kind='cubic', bounds_error=False, fill_value='extrapolate')

    if search_point is not None:
        try:
            search_point.remove()
        except Exception:
            pass
        search_point = None

    if best_point is not None:
        try:
            best_point.remove()
        except Exception:
            pass
        best_point = None

    if min_text is not None:
        try:
            min_text.remove()
        except Exception:
            pass
        min_text = None

    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(x_sorted, y_sorted, 'ro', label="Tacke")
    x_new = np.linspace(x_sorted[0], x_sorted[-1], 500)
    y_new = f_interp(x_new)
    ax.plot(x_new, y_new, 'b-', label="Interpolovana funkcija")
    ax.legend()
    fig.canvas.draw()


# --- METOD SKENIRANJA ---
def metoda_skeniranjem(f, a, b, step = 0.001):
    global iteration_counter
    iteration_counter = 0

    num_points = int((b - a) / step) + 1
    x_values = np.linspace(a, b, num_points)
    best_x = None
    best_val = float('inf')

    for x in x_values:
        iteration_counter += 1
        val = f(x)
        if math.isnan(val) or math.isinf(val):
            continue
        if val < best_val:
            best_val = val
            best_x = x

        yield x, val, best_x, best_val, iteration_counter

    if best_x is None:
        yield best_x, best_val, best_x, best_val, iteration_counter

def start_skeniranja(event):
    global search_generator, anim_running, min_text, start_time, elapsed
    if f_interp is None:
        print('Funkcija nije nacrtana')
        return

    if min_text is not None:
        min_text.remove()
        min_text = None

    start_time = time.time()
    elapsed = 0

    search_generator = metoda_skeniranjem(f_interp, min(x_vals), max(x_vals))
    anim_running = True
    ani.event_source.start()
    ax.set_title('Minimum - metod skeniranja')
    min_text = ax.text(0.005, 0.05, '', transform=ax.transAxes, color='blue', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))


# --- NJUTNOVI METODI ---

def izvod(f, h=1e-5):
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)

def drugi_izvod(f, h=1e-5):
    return lambda x: (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

def newtonova_metoda(f, df, x0, e = 1e-6):
    global iteration_counter
    iteration_counter = 0

    x = x0
    ddf = drugi_izvod(f)

    best_x = x
    best_val = f(x)

    while True:
        iteration_counter += 1

        val = f(x)
        df_val = df(x)
        ddf_val = ddf(x)

        if abs(ddf_val) < e or math.isinf(ddf_val) or math.isnan(ddf_val):
            print("Prekinuto: drugi izvod je 0")
            break

        if val < best_val:
            best_x = x
            best_val = val

        x_new = x - df_val / ddf_val
        x_new = np.clip(x_new, min(x_vals), max(x_vals))

        if abs(x_new - best_x) < e or abs(f(x_new) - best_val) < e:
            print(f"Konvergiralo {best_x}")
            break

        yield x, val, best_x, best_val, iteration_counter

        x = x_new

    if best_x is None:
        yield best_x, best_val, best_x, best_val, iteration_counter

def start_newton(event):
    global search_generator, anim_running, min_text, start_time, elapsed
    if len(x_vals) < 2:
        print("Nacrtaj funkciju prvo!")
        return

    if min_text is not None:
        min_text.remove()
        min_text = None

    start_time = time.time()
    elapsed = 0

    df_interp = izvod(f_interp)
    x0 = x_vals[np.argmin(y_vals)]

    search_generator = newtonova_metoda(f_interp, df_interp, x0)
    anim_running = True
    ani.event_source.start()
    ax.set_title('Minimum - Njutnova metoda')

    try:
        min_text.remove()
    except:
        pass

    min_text = ax.text(0.05, 0.05, '', transform=ax.transAxes, color='blue', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))


# --- ZLATNI PRESEK ---

def zlatni_presek(f, a, b, e=1e-6):
    global iteration_counter
    ro = (-1 + 5**0.5) / 2
    x1 = b - ro * (b - a)
    x2 = a + ro * (b - a)

    iteration_counter = 0

    while (b - a) > e:
        iteration_counter += 1
        f1 = f(x1)
        f2 = f(x2)

        if f1 < f2:
            b = x2
            x2 = x1
            x1 = b - ro * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + ro * (b - a)

        yield x1, f(x1), x2, f(x2), iteration_counter

    x_min = (a + b) / 2
    f_min = f(x_min)

    yield x_min, f_min, x_min, f_min, iteration_counter


def start_zlatni_presek(event):
    global search_generator, anim_running, min_text, start_time, elapsed
    if f_interp is None:
        print('Funkcija nije nacrtana')
        return

    if min_text is not None:
        min_text.remove()
        min_text = None

    start_time = time.time()
    elapsed = 0

    search_generator = zlatni_presek(f_interp, min(x_vals), max(x_vals))
    anim_running = True
    ani.event_source.start()
    ax.set_title('Minimum - Zlatni presek')

    min_text = ax.text(0.05, 0.05, '', transform=ax.transAxes, color='blue', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))


# --- GRADIJENTNI SPUST ---
def gradijentni_spust(f, df, x0, lr = 0.01, e = 1e-6):
    global iteration_counter
    iteration_counter = 0

    df = izvod(f)
    x = x0
    best_x = x0
    best_val = f(x)

    while True:
        iteration_counter += 1
        gradijent = df(x)
        x_new = x - lr * gradijent
        val = f(x_new)

        if math.isinf(val) or math.isnan(val):
            break

        if val < best_val:
            best_x = x_new
            best_val = val

        yield x_new, val, best_x, best_val, iteration_counter

        if abs(x_new - x) < e or abs(val - f(x)) < e:
            break

        x = x_new

    yield best_x, best_val, best_x, best_val, iteration_counter


def start_gradijentni_spust(event):
    global search_generator, anim_running, min_text, start_time, elapsed
    if f_interp is None:
        print('Funkcija nije nacrtana')
        return

    if min_text is not None:
        min_text.remove()
        min_text = None

    start_time = time.time()
    elapsed = 0

    search_generator = gradijentni_spust(f_interp, min(x_vals), max(x_vals))
    anim_running = True
    ani.event_source.start()
    ax.set_title('Minimum - Gradijentni spust')

    min_text = ax.text(0.05, 0.05, '', transform=ax.transAxes, color='blue', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))


# --- ANIMACIJA ---
def update(frame):
    global search_generator, search_point, anim_running, best_point, counter, min_text, start_time, elapsed, iteration_counter

    if not anim_running or search_generator is None:
        return
    try:
        x, val, best_x, best_val, iteration_counter = next(search_generator)

        if search_point is not None:
            search_point.remove()
            search_point = None

        if best_point is not None:
            best_point.remove()
            best_point = None

        search_point = ax.plot(x, val, 'go', label="Trenutna pozicija")[0]
        best_point = ax.plot(best_x, best_val, 'bo', label="Minimum")[0]

        elapsed = time.time() - start_time

        if min_text is not None:
            min_text.set_text(f'Minimum: x={best_x:.6f}, f(x)={best_val:.6f}, '
                              f'\nBroj iteracija = {iteration_counter}, '
                              f'\nVreme: {elapsed:.3f}s')

        ax.set_title(f"Trazenje minimuma x={x:.5f}, f(x)={val:5f}")
        fig.canvas.draw_idle()
    except StopIteration:
        print("Optimizacija je zavrsena")
        anim_running = False
        ani.event_source.stop()


# --- POVEZIVANEJ DOGADJAJA ---
cid_click = fig.canvas.mpl_connect('button_release_event', onclick)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

# --- DUGMICI ---
ax_reset = plt.axes([0.05, 0.05, 0.1, 0.075])
btn_reset = Button(ax_reset, 'Reset')
btn_reset.on_clicked(reset)

ax_button = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_skeniranje = Button(ax_button, label="Metod skeniranja")
btn_skeniranje.on_clicked(start_skeniranja)

ax_button_zlatni = plt.axes([0.4, 0.05, 0.1, 0.075])
btn_zlatni = Button(ax_button_zlatni, label="Zlatni presek")
btn_zlatni.on_clicked(start_zlatni_presek)

ax_button_newton = plt.axes([0.55, 0.05, 0.1, 0.075])
btn_newton = Button(ax_button_newton, label="Njutnov metod")
btn_newton.on_clicked(start_newton)

ax_button_grad = plt.axes([0.25, 0.05, 0.1, 0.075])
btn_grad = Button(ax_button_grad, label="Gradijentni spust")
btn_grad.on_clicked(start_gradijentni_spust)


ani = animation.FuncAnimation(fig, update, interval=10, save_count=1000)
plt.show()