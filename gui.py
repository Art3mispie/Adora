from collections import deque
import tkinter as tk

import numpy as np


PAPER = "#db7d9c"
WHITE = "#ffdfe4"
INK = "#202124"
MUTED = "#6b7075"
GRID = "#d9dcde"
BLUE = "#326891"
ORANGE = "#d9822b"
GREEN = "#3f7d63"
RED = "#b34a4a"


def project(points, axes: tuple[int, int], invert_second: bool = False) -> np.ndarray:
    values = np.asarray(points, dtype=float).reshape(-1, 3)[:, axes].copy()
    if invert_second:
        values[:, 1] *= -1.0
    return values


def panel(parent, title: str) -> tk.Frame:
    frame = tk.Frame(parent, bg=WHITE, highlightbackground=GRID, highlightthickness=1)
    tk.Label(
        frame,
        text=title.upper(),
        bg=WHITE,
        fg=MUTED,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(fill="x", padx=10, pady=(8, 3))
    return frame


class SlicePlot:
    def __init__(
        self,
        parent,
        title: str,
        axes: tuple[int, int],
        labels: tuple[str, str],
        invert_second: bool = False,
    ) -> None:
        self.axes = axes
        self.labels = labels
        self.invert_second = invert_second
        self.frame = panel(parent, title)
        self.canvas = tk.Canvas(
            self.frame, bg=WHITE, highlightthickness=0, width=390, height=280
        )
        self.canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))

    def draw(self, snapshot: dict, trail: deque) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(240, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        left, top, right, bottom = 42, 10, width - 12, height - 30

        course = project(snapshot["course"], self.axes, self.invert_second)
        path = project(snapshot["path"], self.axes, self.invert_second)
        history = project(trail, self.axes, self.invert_second)
        position = project([snapshot["position"]], self.axes, self.invert_second)[0]
        visible = np.vstack((course, path, history, position))
        minimum = visible.min(axis=0)
        maximum = visible.max(axis=0)
        span = np.maximum(maximum - minimum, (8.0, 6.0))
        minimum -= 0.08 * span
        maximum += 0.08 * span

        def screen(point):
            x = left + (point[0] - minimum[0]) / (maximum[0] - minimum[0]) * (
                right - left
            )
            y = bottom - (point[1] - minimum[1]) / (maximum[1] - minimum[1]) * (
                bottom - top
            )
            return float(x), float(y)

        for index in range(5):
            fraction = index / 4
            x = left + fraction * (right - left)
            y = bottom - fraction * (bottom - top)
            canvas.create_line(x, top, x, bottom, fill=GRID)
            canvas.create_line(left, y, right, y, fill=GRID)
        canvas.create_rectangle(left, top, right, bottom, outline=MUTED)

        def line(points, colour, width_px=1, dash=None):
            if len(points) > 1:
                coordinates = [value for point in points for value in screen(point)]
                canvas.create_line(
                    *coordinates, fill=colour, width=width_px, dash=dash
                )

        line(course, MUTED, dash=(3, 4))
        line(path, BLUE, 2)
        line(history, INK, 2)

        active = int(snapshot["active_gate"])
        for index, gate in enumerate(course):
            x, y = screen(gate)
            colour = ORANGE if index == active else MUTED
            radius = 5 if index == active else 4
            canvas.create_rectangle(
                x - radius, y - radius, x + radius, y + radius,
                outline=colour, fill=WHITE, width=2
            )
            canvas.create_text(
                x + 7, y - 5, text=f"G{index + 1}", anchor="sw",
                fill=colour, font=("Segoe UI", 8)
            )

        if snapshot["target"] is not None:
            target = project(
                [snapshot["target"]], self.axes, self.invert_second
            )[0]
            x, y = screen(target)
            canvas.create_line(x - 6, y, x + 6, y, fill=ORANGE, width=2)
            canvas.create_line(x, y - 6, x, y + 6, fill=ORANGE, width=2)
        x, y = screen(position)
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=BLUE, outline=WHITE)
        canvas.create_text(
            (left + right) / 2,
            height - 4,
            text=self.labels[0],
            anchor="s",
            fill=MUTED,
            font=("Segoe UI", 8),
        )
        canvas.create_text(
            4,
            (top + bottom) / 2,
            text=self.labels[1],
            anchor="w",
            fill=MUTED,
            font=("Segoe UI", 8),
            angle=90,
        )


class Dashboard:
    def __init__(self, get_snapshot, get_frame, restart, kill, close) -> None:
        self.get_snapshot = get_snapshot
        self.get_frame = get_frame
        self.restart = restart
        self.kill = kill
        self.close_callback = close
        self.last_frame = None
        self.last_camera_size = None
        self.last_mode = None
        self.trail = deque(maxlen=500)
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Adora — Flight Monitor")
        self.root.geometry("1280x760")
        self.root.minsize(1040, 650)
        self.root.configure(bg=PAPER)

        header = tk.Frame(self.root, bg=PAPER)
        header.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(
            header, text="ADORA", bg=PAPER, fg=INK,
            font=("Segoe UI", 18, "bold")
        ).pack(side="left")
        tk.Label(
            header, text="  FLIGHT MONITOR", bg=PAPER, fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(side="left", pady=(6, 0))
        tk.Button(
            header, text="Kill  [K]", command=self.kill, fg=RED, bg=WHITE,
            font=("Segoe UI", 9, "bold"), padx=12
        ).pack(side="right", padx=(6, 0))
        tk.Button(
            header, text="Restart  [R]", command=self.restart, fg=BLUE, bg=WHITE,
            font=("Segoe UI", 9, "bold"), padx=12
        ).pack(side="right")
        self.mode_label = tk.Label(
            header, text="GROUND", bg=WHITE, fg=MUTED,
            font=("Segoe UI", 9, "bold"), padx=10, pady=5
        )
        self.mode_label.pack(side="right", padx=10)

        content = tk.Frame(self.root, bg=PAPER)
        content.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        content.grid_columnconfigure(0, weight=5)
        content.grid_columnconfigure(1, weight=3)
        content.grid_columnconfigure(2, weight=2)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        camera_panel = panel(content, "Camera evidence")
        camera_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 6))
        self.camera_label = tk.Label(
            camera_panel, text="Waiting for camera", bg="#2a2a28", fg=WHITE
        )
        self.camera_label.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.plan_plot = SlicePlot(
            content, "Plan slice", (0, 1), ("North [m]", "East [m]")
        )
        self.plan_plot.frame.grid(
            row=0, column=1, sticky="nsew", padx=6, pady=(0, 4)
        )
        self.elevation_plot = SlicePlot(
            content,
            "Elevation slice",
            (0, 2),
            ("North [m]", "Altitude [m]"),
            True,
        )
        self.elevation_plot.frame.grid(
            row=1, column=1, sticky="nsew", padx=6, pady=(4, 0)
        )

        metrics_panel = panel(content, "Run state")
        metrics_panel.grid(
            row=0, column=2, rowspan=2, sticky="nsew", padx=(6, 0)
        )
        self.metrics = tk.Label(
            metrics_panel, text="", bg=WHITE, fg=INK, justify="left",
            anchor="nw", font=("Consolas", 9), padx=10, pady=8
        )
        self.metrics.pack(fill="both", expand=True)
        self.message = tk.Label(
            metrics_panel, text="", bg="#eeeeea", fg=MUTED, anchor="w",
            justify="left", wraplength=230, font=("Segoe UI", 9), padx=8, pady=8
        )
        self.message.pack(fill="x", padx=6, pady=(0, 6))

        self.root.bind("<r>", lambda _event: self.restart())
        self.root.bind("<k>", lambda _event: self.kill())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _draw_camera(self) -> None:
        if self.get_frame is None:
            return
        frame = self.get_frame()
        if frame is None:
            return
        import cv2

        panel_size = (
            self.camera_label.winfo_width(),
            self.camera_label.winfo_height(),
        )
        if frame[0] == self.last_frame and panel_size == self.last_camera_size:
            return
        self.last_frame = frame[0]
        self.last_camera_size = panel_size
        source_height, source_width = frame[1].shape[:2]
        available_width = min(560, max(280, panel_size[0] - 12))
        available_height = min(720, max(180, panel_size[1] - 12))
        scale = min(
            available_width / source_width,
            available_height / source_height,
        )
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(
            cv2.resize(frame[1], (width, height), interpolation=interpolation),
            cv2.COLOR_BGR2RGB,
        )
        image = tk.PhotoImage(
            data=f"P6\n{width} {height}\n255\n".encode() + rgb.tobytes()
        )
        self.camera_label.configure(image=image, text="")
        self.camera_label.image = image

    def _update(self) -> None:
        if self.closed:
            return
        snapshot = self.get_snapshot()
        mode = snapshot["mode"]
        if mode == "GROUND" and self.last_mode != "GROUND":
            self.trail.clear()
        self.last_mode = mode
        self.trail.append(np.asarray(snapshot["position"]).copy())
        self.plan_plot.draw(snapshot, self.trail)
        self.elevation_plot.draw(snapshot, self.trail)
        self._draw_camera()

        colours = {
            "GROUND": (WHITE, MUTED),
            "LAUNCH": ("#fff2df", ORANGE),
            "RACE": ("#e5f0eb", GREEN),
            "FINISHED": ("#e8eef3", BLUE),
            "KILL": ("#f4e4e4", RED),
        }
        background, foreground = colours.get(mode, (WHITE, INK))
        self.mode_label.configure(
            text=f"{mode}{' · ARMED' if snapshot['armed'] else ''}",
            bg=background,
            fg=foreground,
        )

        position = snapshot["position"]
        velocity = snapshot["velocity"]
        rate = snapshot["imu_rate"]
        vision = snapshot["vision"]
        plan = snapshot["plan"]
        total = int(snapshot["gate_count"])
        gate = min(snapshot["active_gate"] + 1, total) if total else 0
        plan_text = "pending" if plan is None else f"{plan.duration_s:.2f} s"
        lines = (
            "RUN\n"
            f"  target      {gate} / {total}\n"
            f"  phase       {snapshot['phase_s']:.2f} / {snapshot['duration_s']:.2f} s\n"
            f"  collisions  {snapshot['collisions']}\n\n"
            "ESTIMATE · NED\n"
            f"  p  {position[0]:6.2f} {position[1]:6.2f} {position[2]:6.2f} m\n"
            f"  v  {velocity[0]:6.2f} {velocity[1]:6.2f} {velocity[2]:6.2f}\n"
            f"  speed       {snapshot['speed']:.2f} m/s\n\n"
            "CONTROL\n"
            f"  error       {snapshot['track_error']:.2f} m\n"
            f"  throttle    {snapshot['throttle']:.3f}\n"
            f"  tilt        {snapshot['tilt_deg']:.1f}°\n"
            f"  rates       {rate[0]:.2f} {rate[1]:.2f} {rate[2]:.2f}\n\n"
            "VISION · DIAGNOSTIC\n"
            f"  detector    {vision.get('detector', 'off')}\n"
            f"  frames      {vision.get('camera_frames', 0)}\n"
            f"  runs        {vision.get('detector_runs', 0)}\n"
            f"  latency     {vision.get('detector_ms', 0.0):.2f} ms\n"
            f"  detections  {vision.get('detections', 0)}\n"
            f"  mapped      {vision.get('mapped_gates', 0)} / {total}\n"
            f"  map settled {'yes' if vision.get('map_ready') else 'no'}\n"
            f"  matches     {vision.get('matches', 0)}\n"
            f"  corrections {vision.get('corrections', 0)}\n"
            f"  aim error   {vision.get('aim_error_deg', 0.0):.1f} deg\n"
            f"  gate offset {vision.get('gate_offset_m', 0.0):.2f} m\n\n"
            f"PLAN  {plan_text}"
        )
        self.metrics.configure(text=lines)
        warnings = ", ".join(snapshot["warnings"])
        self.message.configure(text=snapshot["message"] or warnings or "Nominal")
        self.root.after(100, self._update)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.close_callback()
            self.root.destroy()

    def run(self) -> None:
        self._update()
        self.root.mainloop()
