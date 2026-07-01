import reflex as rx
import numpy as np

# --- LÓGICA DE GAUSS-JORDAN ---
def calcular_gauss_jordan(A, b):
    A_original = np.array(A, dtype=float)
    b_original = np.array(b, dtype=float)

    A = A_original.copy()
    b = b_original.copy()
    n = len(b)

    augmented = np.hstack([A, b.reshape(-1, 1)])
    steps = []
    row = 0

    for col in range(n):
        pivot = max(range(row, n), key=lambda r: abs(augmented[r][col]))

        if abs(augmented[pivot][col]) < 1e-10:
            continue

        if pivot != row:
            augmented[[row, pivot]] = augmented[[pivot, row]]

        augmented[row] = augmented[row] / augmented[row][col]

        for r in range(n):
            if r != row:
                augmented[r] = augmented[r] - augmented[r][col] * augmented[row]

        augmented[np.abs(augmented) < 1e-10] = 0
        steps.append(np.round(augmented.copy(), 2).tolist())
        row += 1

        if row == n:
            break

    rank_A = np.linalg.matrix_rank(A_original)
    rank_aug = np.linalg.matrix_rank(
        np.hstack([A_original, b_original.reshape(-1, 1)])
    )

    final_matrix = np.round(augmented, 2).tolist()

    if rank_A < rank_aug:
        return {
            "solution": [],
            "solution_type": "Sin Solución",
            "steps": steps,
            "final_matrix": final_matrix,
            "desc": "El sistema es Incompatible (Sin Solución). El rango de la matriz de coeficientes es menor que el de la matriz ampliada, lo que significa que las ecuaciones se contradicen."
        }

    if rank_A < n:
        return {
            "solution": [],
            "solution_type": "Infinitas Soluciones",
            "steps": steps,
            "final_matrix": final_matrix,
            "desc": "El sistema es Compatible Indeterminado (Infinitas Soluciones). El rango es menor que el número de incógnitas, indicando la presencia de variables libres."
        }

    solution = np.round(augmented[:, -1], 2).tolist()
    sol_str = ", ".join([f"x{i+1} = {v:.2f}" for i, v in enumerate(solution)])

    return {
        "solution": solution,
        "solution_type": "Solución Única",
        "steps": steps,
        "final_matrix": final_matrix,
        "desc": f"¡Sistema resuelto con éxito! Se ha alcanzado una Solución Única mediante el método de reducción completa de Gauss-Jordan. Los valores calculados son: {sol_str}."
    }


# --- ESTADO DE REFLEX ---
class State(rx.State):
    size: int = 2
    matrix: list[list[float]] = [[0.0, 0.0], [0.0, 0.0]]
    vector: list[float] = [0.0, 0.0]

    solution: list[str] = []
    solution_type: str = ""
    solution_desc: str = ""
    steps: list[list[list[str]]] = []
    final_matrix: list[list[str]] = []
    error: str = ""

    def set_size(self, value: str):
        size = int(value)
        self.size = size
        self.matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        self.vector = [0.0 for _ in range(size)]
        self.solution = []
        self.solution_type = ""
        self.solution_desc = ""
        self.steps = []
        self.final_matrix = []
        self.error = ""

    def update_matrix(self, i: int, j: int, value: str):
        matrix = [row[:] for row in self.matrix]
        try: matrix[i][j] = float(value)
        except Exception: matrix[i][j] = 0.0
        self.matrix = matrix

    def update_vector(self, i: int, value: str):
        vector = self.vector[:]
        try: vector[i] = float(value)
        except Exception: vector[i] = 0.0
        self.vector = vector

    def solve(self):
        self.error = ""
        try:
            data = calcular_gauss_jordan(self.matrix, self.vector)
            self.solution = [f"{float(x):.2f}" for x in data.get("solution", [])]
            self.solution_type = data.get("solution_type", "")
            self.solution_desc = data.get("desc", "")

            raw_steps = data.get("steps", [])
            self.steps = [[[f"{float(v):.2f}" for v in row] for row in step] for step in raw_steps]

            raw_final = data.get("final_matrix", [])
            self.final_matrix = [[f"{float(v):.2f}" for v in row] for row in raw_final]
        except Exception as e:
            self.solution = []
            self.solution_type = ""
            self.steps = []
            self.final_matrix = []
            self.error = str(e)


# --- COMPONENTES DE LA INTERFAZ ---

def header_view():
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.image(
                    src="/logo.png", 
                    width="50px", 
                    height="50px"
                ),
                rx.vstack(
                    rx.heading("Métodos Numéricos", size="5", color="#ffffff", font_weight="bold"),
                    rx.text("Universidad de Panamá · Ing. Informática", color="#06b6d4", font_size="12px"),
                    spacing="0",
                ),
                spacing="4",
                align="center",
            ),
            rx.hstack(
                rx.box(width="7px", height="7px", bg="#22c55e", border_radius="full"),
                rx.text("Backend activo", color="#94a3b8", font_size="12px"),
                spacing="2",
                align="center",
            ),
            justify="between",
            width="100%",
            padding_bottom="1.2em",
            border_bottom="1px solid rgba(6, 182, 212, 0.2)",
        ),
        width="100%",
    )


def sidebar_inputs():
    return rx.box(
        rx.vstack(
            rx.text("F(X) / SISTEMA", color="#06b6d4", font_size="11px", font_weight="bold", letter_spacing="1px"),
            rx.text("Dimensión de la Matriz", color="#94a3b8", font_size="13px"),
            rx.select(
                [str(i) for i in range(2, 7)],
                value=State.size.to_string(),
                on_change=State.set_size,
                width="100%",
                color="#ffffff",
                bg="rgba(11, 19, 36, 0.4)", # Selector transparente
                border_color="rgba(6, 182, 212, 0.4)",
            ),
            rx.divider(border_color="rgba(6, 182, 212, 0.2)", margin_y="0.8em"),
            
            rx.text("COEFICIENTES (A) & VECTOR (B)", color="#06b6d4", font_size="11px", font_weight="bold", letter_spacing="1px"),
            
            rx.vstack(
                rx.foreach(
                    State.matrix,
                    lambda row, i:
                    rx.hstack(
                        rx.foreach(
                            row,
                            lambda value, j:
                            rx.input(
                                value=value.to_string(),
                                width="55px",
                                placeholder="0",
                                on_change=lambda v, i=i, j=j: State.update_matrix(i, j, v),
                                color="#ffffff",
                                bg="rgba(11, 19, 36, 0.2)", # Entradas transparentes
                                border_color="rgba(6, 182, 212, 0.4)",
                                _focus={"border_color": "#06b6d4", "bg": "rgba(6, 182, 212, 0.1)"},
                                text_align="center",
                                size="2",
                            ),
                        ),
                        rx.text("=", color="#06b6d4", font_weight="bold"),
                        rx.input(
                            value=State.vector[i].to_string(),
                            width="55px",
                            on_change=lambda v, i=i: State.update_vector(i, v),
                            color="#0b1324",
                            bg="rgba(34, 211, 238, 0.85)", # Un poco transparente para ver el fondo sin perder contraste
                            border_color="#06b6d4",
                            text_align="center",
                            font_weight="bold",
                            size="2",
                        ),
                        spacing="2",
                        align="center",
                    ),
                ),
                spacing="2",
                width="100%",
                overflow_x="auto",
                padding_y="0.4em",
            ),
            
            rx.button(
                "Ejecutar Gauss-Jordan",
                on_click=State.solve,
                width="100%",
                size="3",
                bg="#0891b2",
                color="#ffffff",
                _hover={"bg": "#0e7490", "box_shadow": "0 0 12px rgba(6,182,212,0.5)"},
                margin_top="0.8em",
            ),
            spacing="3",
            width="100%",
        ),
        width="280px",
        padding="1.2em",
        bg="rgba(15, 23, 42, 0.2)", # Tarjeta izquierda azul transparente
        border="1px solid rgba(6, 182, 212, 0.4)",
        border_radius="12px",
        backdrop_filter="blur(4px)",
    )


def results_dashboard():
    return rx.cond(
        State.solution_type != "",
        rx.vstack(
            rx.hstack(
                rx.box(width="6px", height="6px", bg="#22c55e", border_radius="full"),
                rx.text("Cálculo finalizado con éxito", color="#22c55e", font_size="12px", font_weight="bold"),
                bg="rgba(34, 197, 94, 0.08)",
                padding="0.3em 0.8em",
                border_radius="full",
                border="1px solid rgba(34, 197, 94, 0.2)",
                width="fit-content",
            ),
            rx.grid(
                rx.box(
                    rx.text("TIPO DE SOLUCIÓN", font_size="9px", color="#94a3b8", letter_spacing="1px"),
                    rx.heading(State.solution_type, size="3", color="#22d3ee", margin_top="0.2em"),
                    padding="0.8em", bg="rgba(11, 19, 36, 0.3)", border="1px solid rgba(6, 182, 212, 0.3)", border_radius="8px"
                ),
                rx.box(
                    rx.text("NÚMERO DE INCOGNITAS", font_size="9px", color="#94a3b8", letter_spacing="1px"),
                    rx.heading(f"{State.size} Variables", size="3", color="#ffffff", margin_top="0.2em"),
                    padding="0.8em", bg="rgba(11, 19, 36, 0.3)", border="1px solid rgba(6, 182, 212, 0.3)", border_radius="8px"
                ),
                rx.box(
                    rx.text("ESTADO DEL SISTEMA", font_size="9px", color="#94a3b8", letter_spacing="1px"),
                    rx.heading("Estable", size="3", color="#22c55e", margin_top="0.2em"),
                    padding="0.8em", bg="rgba(11, 19, 36, 0.3)", border="1px solid rgba(6, 182, 212, 0.3)", border_radius="8px"
                ),
                columns="3",
                spacing="2",
                width="100%",
                margin_top="0.3em",
            ),
            rx.hstack(
                rx.hstack(
                    rx.icon("check-circle", color="#22c55e", size=18),
                    rx.text(State.solution_desc, color="#cbd5e1", font_size="12px"),
                    spacing="2",
                    align="center",
                ),
                rx.button(
                    "Copiar", 
                    size="1", 
                    variant="soft", 
                    color_scheme="cyan",
                    on_click=rx.set_clipboard(State.solution_desc)
                ),
                bg="rgba(11, 19, 36, 0.2)",
                border="1px solid rgba(6, 182, 212, 0.2)",
                padding="0.8em",
                border_radius="8px",
                justify="between",
                align="center",
                width="100%",
                margin_top="0.3em",
            ),
            rx.cond(
                State.solution != [],
                rx.box(
                    rx.heading("Valores de las Variables", size="2", color="#ffffff", margin_bottom="0.4em"),
                    rx.grid(
                        rx.foreach(
                            State.solution,
                            lambda value, i:
                            rx.box(
                                rx.text(f"x{i + 1} = ", font_weight="bold", color="#06b6d4", as_="span", font_size="13px"),
                                rx.text(value, color="#ffffff", font_weight="bold", as_="span", font_size="13px"),
                                padding="0.4em 0.8em",
                                bg="rgba(11, 19, 36, 0.3)", # Resultados de variables transparentes
                                border="1px solid rgba(6, 182, 212, 0.4)",
                                border_radius="6px",
                                text_align="center",
                            ),
                        ),
                        columns="4",
                        spacing="2",
                        width="100%",
                    ),
                    width="100%",
                    margin_top="0.8em",
                )
            ),
            rx.box(
                rx.heading("Matriz Escalonada Completa (Resultado)", size="2", color="#ffffff", margin_bottom="0.4em"),
                rx.vstack(
                    rx.foreach(
                        State.final_matrix,
                        lambda row:
                        rx.hstack(
                            rx.foreach(
                                row,
                                lambda val:
                                rx.box(
                                    rx.text(val, font_family="monospace", color="#ffffff", font_size="12px"),
                                    border="1px solid rgba(6, 182, 212, 0.4)", # Celdas de matriz transparentes
                                    padding="0.3em",
                                    min_width="55px",
                                    text_align="center",
                                    border_radius="5px",
                                    bg="rgba(11, 19, 36, 0.3)",
                                ),
                            ),
                            spacing="1",
                        ),
                    ),
                    spacing="1",
                ),
                width="100%",
                margin_top="0.8em",
            ),
            rx.box(
                rx.heading("Historial de Operaciones por Filas", size="2", color="#ffffff", margin_bottom="0.4em"),
                rx.box(
                    rx.vstack(
                        rx.foreach(
                            State.steps,
                            lambda step, idx:
                            rx.box(
                                rx.text(f"Paso {idx + 1} - Reducción del Pivote", font_weight="bold", color="#06b6d4", margin_bottom="0.3em", font_size="11px"),
                                rx.vstack(
                                    rx.foreach(
                                        step,
                                        lambda row:
                                        rx.hstack(
                                            rx.foreach(
                                                row,
                                                lambda val:
                                                rx.box(
                                                    rx.text(val, font_family="monospace", color="#94a3b8", font_size="11px"),
                                                    border="1px solid rgba(6, 182, 212, 0.25)",
                                                    padding="0.2em",
                                                    min_width="50px",
                                                    text_align="center",
                                                    border_radius="4px",
                                                    bg="rgba(11, 19, 36, 0.3)", # Filas de pasos transparentes
                                                ),
                                            ),
                                            spacing="1",
                                        ),
                                    ),
                                    spacing="1",
                                ),
                                width="100%",
                                bg="rgba(15, 23, 42, 0.2)", # Bloque de cada paso transparente
                                padding="0.6em",
                                border_radius="6px",
                                border="1px solid rgba(6, 182, 212, 0.3)",
                            ),
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    max_height="200px",
                    overflow_y="auto",
                    padding_right="0.4em",
                ),
                width="100%",
                margin_top="0.8em",
            ),
            spacing="3",
            width="100%",
            align="start",
        ),
        rx.center(
            rx.vstack(
                rx.icon("table", size=30, color="#06b6d4"),
                rx.text("Ingrese los datos de la matriz y presione calcular para ver los resultados.", color="#475569", text_align="center", font_size="13px"),
                align="center",
                spacing="2",
            ),
            width="100%",
            height="260px",
            border="2px dashed rgba(6, 182, 212, 0.4)",
            border_radius="12px",
        )
    )


def index():
    return rx.box(
        rx.el.style(
            """
            @keyframes scrollBackground {
                0% { background-position: 0 0; }
                100% { background-position: 300px 300px; }
            }
            .doodle-background-layer {
                background-color: #040811;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'%3E%3Cg fill='none' stroke='%230891b2' stroke-width='1' stroke-opacity='0.15'%3E%3Cpath d='M20 30 l15 0 M27 23 l0 14'/%3E%3Ccircle cx='70' cy='30' r='8'/%3E%3Cpath d='M65 30 l10 0'/%3E%3Ctext x='110' y='35' font-family='serif' font-size='16' fill='%230891b2' fill-opacity='0.15' font-style='italic'%3E∫%3C/text%3E%3Cpath d='M20 75 Q 30 60, 40 75 T 60 75' /%3E%3Ctext x='85' y='80' font-family='serif' font-size='14' fill='%230891b2' fill-opacity='0.15'%3Eπ%3C/text%3E%3Cpath d='M120 70 l15 15 M135 70 l-15 15'/%3E%3Ctext x='20' y='130' font-family='serif' font-size='14' fill='%230891b2' fill-opacity='0.15' font-style='italic'%3Ef(x)%3C/text%3E%3Cpath d='M70 115 l10 0 M70 123 l10 0 M70 131 l10 0'/%3E%3Crect x='115' y='115' width='20' height='20' rx='3'/%3E%3C/g%3E%3C/svg%3E");
                background-repeat: repeat;
                animation: scrollBackground 35s linear infinite;
            }
            """
        ),
        # Fondo móvil animado
        rx.box(
            width="100%",
            height="100%",
            position="absolute",
            top="0",
            left="0",
            z_index="0",
            class_name="doodle-background-layer",
        ),
        rx.center(
            # Panel Base Esmerilado General
            rx.box(
                rx.vstack(
                    header_view(),
                    rx.hstack(
                        sidebar_inputs(),
                        rx.box(
                            results_dashboard(),
                            flex="1",
                            min_width="380px",
                            padding="1.2em",
                            bg="rgba(15, 23, 42, 0.2)", # Tarjeta derecha transparente
                            border="1px solid rgba(6, 182, 212, 0.4)",
                            border_radius="12px",
                            backdrop_filter="blur(4px)",
                        ),
                        width="100%",
                        spacing="4",
                        align="start",
                        flex_wrap="wrap",
                    ),
                    spacing="4",
                    width="100%",
                ),
                width="100%",
                max_width="1050px",
                padding="1.8em",
                bg="rgba(10, 17, 32, 0.7)", 
                backdrop_filter="blur(16px)", 
                border="1px solid rgba(6, 182, 212, 0.5)", 
                border_radius="16px",
                box_shadow="0 25px 50px rgba(4, 145, 178, 0.15)",
            ),
            padding="2em",
            min_height="100vh",
            position="relative",
            z_index="1",
        ),
        min_height="100vh",
        overflow="hidden",
    )

app = rx.App()
app.add_page(index)