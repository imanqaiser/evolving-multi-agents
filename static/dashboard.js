// Dashboard logic. Flask (app.py) owns the simulation. This file only asks the
// server for the current state as JSON (see build_state in app.py) and draws it,
// so the page never has to reload.

const COLORS = { A: "#1556b8", B: "#c2560c" };
const FOOD_EMPTY = [225, 236, 216];  // Depleted patch: pale green (red, green, blue).
const FOOD_FULL = [61, 139, 61];     // Full patch: deep green.
const GROUND = "#f3efe6";            // Cells where no food grows.
const TRAITS = [
    ["exploratory_tendency", "Exploration"],
    ["risk_tolerance", "Risk tolerance"],
    ["sociability", "Sociability"],
    ["cooperative_tendency", "Cooperation"],
];
const SPEEDS = [1, 2, 5, 10, 20, 40, 80];  // Timesteps per second for each slider position.

let state = null;        // The latest JSON from the server.
let history = {};        // DataCollector column → list of values, one per timestep.
let events = [];         // Event log entries, oldest first.
let selectedId = null;   // Agent shown in the inspector.
let selectedCell = null; // [row, column] of the last clicked cell.
let cellAgents = {};     // "row,column" → agents in that cell, rebuilt on every draw.
let running = false;     // True while the Run button is active.
let loopActive = false;  // True while the run loop is waiting for the server.

function byId(id) {
    return document.getElementById(id);
}

// Names and text from the server are inserted into HTML, so escape < > & " first.
function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, (character) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character]);
}

function format(number, digits = 2) {
    return Number(number).toFixed(digits);
}

// ---------------------------------------------------------------------------
// Talking to the server
// ---------------------------------------------------------------------------

// GET when there is no body, POST with JSON when there is one.
async function callServer(url, body) {
    const options = {};
    if (body !== undefined) {
        options.method = "POST";
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    if (!response.ok) {
        throw new Error(await response.text());
    }
    return response.json();
}

// The server only sends history rows and events newer than this timestep.
function lastTimestep() {
    return state ? state.timestep : -1;
}

async function loadState() {
    let url = `/api/state?since=${lastTimestep()}`;
    if (selectedId !== null) {
        url += `&selected=${selectedId}`;
    }
    receive(await callServer(url));
}

async function stepSimulation(count) {
    receive(await callServer("/api/step", { steps: count, since: lastTimestep(), selected: selectedId }));
}

async function resetSimulation() {
    pause();
    const newState = await callServer("/api/reset", {
        world: byId("world").value,
        environment: byId("environment").value,
    });
    forgetRun();
    // Individual encounters flood the log in the large world, so start with them hidden there.
    document.querySelector('[data-kind="encounter"]').checked = newState.world === "SMALL";
    receive(newState);
}

function forgetRun() {
    state = null;
    history = {};
    events = [];
    selectedId = null;
    selectedCell = null;
}

// Store a new state from the server, add its new history rows, and redraw.
function receive(newState) {
    // A different run_id means the world was reset somewhere else (another tab
    // or the debug page). Our history belongs to the old world, so start over.
    if (state && newState.run_id !== state.run_id) {
        forgetRun();
        loadState();
        return;
    }
    for (const column in newState.history) {
        if (!history[column]) {
            history[column] = [];
        }
        history[column].push(...newState.history[column]);
    }
    events.push(...newState.events);
    if (events.length > 500) {
        events = events.slice(-500);
    }
    state = newState;
    byId("error-banner").hidden = true;
    render();
}

function showError(error) {
    pause();
    byId("error-banner").textContent = `The server did not respond: ${error.message}`;
    byId("error-banner").hidden = false;
}

// ---------------------------------------------------------------------------
// Run, pause and step
// ---------------------------------------------------------------------------

// Keep asking the server for more timesteps until paused or everyone is dead.
// At high speeds several timesteps are run per request, so the browser does
// not have to redraw more often than about 20 times per second.
async function run() {
    running = true;
    updateButtons();
    if (loopActive) {
        return;
    }
    loopActive = true;
    while (running && state.running) {
        const started = performance.now();
        const speed = SPEEDS[byId("speed").value];
        const batch = Math.max(1, Math.round(speed / 20));
        try {
            await stepSimulation(batch);
        } catch (error) {
            showError(error);
            break;
        }
        const wait = (batch * 1000) / speed - (performance.now() - started);
        await new Promise((resolve) => setTimeout(resolve, Math.max(0, wait)));
    }
    loopActive = false;
    running = false;
    updateButtons();
}

function pause() {
    running = false;
    updateButtons();
}

function updateButtons() {
    const ended = state && !state.running;
    byId("run-button").textContent = running ? "❚❚ Pause" : "▶ Run";
    byId("run-button").disabled = ended;
    byId("step-button").disabled = ended || running;
    byId("speed-label").textContent = `${SPEEDS[byId("speed").value]} steps/s`;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function render() {
    byId("timestep").textContent = state.timestep;
    byId("run-label").textContent = `${state.world} · ${state.environment}`;
    byId("ended-banner").hidden = state.running;
    const chosen = byId("world").value !== state.world || byId("environment").value !== state.environment;
    byId("reset-hint").textContent = chosen ? "Press Reset to apply" : "";
    updateButtons();
    renderStats();
    drawGrid();
    renderTraits();
    renderInspector();
    renderEvents();
    drawCharts();
    renderParameters();
}

// A canvas has its own pixel size, separate from its size on the page.
// Matching it to the screen (times devicePixelRatio) keeps drawings sharp.
function prepareCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, width, height };
}

// Blend from the pale "empty patch" green to the deep "full patch" green.
function foodColor(fraction) {
    const f = Math.max(0, Math.min(1, fraction));
    const mixed = FOOD_EMPTY.map((empty, index) => Math.round(empty + (FOOD_FULL[index] - empty) * f));
    return `rgb(${mixed.join(",")})`;
}

// Where to put each agent inside a cell. One agent sits in the middle; more
// agents are arranged in a 2×2 or 3×3 pattern. Returns the centers and radius.
function slotsInCell(count, cellSize) {
    const across = count === 1 ? 1 : count <= 4 ? 2 : 3;
    const slotSize = cellSize / across;
    const radius = Math.min(slotSize * 0.36, 26);
    const slots = [];
    for (let index = 0; index < Math.min(count, across * across); index++) {
        const row = Math.floor(index / across);
        const column = index % across;
        slots.push([(column + 0.5) * slotSize, (row + 0.5) * slotSize]);
    }
    return { slots, radius };
}

function drawGrid() {
    const canvas = byId("grid");
    const { context, width } = prepareCanvas(canvas);
    const size = state.grid_size;
    const cell = width / size;
    const large = cell >= 70; // Big cells (SMALL world) have room for text labels.

    // 1. Ground and food.
    for (let row = 0; row < size; row++) {
        for (let column = 0; column < size; column++) {
            const capacity = state.capacity[row][column];
            context.fillStyle = capacity > 0 ? foodColor(state.food[row][column] / capacity) : GROUND;
            context.fillRect(column * cell, row * cell, cell, cell);
            if (large && capacity > 0) {
                context.fillStyle = state.food[row][column] / capacity > 0.55 ? "white" : "#4a5a40";
                context.font = "12px sans-serif";
                context.fillText(`food ${format(state.food[row][column])} / ${format(capacity, 1)}`,
                    column * cell + 6, row * cell + 16);
            }
        }
    }

    // 2. Thin grid lines, skipped when cells are too small to need them.
    if (cell >= 8) {
        context.strokeStyle = "rgba(0, 0, 0, 0.07)";
        context.lineWidth = 1;
        context.beginPath();
        for (let index = 0; index <= size; index++) {
            context.moveTo(index * cell, 0);
            context.lineTo(index * cell, width);
            context.moveTo(0, index * cell);
            context.lineTo(width, index * cell);
        }
        context.stroke();
    }

    // 3. Group agents by cell so agents sharing a cell do not hide each other.
    cellAgents = {};
    for (const agent of state.agents) {
        const key = agent.position.join(",");
        if (!cellAgents[key]) {
            cellAgents[key] = [];
        }
        cellAgents[key].push(agent);
    }

    // 4. Agents. Low resources make the dot fade; full strength means the
    //    agent has enough resources to reproduce.
    for (const key in cellAgents) {
        const [row, column] = key.split(",").map(Number);
        const agents = cellAgents[key];
        const { slots, radius } = slotsInCell(agents.length, cell);
        const extra = agents.length - slots.length;
        slots.forEach(([x, y], index) => {
            const centerX = column * cell + x;
            const centerY = row * cell + y + (large ? 8 : 0);
            // When the cell is too crowded, the last slot shows "+n" instead of an agent.
            if (extra > 0 && index === slots.length - 1) {
                context.fillStyle = "#1f2328";
                context.font = `bold ${Math.max(8, radius * 1.1)}px sans-serif`;
                context.textAlign = "center";
                context.textBaseline = "middle";
                context.fillText(`+${extra + 1}`, centerX, centerY);
                context.textAlign = "start";
                context.textBaseline = "alphabetic";
                return;
            }
            drawAgent(context, agents[index], centerX, centerY, radius, large);
        });
    }

    // 5. A red × where an agent died during the latest timestep.
    context.strokeStyle = "#b3261e";
    context.lineWidth = Math.max(1.5, cell * 0.06);
    for (const event of events) {
        if (event.kind === "death" && event.timestep === state.timestep) {
            const [row, column] = event.position;
            const arm = Math.min(cell * 0.18, 10);
            const x = (column + 0.85) * cell - arm;
            const y = (row + 0.15) * cell + arm;
            context.beginPath();
            context.moveTo(x - arm, y - arm);
            context.lineTo(x + arm, y + arm);
            context.moveTo(x + arm, y - arm);
            context.lineTo(x - arm, y + arm);
            context.stroke();
        }
    }

    // 6. Outline the clicked cell.
    if (selectedCell) {
        context.strokeStyle = "#1f2328";
        context.lineWidth = 2;
        context.strokeRect(selectedCell[1] * cell + 1, selectedCell[0] * cell + 1, cell - 2, cell - 2);
    }
}

function drawAgent(context, agent, x, y, radius, withLabel) {
    const strength = Math.min(1, agent.resources / state.stats.reproduction_threshold);
    context.globalAlpha = 0.35 + 0.65 * strength;
    context.fillStyle = COLORS[agent.population];
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
    context.globalAlpha = 1;

    // Newborns (age 0) get a gold ring.
    if (agent.age === 0) {
        context.strokeStyle = "#e3a700";
        context.lineWidth = Math.max(1.5, radius * 0.3);
        context.beginPath();
        context.arc(x, y, radius + context.lineWidth, 0, Math.PI * 2);
        context.stroke();
    }
    // The inspected agent gets a thick dark ring.
    if (agent.id === selectedId) {
        context.strokeStyle = "#1f2328";
        context.lineWidth = Math.max(2, radius * 0.3);
        context.beginPath();
        context.arc(x, y, radius + context.lineWidth * 1.6, 0, Math.PI * 2);
        context.stroke();
    }
    if (withLabel) {
        context.fillStyle = "white";
        context.font = `bold ${Math.round(radius * 0.62)}px sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(agent.name, x, y);
        context.textAlign = "start";
        context.textBaseline = "alphabetic";
    }
}

// Value of a DataCollector column at the latest timestep.
function latest(column) {
    const values = history[column] || [];
    return values.length ? values[values.length - 1] : 0;
}

function renderStats() {
    const stats = state.stats;
    const tiles = [
        ["Population A", stats.population_a, "alive", "A"],
        ["Population B", stats.population_b, "alive", "B"],
        ["Highest generation", stats.highest_generation, `average ${format(stats.average_generation, 1)}`],
        ["Births", stats.cumulative_births, `+${stats.births_this_step} this step`],
        ["Deaths", stats.starvation_deaths + stats.old_age_deaths,
            `${stats.starvation_deaths} starved · ${stats.old_age_deaths} old age`],
        ["Encounters", stats.cumulative_encounters, `+${stats.encounters_per_timestep} this step`],
        ["Food on grid", format(stats.environmental_resources, 1), "units in all patches"],
        ["Avg resources", format(stats.average_resources), "per living agent"],
        ["Agents alive", stats.population_a + stats.population_b, "both populations"],
    ];
    byId("stats").innerHTML = tiles.map(([label, value, detail, population]) => `
        <div>
            <div class="stat-label">${label}</div>
            <div class="stat-value ${population || ""}">${value}</div>
            <div class="stat-detail">${detail}</div>
        </div>`).join("");
}

// A bar for a value between 0 and 1, with an optional mark (e.g. the population average).
function traitBar(value, population, mark) {
    const markHtml = mark === undefined ? "" : `<span class="bar-mark" style="left: ${mark * 100}%"></span>`;
    return `<div class="bar"><span class="bar-fill ${population}" style="width: ${value * 100}%"></span>${markHtml}</div>`;
}

function renderTraits() {
    let html = "";
    for (const [trait, label] of TRAITS) {
        html += `<div class="trait-group"><div class="trait-group-title">${label}</div>`;
        for (const population of ["A", "B"]) {
            const value = latest(`Population ${population} average ${trait}`);
            html += `<div class="trait-row">
                <span class="trait-name">Population ${population}</span>
                ${traitBar(value, population)}
                <span class="trait-value">${format(value)}</span>
            </div>`;
        }
        html += "</div>";
    }
    byId("traits").innerHTML = html;
}

function renderInspector() {
    const agent = state.selected;
    const box = byId("inspector");
    if (!agent) {
        box.innerHTML = `<p class="hint">Click an agent on the grid.</p>`;
        return;
    }
    let parents = "Initial agent";
    if (agent.parents.length) {
        parents = agent.parents.map((parent) =>
            `<button class="link" data-agent="${parent.id}">${escapeHtml(parent.name)}</button>` +
            (parent.alive ? "" : " (dead)")).join(" + ");
    }
    let genome = "";
    for (const [trait, label] of TRAITS) {
        const value = agent.genome[trait];
        const average = latest(`Population ${agent.population} average ${trait}`);
        genome += `<div class="trait-row">
            <span class="trait-name">${label}</span>
            ${traitBar(value, agent.population, average)}
            <span class="trait-value">${format(value)}</span>
        </div>`;
    }
    // Other agents in the same cell, so crowded cells can be explored.
    let cellmates = "";
    if (agent.alive) {
        const others = (cellAgents[agent.position.join(",")] || []).filter((other) => other.id !== agent.id);
        if (others.length) {
            cellmates = `<div class="cellmates">Also in this cell: ` + others.map((other) =>
                `<button data-agent="${other.id}" style="color: ${COLORS[other.population]}">${escapeHtml(other.name)}</button>`).join("") + "</div>";
        }
    }
    box.innerHTML = `
        <div class="inspector-title">
            <strong>${escapeHtml(agent.name)}</strong>
            <span class="badge ${agent.population}">Population ${agent.population}</span>
            ${agent.alive ? "" : `<span class="badge dead">${escapeHtml(agent.status)}</span>`}
        </div>
        <dl class="facts">
            <dt>ID</dt><dd>${agent.id}</dd>
            <dt>Position</dt><dd>(${agent.position.join(", ")})</dd>
            <dt>Age</dt><dd>${agent.age} / ${state.parameters["Maximum age"]}</dd>
            <dt>Generation</dt><dd>${agent.generation}</dd>
            <dt>Resources</dt><dd>${format(agent.resources)}</dd>
            <dt>Parents</dt><dd>${parents}</dd>
            <dt>Reproduction</dt><dd>${escapeHtml(agent.reproduction_status)}</dd>
        </dl>
        <div class="trait-group-title">Genome <span class="hint">(dark mark = population average)</span></div>
        ${genome}
        <div class="last-action"><strong>Last turn:</strong> ${escapeHtml(agent.last_action)}</div>
        ${cellmates}`;
}

function renderEvents() {
    const shown = {};
    document.querySelectorAll(".filters input").forEach((box) => { shown[box.dataset.kind] = box.checked; });
    shown.extinction = true;
    const visible = events.filter((event) => shown[event.kind]).slice(-150).reverse();
    if (!visible.length) {
        byId("events").innerHTML = `<li class="hint">No events yet.</li>`;
        return;
    }
    byId("events").innerHTML = visible.map((event) => `
        <li class="${event.kind}" data-agent="${event.agent_ids.length ? event.agent_ids[0] : ""}">
            <span class="event-time">t${event.timestep}</span>
            <span class="event-kind ${event.kind}"></span>
            <span>${escapeHtml(event.text)}</span>
        </li>`).join("");
}

function renderParameters() {
    let html = "";
    for (const name in state.parameters) {
        html += `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(state.parameters[name])}</td></tr>`;
    }
    byId("parameters").innerHTML = html;
}

// A simple line chart. "lines" is a list of {values, color}. The y axis runs
// from 0 to yMax (or to the largest value when yMax is not given).
function drawChart(canvas, lines, yMax) {
    const { context, width, height } = prepareCanvas(canvas);
    const left = 34, right = 8, top = 6, bottom = 16;
    const count = Math.max(...lines.map((line) => line.values.length));
    if (count < 2) {
        return;
    }
    if (yMax === undefined) {
        yMax = Math.max(1, ...lines.map((line) => Math.max(...line.values))) * 1.1;
    }
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const toX = (index) => left + (index / (count - 1)) * plotWidth;
    const toY = (value) => top + plotHeight - (value / yMax) * plotHeight;

    // Axis labels and three light horizontal guide lines.
    context.font = "10px sans-serif";
    context.fillStyle = "#667080";
    context.strokeStyle = "#e2e5ea";
    context.lineWidth = 1;
    for (const fraction of [0, 0.5, 1]) {
        const y = toY(yMax * fraction);
        context.beginPath();
        context.moveTo(left, y);
        context.lineTo(width - right, y);
        context.stroke();
        const label = yMax <= 1 ? format(yMax * fraction, 1) : Math.round(yMax * fraction);
        context.fillText(label, 2, y + 3);
    }
    context.fillText("0", left, height - 3);
    context.textAlign = "right";
    context.fillText(`t${count - 1}`, width - right, height - 3);
    context.textAlign = "start";

    // Very long runs have more points than pixels, so skip some points.
    const every = Math.max(1, Math.floor(count / plotWidth));
    for (const line of lines) {
        context.strokeStyle = line.color;
        context.lineWidth = 1.8;
        context.beginPath();
        line.values.forEach((value, index) => {
            if (index % every !== 0 && index !== line.values.length - 1) {
                return;
            }
            if (index === 0) {
                context.moveTo(toX(index), toY(value));
            } else {
                context.lineTo(toX(index), toY(value));
            }
        });
        context.stroke();
    }
}

function drawCharts() {
    drawChart(byId("chart-population"), [
        { values: history["Population A"] || [], color: COLORS.A },
        { values: history["Population B"] || [], color: COLORS.B },
    ]);
    drawChart(byId("chart-food"), [
        { values: history["Environmental resources"] || [], color: "#3d8b3d" },
    ]);
    for (const [trait] of TRAITS) {
        drawChart(byId(`chart-${trait}`), [
            { values: history[`Population A average ${trait}`] || [], color: COLORS.A },
            { values: history[`Population B average ${trait}`] || [], color: COLORS.B },
        ], 1);
    }
}

// ---------------------------------------------------------------------------
// Clicking and hovering
// ---------------------------------------------------------------------------

// Convert a mouse position on the canvas into a [row, column] grid cell.
function cellAt(mouseEvent) {
    const canvas = byId("grid");
    const box = canvas.getBoundingClientRect();
    const cell = box.width / state.grid_size;
    const column = Math.floor((mouseEvent.clientX - box.left) / cell);
    const row = Math.floor((mouseEvent.clientY - box.top) / cell);
    if (row < 0 || column < 0 || row >= state.grid_size || column >= state.grid_size) {
        return null;
    }
    return [row, column];
}

function selectAgent(id) {
    selectedId = id;
    loadState().catch(showError);
}

function onGridClick(mouseEvent) {
    const position = cellAt(mouseEvent);
    if (!position) {
        return;
    }
    const agents = cellAgents[position.join(",")] || [];
    if (!agents.length) {
        selectedCell = position;
        selectAgent(null);
        return;
    }
    // Clicking the same cell again moves on to the next agent in that cell.
    let index = agents.findIndex((agent) => agent.id === selectedId);
    index = index === -1 ? 0 : (index + 1) % agents.length;
    selectedCell = position;
    selectAgent(agents[index].id);
}

function onGridHover(mouseEvent) {
    const tooltip = byId("tooltip");
    const position = state ? cellAt(mouseEvent) : null;
    if (!position) {
        tooltip.hidden = true;
        return;
    }
    const [row, column] = position;
    const agents = cellAgents[position.join(",")] || [];
    let names = agents.slice(0, 6).map((agent) => agent.name).join(", ");
    if (agents.length > 6) {
        names += ` +${agents.length - 6} more`;
    }
    const capacity = state.capacity[row][column];
    const food = capacity > 0 ? `food ${format(state.food[row][column])} / ${format(capacity, 1)}` : "no food grows";
    tooltip.innerHTML = `(${row}, ${column}) · ${food}${names ? "<br>" + escapeHtml(names) : ""}`;
    const box = byId("grid").getBoundingClientRect();
    tooltip.style.left = `${mouseEvent.clientX - box.left + 14}px`;
    tooltip.style.top = `${mouseEvent.clientY - box.top + 14}px`;
    tooltip.hidden = false;
}

// ---------------------------------------------------------------------------
// Start-up: connect buttons, then load the current world.
// ---------------------------------------------------------------------------

byId("run-button").addEventListener("click", () => (running ? pause() : run()));
byId("step-button").addEventListener("click", () => stepSimulation(1).catch(showError));
byId("reset-button").addEventListener("click", () => resetSimulation().catch(showError));
byId("speed").addEventListener("input", updateButtons);
byId("world").addEventListener("change", render);
byId("environment").addEventListener("change", render);
byId("grid").addEventListener("click", onGridClick);
byId("grid").addEventListener("mousemove", onGridHover);
byId("grid").addEventListener("mouseleave", () => { byId("tooltip").hidden = true; });
document.querySelectorAll(".filters input").forEach((box) => box.addEventListener("change", renderEvents));

// Clicking an agent name in the inspector or a line in the event log selects that agent.
document.addEventListener("click", (clickEvent) => {
    const target = clickEvent.target.closest("[data-agent]");
    if (target && target.dataset.agent !== "") {
        selectedCell = null;
        selectAgent(Number(target.dataset.agent));
    }
});

// Keyboard shortcuts: space = run/pause, S = one step.
document.addEventListener("keydown", (keyEvent) => {
    if (keyEvent.target.tagName === "SELECT" || keyEvent.target.tagName === "INPUT") {
        return;
    }
    if (keyEvent.code === "Space") {
        keyEvent.preventDefault();
        running ? pause() : run();
    } else if (keyEvent.key === "s" && !running) {
        stepSimulation(1).catch(showError);
    }
});

window.addEventListener("resize", () => { if (state) render(); });

loadState().then(() => {
    document.querySelector('[data-kind="encounter"]').checked = state.world === "SMALL";
    renderEvents();
}).catch(showError);
