from IPython.display import display # Pour afficher des DataFrames avec display(df)
import pandas as pd
import os
import sys
# import mplcursors

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common

if not sequence:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))


def plot_2_points():
    deux_caracteristiques = common.challenge.deux_caracteristiques

    data = common.challenge.d_train[0:2]
    r = common.challenge.r_train[0:2]
    c_train = [deux_caracteristiques(d) for d in data]

    id = uuid.uuid4().hex
    display(HTML(f'<canvas id="{id}"></canvas>'))
    
    params = {
        'd1': data[0],
        'd2': data[1],
        'x1': c_train[0][0],
        'y1': c_train[0][1],
        'x2': c_train[1][0],
        'y2': c_train[1][1],
    }
    
    run_js(f"setTimeout(() => window.mathadata.plot_2_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 100)")

    df = pd.DataFrame()
    labels = ['Point A :', 'Point B :']
    df.index = labels
    #df.index.name = 'Point'
    df['$r$'] = [f'${r[0]}$', f'${r[1]}$']
    df['$x$'] = ['$?$', '$?$']
    df['$y$'] = ['$?$', '$?$']
    display(df)
    return


def plot_200_points(nb=200):
    id = uuid.uuid4().hex
    display(HTML(f'<canvas id="{id}"></canvas>'))
    
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=common.challenge.d_train[0:nb], r_train=common.challenge.r_train[0:nb])
    params = {
        'points': c_train_par_population,
        'hideClasses': True,
    }
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 100)")

def plot_10_points_line():
    data = common.challenge.d_train[20:30]
    labels = common.challenge.r_train[20:30]

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=data, r_train=labels)

    id = uuid.uuid4().hex
    display(HTML(f'<canvas id="{id}"></canvas>'))
    
    params = {
        'points': c_train_par_population,
        'line': {
            'm': 0.5,
            'p': 20
        },
        'hover': True
    }
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 100)")


def plot_points_line(id=None, input="range", carac=None):
    if id is None:
        id = uuid.uuid4().hex
    display(HTML(f'''
        <div style="text-align: center; font-weight: bold; font-size: 2rem;">Error: <span id="{id}-score">...</span></div>

        <canvas id="{id}-chart"></canvas>

        <div style="display: flex; gap: 1rem; justify-content: center; flex-direction: {'column' if input == "range" else 'row'}">
            <div>
                <label for="{id}-input-m" id="{id}-label-m"></label>
                <input type="{input}" min="0" max="5" value="2" step="0.1" id="{id}-input-m">
            </div>
            <div>
                <label for="{id}-input-p" id="{id}-label-p"></label>
                <input type="{input}" min="-10" max="10" value="0" step="0.1" id="{id}-input-p">
            </div>
        </div>             
    '''))
    
    if carac is None:
        carac = common.challenge.deux_caracteristiques
    
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

    params = {
        'points': c_train_par_population,
        'custom': carac == common.challenge.deux_caracteristiques_custom,
        'hover': True,
        'displayValue': input == "range"
    }
    
    run_js(f"setTimeout(() => window.mathadata.plot_points_line('{id}', '{json.dumps(params, cls=NpEncoder)}'), 100)")


def create_graph(figsize=(figw_full, figw_full)):
    fig, ax = plt.subplots(figsize=figsize)

    # Enlever les axes de droites et du haut
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Centrer les axes en (0,0)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(("data", 0))

    
    #Afficher les flèches au bout des axes
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)   
    
    # Nom des axex
    ax.set_xlabel('$x$', loc='right')
    ax.set_ylabel('$y$', loc='top', rotation='horizontal')

    return fig, ax


def tracer_droite(ax, m, p, x_min, x_max, color='black'):
    # Ajouter la droite
    x = np.linspace(x_min, x_max, 1000)
    y = m*x + p
    ax.plot(x, y, c=color)  # Ajout de la droite en noir

    # Calculate a point along the line
    x_text = x_max - 2
    y_text = m*x_text + p - 4
    if y_text > x_max - 2:
        y_text = x_max - 2
        x_text = (y_text - p)/m + 4

    # Calculate the angle of the line
    angle = np.arctan(m) * 180 / np.pi

    # Display the equation of the line
    equation = f'$y = {m}x + {p}$'
    ax.text(x_text, y_text, equation, rotation=angle, color=color, verticalalignment='top', horizontalalignment='right')
    
def plot_point_line():
    global g_m, g_p
    deux_caracteristiques = common.challenge.deux_caracteristiques
    m = g_m
    p = g_p

    points = [(20, 40), (35,25)]

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    y += [m * k1 + p for k1 in x]
    x += x

    fig, ax = create_graph()

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    x_min, x_max = min(0, np.min(x) - 2, np.min(y) - 2), max(0, np.max(x) + 2, np.max(y) + 2)
    x_max *= 1.2
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    # Set the ticks on the x-axis at intervals of 5
    ax.set_xticks(np.arange(x_min, x_max, 5))

    # Set the ticks on the y-axis at intervals of 5
    ax.set_yticks(np.arange(x_min, x_max, 5))

    scatter = ax.scatter(x, y, marker = '+', c='black')

    labels = ['A', 'B', 'M', 'N']
    for i in range(len(labels)):
        ax.annotate(labels[i], (x[i] + 0.5, y[i] - 0.5), va='center')

        # Draw a dotted line from the point to the x-axis
        ax.axhline(y[i], xmin=0, xmax=x[i]/x_max, linestyle='dotted', color='gray')

        # Draw a dotted line from the point to the y-axis
        ax.axvline(x[i], ymin=0, ymax=y[i]/x_max, linestyle='dotted', color='gray')

        if i >= len(points):
            # Annotate the y-axis with the y value
            ax.annotate(f"${labels[i]}y = ?$", (0, y[i]), textcoords="offset points", xytext=(-25,0), ha='right', va='center')
        
    tracer_droite(ax, m, p, x_min, x_max)
    
    plt.show()
    plt.close()
    
pointA = (20, 40)

def tracer_exercice_classification(display_M_coords=False):
    global g_m, g_p
    m = g_m
    p = g_p

    x = [pointA[0]]
    y = [pointA[1]]

    y += [m * k1 + p for k1 in x]
    x += x

    fig, ax = create_graph()

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    x_min, x_max = min(0, np.min(x) - 2, np.min(y) - 2), max(0, np.max(x) + 2, np.max(y) + 2)
    x_max *= 1.2
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))
    
    # Set the ticks on the x-axis at intervals of 5
    ax.set_xticks(np.arange(x_min, x_max, 5))

    # Set the ticks on the y-axis at intervals of 5
    #ax.set_yticks(np.arange(x_min, x_max, 5))
    # remove the y axis ticks and labels
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels([])

    Mk2 = m * pointA[0] + p
    
    labels = [f'A({pointA[0]}, {pointA[1]})', f'M({pointA[0]}, {round(Mk2,2)})' if display_M_coords else 'M(?, ?)']
    colors = ['C4', 'C3']
    for i in range(len(labels)):
        # Draw a dotted line from the point to the x-axis
        ax.axhline(y[i], xmin=0, xmax=x[i]/x_max, linestyle='dotted', color='gray')

        # Draw a dotted line from the point to the y-axis
        ax.axvline(x[i], ymin=0, ymax=y[i]/x_max, linestyle='dotted', color='gray')

        ax.annotate(labels[i], (x[i] + 1, y[i]), va='center', color=colors[i])
        ax.scatter(x[i], y[i], marker = '+', c=colors[i])

    tracer_droite(ax, m, p, x_min, x_max, color=colors[1])

    return ax
    
    
def plot_point_line():
    ax = tracer_exercice_classification()
    plt.show()
    plt.close()

    
def display_custom_zones(A1, B1, A2, B2):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)
    plot_points_line(input="number", carac=common.challenge.deux_caracteristiques_custom)

def display_customization():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    plot_points_line(id=id, input="number", carac=common.challenge.deux_caracteristiques_custom)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                const params = {{
                    points,
                    custom: true,
                    hover: true,
                }}
                mathadata.plot_points_line('{id}', params)
            }})
        }}
    ''')

def update_custom():
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques_custom)
    return json.dumps(c_train_par_population, cls=NpEncoder)

def _classification(m, p, c_train):
    r_est_train = np.array([2 if k[1] > m*k[0]+p else 7 for k in c_train])
    return r_est_train
    
def erreur_lineaire(m, p, c_train):
    r_est_train = _classification(m, p, c_train)
    erreurs = (r_est_train != common.challenge.r_train).astype(int)
    return 100*np.mean(erreurs)
 
# JS

run_js('''
    window.mathadata.plot_2_points = function(id, params) {
        params = JSON.parse(params);
        const {d1, d2, x1, y1, x2, y2} = params;

        const chartData = {
            datasets: [{
                label: 'Points',
                data: [
                    { x: x1, y: y1, label: 'A' },
                    { x: x2, y: y2, label: 'B' },
                ],
                backgroundColor: 'black',
                borderColor: 'black',
                pointStyle: 'cross',
                pointRadius: 10,
                pointHoverRadius: 10,
            }]
        };
        
        const min = Math.floor(Math.min(0, x1, x2, y1, y2));
        const max = Math.ceil(Math.max(0, x1 + 1, x2 + 1, y1 + 1, y2 + 1));

        let customDisplaysDone = false;
        
        // Create the Chart.js configuration
        const chartConfig = {
            type: 'scatter',
            data: chartData,
            options: {
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Feature x'
                        },
                        min,
                        max,
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Feature y'
                        },
                        min,
                        max,
                    }
                },
                aspectRatio: 1,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        enabled: false,
                    }
                },
            },
            plugins: [{
                afterDatasetsDraw: function(chart) {
                    const offset = 10
                    const ctx = chart.ctx;
                    chart.data.datasets.forEach((dataset, datasetIndex) => {
                        const meta = chart.getDatasetMeta(datasetIndex);
                        meta.data.forEach((point, index) => {
                            const { x, y } = point.getProps(['x', 'y'], true);
                            const label = dataset.data[index].label;
                            ctx.save();
                            ctx.font = '18px Arial';
                            ctx.fillStyle = 'black';
                            ctx.textAlign = 'center';
                            ctx.fillText(label, x + offset, y - offset);
                            ctx.restore();
                        });
                    });
                },
                afterRender: function(chart, args, options) {
                    if (customDisplaysDone) {
                        return;
                    }
                    
                    const size = 180;
                    const offset = 10
                
                    const chartArea = chart.chartArea;
                    const ctx = chart.ctx;
                    const datasets = chart.data.datasets;

                    const {offsetLeft, offsetTop} = chart.canvas;
                    const meta = chart.getDatasetMeta(0);

                    datasets[0].data.forEach((dataPoint, index) => {
                        const point = meta.data[index];

                        // Create a container div for each data point
                        const containerDiv = document.createElement('div');
                        containerDiv.style.position = 'absolute';
                        containerDiv.style.left = `${offsetLeft + point.x - size - offset}px`;
                        containerDiv.style.top = `${offsetTop + point.y + offset}px`;
                        containerDiv.style.width = `${size}px`;
                        containerDiv.style.height = `${size}px`;
                        
                        const divId = `${id}-data-${index}`;
                        containerDiv.id = divId;

                        // Append the container div to the chart's container
                        const chartContainer = chart.canvas.parentElement;
                        chartContainer.appendChild(containerDiv);

                        // display data in container
                        window.mathadata.display(divId, index == 0 ? d1 : d2);                            

                    });

                    customDisplaysDone = true;
                },
            }]
        };

        window.mathadata.create_chart(id, chartConfig); 
    }
    
    window.mathadata.tracer_points = function(id, params) {
        if (typeof params === 'string') {
            params = JSON.parse(params);
        }
        const {points, line, hover, hideClasses} = params;

        // Calculate the min and max values for the axes
        const allData = points.flat(2);
        const max = Math.ceil(Math.max(...allData, -1) + 1);
        const min = Math.floor(Math.min(...allData, 1) - 1);
        
        // Colors for the populations
        const colors = mathadata.classColorCodes.map(c => `rgba(${c}, ${line ? 1 : 0.5})`);
        
        // Prepare the data for Chart.js
        const datasets = points.map((set, index) => {
            return {
                label: `Images of ${hideClasses ? '?' : (index === 0 ? '2' : '7')}`,
                data: set.map(([x, y]) => ({ x, y })),
                backgroundColor: colors[index],
                borderColor: colors[index],
                pointStyle: 'cross',
                pointRadius: 5
            }
        });

        if (line) {
            const {m, p} = line;
            datasets.push({
                label: `y = ${m}x + ${p}`,
                type: 'line',
                data: [{ x: min, y: min * m + p }, { x: max, y: max * m + p }],
                pointRadius: 0,
                borderColor: 'black',
                borderWidth: 1,
            });

            datasets.push({
                label: 'Area r^ = 2',
                type: 'line',
                data: [{ x: min, y: min * m + p }, { x: max, y: max * m + p }],
                pointsRadius: 0,
                borderColor: 'transparent',
                fill: 'end',
                backgroundColor: `rgba(${mathadata.classColorCodes[0]}, 0.1)`,
            });

            datasets.push({
                label: 'Area r^ = 7',
                type: 'line',
                data: [{ x: min, y: min * m + p }, { x: max, y: max * m + p }],
                pointsRadius: 0,
                borderColor: 'transparent',
                fill: 'origin',
                backgroundColor: `rgba(${mathadata.classColorCodes[1]}, 0.1)`,
            });
                
        }
        
        // Create the Chart.js configuration
        const chartConfig = {
            type: 'scatter',
            data: {
                datasets,
            },
            options: {
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Feature x'
                        },
                        min,
                        max,
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Feature y'
                        },
                        min,
                        max,
                    }
                },
                aspectRatio: 1,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                    }
                },
            },
            /*
            plugins: [{
                afterDatasetsUpdate: function(chart) {
                    console.log('afterDatasetsUpdate');
                    
                    const ctx = chart.ctx;
                    const datasets = chart.data.datasets;

                    // Collect all points with their datasetIndex and dataIndex
                    const points = [];
                    datasets.forEach((dataset, datasetIndex) => {
                        dataset.data.forEach((point, dataIndex) => {
                            points.push({ datasetIndex, dataIndex });
                        });
                    });

                    // Shuffle the points array to randomize render order
                    for (let i = points.length - 1; i > 0; i--) {
                        const j = Math.floor(Math.random() * (i + 1));
                        [points[i], points[j]] = [points[j], points[i]];
                    }

                    // Render points in the randomized order
                    points.forEach(({ datasetIndex, dataIndex }) => {
                        const meta = chart.getDatasetMeta(datasetIndex);
                        const point = meta.data[dataIndex];

                        if (point && !point.hidden) {
                            point.draw(ctx);
                        }
                    });
                }
            }]
            */
        };

        if (hover) {
            const getOrCreateTooltip = (chart) => {
                let tooltipEl = chart.canvas.parentNode.querySelector('div');

                if (!tooltipEl) {
                    tooltipEl = document.createElement('div');
                    tooltipEl.style.background = 'rgba(0, 0, 0, 0.7)';
                    tooltipEl.style.borderRadius = '3px';
                    tooltipEl.style.color = 'white';
                    tooltipEl.style.opacity = 1;
                    tooltipEl.style.pointerEvents = 'none';
                    tooltipEl.style.position = 'absolute';
                    tooltipEl.style.transform = 'translate(-50%, 0)';
                    tooltipEl.style.transition = 'opacity .3s ease';
                    tooltipEl.style.display = 'flex';
                    tooltipEl.style.flexDirection = 'column';
                    tooltipEl.style.gap = '5px';

                    chart.canvas.parentNode.appendChild(tooltipEl);
                }

                return tooltipEl;
            };
        
            chartConfig.options.plugins.tooltip = {
                enabled: false,
                position: 'nearest',
                external: (context) => {
                    // Tooltip Element
                    const {chart, tooltip} = context;
                    const tooltipEl = getOrCreateTooltip(chart);

                    const {xAlign, x, caretX} = tooltip;

                    // Hide if no tooltip
                    if (tooltip.opacity === 0) {
                        tooltipEl.style.opacity = 0;
                        return;
                    }

                    // Set Text
                    if (tooltip.body) {
                        const titleLines = tooltip.title || [];
                        const bodyLines = tooltip.body.map(b => b.lines);

                        tooltipEl.innerHTML = '';

                        titleLines.forEach(title => {
                            const span = document.createElement('span');
                            span.innerText = title;
                            span.style.fontWeight = 'bold';
                            tooltipEl.appendChild(span);
                        });

                        bodyLines.forEach((body, i) => {
                            const colors = tooltip.labelColors[i];
                            const div = document.createElement('div');

                            const span = document.createElement('span');
                            span.style.background = colors.backgroundColor;
                            span.style.borderColor = colors.borderColor;
                            span.style.borderWidth = '2px';
                            span.style.marginRight = '10px';
                            span.style.height = '10px';
                            span.style.width = '10px';
                            span.style.display = 'inline-block';

                            const text = document.createTextNode(body);

                            div.appendChild(span);
                            div.appendChild(text);
                            tooltipEl.appendChild(div);
                        });

                        const dataDiv = document.createElement('div');
                        dataDiv.style.width = '100%';
                        dataDiv.style.aspectRatio = '1';
                        dataDiv.style.textAlign = 'center';
                        
                        dataDiv.innerHTML = "Loading image...";
                        dataDiv.id = `${id}-tooltip-data`;
                        
                        tooltipEl.appendChild(dataDiv);
                        
                        const {dataIndex, datasetIndex} = tooltip.dataPoints[0];
                        const dataClass = datasetIndex === 0 ? 2 : 7;
                        window.mathadata.run_python(`get_data(index=${dataIndex}, dataClass=${dataClass})`, (data) => {
                            window.mathadata.display(`${id}-tooltip-data`, data);
                        });
                    }

                    const {offsetLeft: positionX, offsetTop: positionY} = chart.canvas;

                    // Display, position, and set styles for font
                    tooltipEl.style.opacity = 1;
                    tooltipEl.style.left = positionX + tooltip.caretX + 'px';
                    let translateX = '0';
                    if (tooltip.xAlign === 'center') {
                        translateX = '-50%';
                    } else if (tooltip.xAlign === 'right') {
                        translateX = '-100%';
                    }
                    
                    tooltipEl.style.top = positionY + tooltip.caretY + 'px';
                    let translateY = '0';
                    if (tooltip.yAlign === 'center') {
                        translateY = '-50%';
                    } else if (tooltip.yAlign === 'bottom') {
                        translateY = '-100%';
                    }

                    tooltipEl.style.transform = `translate(${translateX}, ${translateY})`;
                        
                    tooltipEl.style.font = tooltip.options.bodyFont.string;
                    tooltipEl.style.padding = tooltip.options.padding + 'px ' + tooltip.options.padding + 'px';

                    // Set caret Position
                    tooltipEl.classList.remove('above', 'below', 'no-transform');
                    if (tooltip.yAlign) {
                        tooltipEl.classList.add(tooltip.yAlign);
                    } else {
                        tooltipEl.classList.add('no-transform');
                    }
                }
            }
        }
        
        window.mathadata.create_chart(id, chartConfig);
    }

    window.mathadata.plot_points_line = function(id, params) {
        mathadata.tracer_points(`${id}-chart`, params);
        if (typeof params === 'string') {
            params = JSON.parse(params);
        }
        const {custom, displayValue} = params;
        const chart = window.mathadata.charts[`${id}-chart`]

        chart.data.datasets.push({
            type: 'line',
            data: [],
            borderColor: 'black',
            borderWidth: 1,
            pointRadius: 0,
            pointHitRadius: 0,
            zIndex: 10,
            label: 'y = mx + p'
        });
        
        let slider_m = document.getElementById(`${id}-input-m`)
        let slider_p = document.getElementById(`${id}-input-p`)
        const label_m = document.getElementById(`${id}-label-m`)
        const label_p = document.getElementById(`${id}-label-p`)
        const score = document.getElementById(`${id}-score`)

        let exec = null;
        const update = () => {
            const m = parseFloat(slider_m.value);
            const p = parseFloat(slider_p.value);

            if (isNaN(m) || isNaN(p)) {
                return
            }

            const python = `compute_score_json(${m}, ${p}, custom=${custom ? 'True' : 'False'})`
            if (exec) {
                clearTimeout(exec)
            }

            exec = setTimeout(() => {
                mathadata.run_python(python, ({error}) => {
                    if (error > 50) {
                        error = 100 - error
                    }
                    error = Math.round(error * 100) / 100
                    score.innerHTML = `${error}%`
                })
            }, 200)

            const min_x = chart.options.scales.x.min
            const max_x = chart.options.scales.x.max
            
            const data = [{x: min_x, y: m * min_x + p}, {x: max_x, y: m * max_x + p}]
            chart.data.datasets[2].data = data
            chart.data.datasets[2].label = `y = ${m}x + ${p}`

            chart.update()
            
            label_m.innerHTML =`m = ${displayValue ? m : ''}`
            label_p.innerHTML = `p = ${displayValue ? p : ''}`
        }

        // Function to remove all event listeners by cloning the element
        function removeAllEventListeners(element) {
            const clone = element.cloneNode(true);
            element.parentNode.replaceChild(clone, element);
            return clone;
        }

        // Remove all event listeners and add the new ones
        slider_m = removeAllEventListeners(slider_m);
        slider_p = removeAllEventListeners(slider_p);

        slider_m.addEventListener("input", update);
        slider_p.addEventListener("input", update);
        update()
    }
''')

# TODO
# run_js("""
#     // Create a MutationObserver instance
#     const observer = new MutationObserver(function(mutations) {
#         console.log("working")
#         mutations.forEach(function(mutation) {
#             mutation.addedNodes.forEach(function(node) {
#                 console.log("node added")
#                 console.log(node)
#                 // Check if the added node is an element with the specified ID
#                 if (node.id === "container_chart") {
#                     console.log("setup charts")
#                     Jupyter.notebook.kernel.execute("setup_charts()")
#                 } else if (node.id === "container_chart_custom") {
#                     console.log("setup charts 2")
#                     Jupyter.notebook.kernel.execute("setup_charts_2()")
#                 }
#             });
#         });
#     });

#     // Start observing the document for mutations
#     observer.observe(document, { childList: true, subtree: true });
# """)

g_m = None
g_p = None
def compute_score(m, p, custom=False):
    if custom:
        carac = common.challenge.deux_caracteristiques_custom
    else:
        carac = common.challenge.deux_caracteristiques
        
    c_train = compute_c_train(carac, common.challenge.d_train)
    error = erreur_lineaire(m, p, c_train)
    return error

def compute_score_json(m, p, custom=False):
    global g_m, g_p
    g_m = m
    g_p = p
    error = compute_score(m, p, custom=custom)
    return json.dumps({'error': error})
    

def compute_score_line():
    global g_m, g_p
    
    deux_caracteristiques = common.challenge.deux_caracteristiques
    def algorithme(d):
        k1, k2 = deux_caracteristiques(d)
        if k2 > g_m * k1 + g_p:
            return 2
        else:
            return 7
        
    def cb(score):
        if score < 0.08:
            validation_score_droite()
        else:
            print_error("You can move on when you have an error rate of less than 8%.")

    calculer_score(algorithme, method="2 moyennes", parameters=f"m={g_m}, p={g_p}", cb=cb) 

def compute_score_custom_line():
    global g_m, g_p 
    if compute_score(g_m, g_p, custom=True) <= 50:
        above = 2
        below = 7
    else:
        above = 7
        below = 2
    
    deux_caracteristiques = common.challenge.deux_caracteristiques_custom
    def algorithme(d):
        k1, k2 = deux_caracteristiques(d)
        if k2 > g_m * k1 + g_p:
            return above
        else:
            return below
        
    def cb(score):
        if score < 0.06:
            validation_score_droite_custom()
        else:
            print_error("Keep looking for 2 zones to have less than 6% error. Remember to update the values of m and p after defining your zone.")

    calculer_score(algorithme, method="2 moyennes custom", parameters=f"m={g_m}, p={g_p}", cb=cb) 
 
### Validation

def check_coordinates(coords, errors):
    if not (isinstance(coords, tuple)):
        errors.append("Coordinates must be written between parentheses separated by a comma. Example: (3, 5)")
        return False
    if len(coords) != 2:
        errors.append("Coordinates must be composed of two values separated by a comma.")
        return False
    if coords[0] is Ellipsis or coords[1] is Ellipsis:
        errors.append("You did not replace the ...")
        return False
    if not (isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float))):
        errors.append("Coordinates must be numbers.)")
        return False
    return True

def function_validation_2_points(errors, answers):
    A = answers['A']
    B = answers['B']
    if not check_coordinates(A, errors) or not check_coordinates(B, errors):
        return False
    
    deux_caracteristiques = common.challenge.deux_caracteristiques
    
    A_true = deux_caracteristiques(common.challenge.d_train[0])
    B_true = deux_caracteristiques(common.challenge.d_train[1])
    
    distA = np.sqrt((A[0] - A_true[0])**2 + (A[1] - A_true[1])**2)
    distB = np.sqrt((B[0] - B_true[0])**2 + (B[1] - B_true[1])**2)
    
    if distA > 3:
        distARev = np.sqrt((A[1] - A_true[0])**2 + (A[0] - A_true[1])**2)
        distAB = np.sqrt((A[0] - B_true[0])**2 + (A[1] - B_true[1])**2)
        if distAB < 3:
            errors.append("Coordinates of A are not correct. You may have given the coordinates of point B instead?")
        elif distARev < 3:
            errors.append("Coordinates of A are not correct. Be careful, the first coordinate is the x value and the second the y value.")
        else:
            errors.append("Coordinates of A are not correct.")
    if distB > 3:
        distBRev = np.sqrt((B[1] - B_true[0])**2 + (B[0] - B_true[1])**2)
        distAB = np.sqrt((B[0] - A_true[0])**2 + (B[1] - A_true[1])**2)
        if distAB < 3:
            errors.append("Coordinates of B are not correct. You may have given the coordinates of point A instead?")
        elif distBRev < 3:
            errors.append("Coordinates of B are not correct. Be careful, the first coordinate is the x value and the second the y value.")
        else:
            errors.append("Coordinates of B are not correct.")

def function_validation_equation(errors, answers):
    m = g_m
    p = g_p
    x_M = answers['x_M']
    y_M = answers['y_M']
    
    if not (isinstance(x_M, (int, float)) and isinstance(y_M, (int, float))):
        errors.append("Coordinates of M must be numbers.")
        return False
    
    if x_M != pointA[0]:
        errors.append("The x-coordinate of M is not correct.")
        return False
    
    if y_M != m*x_M + p:
        errors.append("The y-coordinate of M is not correct.")
        return False

    return True

validation_execution_2_points = MathadataValidate(success="")
validation_question_2_points = MathadataValidateVariables({
    'A': None,
    'B': None,
}, function_validation=function_validation_2_points)
validation_execution_200_points = MathadataValidate(success="")
validation_question_couleur = MathadataValidateVariables({
    'blue_points_class': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "blue_points_class doesn't have the right value. You must answer by 2 or 7."
            }
        ]
    },
    'orange_points_class': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "orange_points_class doesn't have the right value. You must answer by 2 or 7."
            }
        ]
    }
})
validation_execution_10_points = MathadataValidate(success="")
validation_question_score_droite = MathadataValidateVariables({
    'error_10': {
        'value': 20,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 100,
                },
                'else': "That's not the right value. The percentage error must be between 0 and 100."
            },
            {
                'value': 2,
                'if': "That's not the right value. You gave the number of errors and not the percentage error."
            }
        ]
    }
},
    success="That's the right answer. There is a blue point below the line and an orange point above it, so two errors, which is 20%.",
    on_success=lambda answers: set_step(2)
)
validation_execution_tracer_points_droite = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Well done, you can move on to the next part.")
validation_execution_point_droite = MathadataValidate(success="")
validation_question_equation = MathadataValidateVariables({
    'x_M': 20,
    'y_M': None
}, function_validation=function_validation_equation, on_success=lambda answers: set_step(3))
validation_execution_caracteristiques_custom = MathadataValidate(success="")
validation_score_droite_custom = MathadataValidate(success="Well done, you can continue to keep improving your score. It's possible to get down to 3% error.")
validation_execution_scatter_caracteristiques_ripou = MathadataValidate(success="")
