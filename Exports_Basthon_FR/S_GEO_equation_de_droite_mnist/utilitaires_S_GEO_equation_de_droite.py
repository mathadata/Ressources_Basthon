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


def tracer_2_points():
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
    
    run_js(f"setTimeout(() => window.mathadata.tracer_2_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

    df = pd.DataFrame()
    labels = ['Point A :', 'Point B :']
    df.index = labels
    #df.index.name = 'Point'
    df['$r$'] = [f'${r[0]}$', f'${r[1]}$']
    df['$x$'] = ['$?$', '$?$']
    df['$y$'] = ['$?$', '$?$']
    display(df)
    return


def tracer_200_points(nb=200):
    id = uuid.uuid4().hex
    display(HTML(f'<canvas id="{id}"></canvas>'))
    
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=common.challenge.d_train[0:nb], r_train=common.challenge.r_train[0:nb])
    params = {
        'points': c_train_par_population,
        'hideClasses': True,
    }
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

def tracer_10_points_droite():
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
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def tracer_points_droite(id=None, input="range", carac=None, initial_hidden=False, save=True):
    if id is None:
        id = uuid.uuid4().hex

    display(HTML(f'''
        <div id="{id}-score-container" style="text-align: center; font-weight: bold; font-size: 2rem; {'' if not initial_hidden else 'display:none;'}">Erreur : <span id="{id}-score">...</span></div>

        <canvas id="{id}-chart"></canvas>

        <div id="{id}-inputs" style="display: flex; gap: 1rem; justify-content: center; flex-direction: {'column' if input == "range" else 'row'}; {'' if not initial_hidden else 'display:none;'}">
            <div>
                <label for="{id}-input-m" id="{id}-label-m"></label>
                <input type="{input}" {input == "range" and 'min="0" max="5"'} value="2" step="0.1" id="{id}-input-m">
            </div>
            <div>
                <label for="{id}-input-p" id="{id}-label-p"></label>
                <input type="{input}" {input == "range" and 'min="-10" max="10"'} value="0" step="0.1" id="{id}-input-p">
            </div>
        </div>
    '''))

    if not initial_hidden:
        if carac is None:
            carac = common.challenge.deux_caracteristiques
        
        c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

        params = {
            'points': c_train_par_population,
            'custom': carac == common.challenge.deux_caracteristiques_custom,
            'hover': True,
            'displayValue': input == "range",
            'save': save
        }
    
        run_js(f"setTimeout(() => window.mathadata.tracer_points_droite('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


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
    
def tracer_point_droite():
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
    
    
def tracer_point_droite():
    ax = tracer_exercice_classification()
    plt.show()
    plt.close()

    
def affichage_zones_custom(A1, B1, A2, B2):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)
    tracer_points_droite(input="number", carac=common.challenge.deux_caracteristiques_custom, save=False)

def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    tracer_points_droite(id=id, input="number", carac=common.challenge.deux_caracteristiques_custom, initial_hidden=True, save=False)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                const params = {{
                    points,
                    custom: true,
                    hover: true,
                    save: false,
                }}
                mathadata.tracer_points_droite('{id}', params)

                // AFFICHER LE GRAPH APRÈS LA SÉLECTION
                document.getElementById('{id}-score-container').style.display = 'block';
                document.getElementById('{id}-inputs').style.display = 'flex';
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
    if(localStorage.getItem('m') !== null && localStorage.getItem('p') !== null) {
        mathadata.run_python(`set_g_mp(${localStorage.getItem('m')}, ${localStorage.getItem('p')})`);
    }

    window.mathadata.tracer_2_points = function(id, params) {
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
                            text: 'Caractéristique x'
                        },
                        min,
                        max,
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Caractéristique y'
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
                    
                    const size = 80;
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
                        window.mathadata.affichage(divId, index == 0 ? d1 : d2);                            

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
                label: `Images de ${hideClasses ? '?' : (index === 0 ? '2' : '7')}`,
                data: set.map(([x, y]) => ({ x, y })),
                backgroundColor: colors[index],
                borderColor: colors[index],
                pointStyle: 'cross',
                pointRadius: 5,
                order: 1,
            }
        });

        if (line) {
            const {m, p} = line;
            datasets.push({
                label: `y = ${m}x ${p < 0 ? '-' : '+'} ${Math.abs(p)}`,
                type: 'line',
                data: [{ x: min, y: min * m + p }, { x: max, y: max * m + p }],
                pointRadius: 0,
                pointHitRadius: 0,
                borderColor: 'black',
                borderWidth: 1,
            });

            datasets.push({
                label: 'Zone de la classe  r^ = 2',
                type: 'line',
                data: [{ x: min, y: min * m + p }, { x: max, y: max * m + p }],
                pointsRadius: 0,
                pointHitRadius: 0,
                borderColor: 'transparent',
                fill: 'end',
                backgroundColor: `rgba(${mathadata.classColorCodes[0]}, 0.1)`,
            });

            datasets.push({
                label: 'Zone de la classe  r^ = 7',
                type: 'line',
                data: [{ x: min, y: min * m + p }, { x: max, y: max * m + p }],
                pointsRadius: 0,
                pointHitRadius: 0,
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
                            text: 'Caractéristique x'
                        },
                        min,
                        max,
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Caractéristique y'
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
                    mathadata.dataTooltip(context, id)
                }
            }
        }
        
        window.mathadata.create_chart(id, chartConfig);
    }

    window.mathadata.tracer_points_droite = function(id, params) {
        mathadata.tracer_points(`${id}-chart`, params);
        if (typeof params === 'string') {
            params = JSON.parse(params);
        }
        const {custom, displayValue, save} = params;
        const chart = window.mathadata.charts[`${id}-chart`]

        chart.data.datasets.push({
            type: 'line',
            data: [],
            borderColor: 'black',
            borderWidth: 1,
            pointRadius: 0,
            pointHitRadius: 0,
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


            const min_x = chart.options.scales.x.min
            const max_x = chart.options.scales.x.max
            
            const data = [{x: min_x, y: m * min_x + p}, {x: max_x, y: m * max_x + p}]
            chart.data.datasets[2].data = data
            chart.data.datasets[2].label = `y = ${m}x ${p < 0 ? '-' : '+'} ${Math.abs(p)}`

            chart.update()

            if (save) {
                localStorage.setItem('m', m);
                localStorage.setItem('p', p);
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

            label_m.innerHTML = `m = ${displayValue ? m : ''}`;
            label_p.innerHTML = `p = ${displayValue ? p : ''}`;
        };

        if (save) {
            // Vérifier si des valeurs de m et p sont stockées dans le localStorage
            const saved_m = localStorage.getItem('m');
            const saved_p = localStorage.getItem('p');

            console.log('Valeur sauvegardée de m:', saved_m);
            console.log('Valeur sauvegardée de p:', saved_p);

            if (saved_m !== null) {
                slider_m.value = Number(saved_m);
            }
            if (saved_p !== null) {
                slider_p.value = Number(saved_p);
            }
        }

        // Fonction pour supprimer tous les écouteurs d'événements en clonant l'élément
        function removeAllEventListeners(element) {
            const clone = element.cloneNode(true);
            element.parentNode.replaceChild(clone, element);
            return clone;
        }

        // Supprimer tous les écouteurs d'événements et ajouter les nouveaux
        slider_m = removeAllEventListeners(slider_m);
        slider_p = removeAllEventListeners(slider_p);

        slider_m.addEventListener("input", update);
        slider_p.addEventListener("input", update);

        // Mise à jour initiale
        update();
    };



       
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

def set_g_mp(m, p):
    global g_m, g_p
    g_m = m
    g_p = p

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
    

def calculer_score_droite():
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
            print_error("Vous pourrez passer à la suite quand vous aurez un pourcentage d'erreur de moins de 8%.")

    calculer_score(algorithme, method="2 moyennes", parameters=f"m={g_m}, p={g_p}", cb=cb) 

def calculer_score_custom_droite():
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
            print_error("Continuez à chercher 2 zones pour avoir moins de 6% d'erreur. Pensez à changer les valeurs de m et p après avoir défini votre zone.")

    calculer_score(algorithme, method="2 moyennes custom", parameters=f"m={g_m}, p={g_p}", cb=cb) 
 
### Validation

def check_coordinates(coords, errors):
    if not (isinstance(coords, tuple)):
        errors.append("Les coordonnées doivent être écrites entre parenthèses séparés par une virgule. Exemple : (3, 5)")
        return False
    if len(coords) != 2:
        errors.append("Les coordonnées doivent être composées de deux valeurs séparés par une virgule. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    if coords[0] is Ellipsis or coords[1] is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if not (isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float))):
        errors.append("Les coordonnées doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
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
            errors.append("Les coordonnées de A ne sont pas correctes. Tu as peut être donné les coordonnées du point B à la place ?")
        elif distARev < 3:
            errors.append("Les coordonnées de A ne sont pas correctes. Attention, la première coordonnée est l'abscisse x et la deuxième l'ordonnée y.")
        else:
            errors.append("Les coordonnées de A ne sont pas correctes.")
    if distB > 3:
        distBRev = np.sqrt((B[1] - B_true[0])**2 + (B[0] - B_true[1])**2)
        distAB = np.sqrt((B[0] - A_true[0])**2 + (B[1] - A_true[1])**2)
        if distAB < 3:
            errors.append("Les coordonnées de B ne sont pas correctes. Tu as peut être donné les coordonnées du point A à la place ?")
        elif distBRev < 3:
            errors.append("Les coordonnées de B ne sont pas correctes. Attention, la première coordonnée est l'abscisse x et la deuxième l'ordonnée y.")
        else:
            errors.append("Les coordonnées de B ne sont pas correctes.")

def function_validation_equation(errors, answers):
    m = g_m
    p = g_p
    abscisse_M = answers['abscisse_M']
    ordonnee_M = answers['ordonnee_M']
    
    if not (isinstance(abscisse_M, (int, float)) and isinstance(ordonnee_M, (int, float))):
        errors.append("Les coordonnées de M doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    
    if abscisse_M != pointA[0]:
        errors.append("L'abscisse x de M n'est pas correcte.")
        return False
    
    if ordonnee_M != m*abscisse_M + p:
        errors.append("L'ordonnée de M n'est pas correcte.")
        return False

    return True

validation_execution_2_points = MathadataValidate(success="")
validation_question_2_points = MathadataValidateVariables({
    'A': None,
    'B': None,
}, function_validation=function_validation_2_points)
validation_execution_200_points = MathadataValidate(success="")
validation_question_couleur = MathadataValidateVariables({
    'classe_points_bleus': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_points_bleus n'a pas la bonne valeur. Vous devez répondre par 2 ou 7."
            }
        ]
    },
    'classe_points_oranges': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_points_oranges n'a pas la bonne valeur. Vous devez répondre par 2 ou 7."
            }
        ]
    }
})

validation_execution_10_points = MathadataValidate(success="")
validation_question_score_droite = MathadataValidateVariables({
    'erreur_10': {
        'value': 20,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 100,
                },
                'else': "Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100."
            },
            {
                'value': 2,
                'if': "Ce n'est pas la bonne valeur. Vous avez donné le nombre d'erreurs et non le pourcentage d'erreur."
            }
        ]
    }
},
    success="C'est la bonne réponse. Il y a un point bleu en dessous de la ligne et un point orange au-dessus, donc deux erreurs, ce qui fait 20%.",
    on_success=lambda answers: set_step(2)
)
validation_execution_tracer_points_droite = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_execution_point_droite = MathadataValidate(success="")
validation_question_equation = MathadataValidateVariables({
    'abscisse_M': 20,
    'ordonnee_M': None
}, function_validation=function_validation_equation, on_success=lambda answers: set_step(3))
validation_score_droite_custom = MathadataValidate(success="Bravo, vous pouvez continuer à essayer d'améliorer votre score. Il est possible de faire seulement 3% d'erreur.")
validation_execution_scatter_caracteristiques_ripou = MathadataValidate(success="")
validation_execution_afficher_customisation = MathadataValidate(success="")
