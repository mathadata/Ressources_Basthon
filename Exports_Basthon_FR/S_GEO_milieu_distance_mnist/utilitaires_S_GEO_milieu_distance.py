from IPython.display import display, IFrame, clear_output # Pour afficher des DataFrames avec display(df)
import pandas as pd
import os
import sys
import numpy as np
import time
#import ipywidgets as widgets
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
        # On hardcode les coordonnées pour que ça tombe juste
        'x1': 15, #c_train[0][0],
        'y1': 20, #c_train[0][1],
        'x2': 25, #c_train[1][0],
        'y2': 30, #c_train[1][1],
    }

    # On stocke en global pour la fonction validation
    global A_true, B_true
    A_true = [params['x1'], params['y1']]
    B_true = [params['x2'], params['y2']]
    
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
        'hideCentroids': True,
    }
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Si on veut utiliser des points existants passer par la méthode mise en commentaires ... (Auguste)

def trouver_point_le_plus_proche(points, coordonnees_cible):
    coordonnees_cible = np.array(coordonnees_cible)
    distances = np.linalg.norm(points - coordonnees_cible, axis=1)
    index_min = np.argmin(distances)
    return points[index_min]

def tracer_6000_points():
    id = uuid.uuid4().hex
    display(HTML(f'<canvas id="{id}"></canvas>'))
    
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=common.challenge.d_train, r_train=common.challenge.r_train)
    # Coordonnées cibles pour A, B, et C
    # coordonnees_A = [32, 75]
    # coordonnees_B = [68, 38]
    # coordonnees_C = [40, 45]

    # Trouver les points les plus proches dans c_train_par_population
    # points_population_1 = c_train_par_population[0]
    # points_population_2 = c_train_par_population[1]

    # point_A = trouver_point_le_plus_proche(points_population_1, coordonnees_A)
    # point_B = trouver_point_le_plus_proche(points_population_1, coordonnees_B)
    # point_C = trouver_point_le_plus_proche(points_population_2, coordonnees_C)
    #print(point_A, point_B, point_C)
    params = {
        'points': c_train_par_population,
        'hideClasses': False,
        'hideCentroids': True,
        'additionalPoints': {
            'A': [27,60],
            'B': [55,32],
            'C': [65,61]    
        },
        'hover': True,
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
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points_2('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def tracer_10_points_centroides():
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
        'segmentHover': True,
    }
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def tracer_points_droite(id=None, input="range", carac=None, initial_hidden=False):
    if id is None:
        id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}-bigBlock" style="{'' if not initial_hidden else 'display:none;'}">
            <div style="text-align: center; font-weight: bold; font-size: 2rem;">Erreur : <span id="{id}-score">...</span></div>

            <canvas id="{id}-chart"></canvas>
            <!--
            <div style="display: flex; gap: 1rem; justify-content: center; flex-direction: {'column' if input == "range" else 'row'}">
                <div>
                    <label for="{id}-input-m" id="{id}-label-m"></label>
                    <input type="{input}" {input == "range" and 'min="0" max="5"'} value="2" step="0.1" id="{id}-input-m">
                </div>
                <div>
                    <label for="{id}-input-p" id="{id}-label-p"></label>
                    <input type="{input}" {input == "range" and 'min="-10" max="10"'} value="0" step="0.1" id="{id}-input-p">
                </div>
            </div>  
            -->           
        </div>
    '''))
    
    if carac is None:
        carac = common.challenge.deux_caracteristiques
    
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)
    #uniquement les points de la classe 2
    groupe_2= [c_train_par_population[0]]
    #uniquement les points de la classe 7
    groupe_7= [c_train_par_population[1]]
    params = {
        'points': c_train_par_population,
        'custom': carac == common.challenge.deux_caracteristiques_custom,
        'hover': True,
        'displayValue': input == "range"
    }
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points_droite('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

def tracer_points_centroides(id=None, carac=None, droite=False, initial_hidden=False):
    if id is None:
        id = uuid.uuid4().hex
    display(HTML(f'''
            {droite and f'<div id="{id}-score-container" style="text-align: center; font-weight: bold; font-size: 2rem; {"display:none;" if initial_hidden else ""}">Erreur : <span id="{id}-score">...</span></div>'}
            <canvas id="{id}-chart"></canvas>
    '''))
    
    if not initial_hidden:
        if carac is None:
            carac = common.challenge.deux_caracteristiques
        
        c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

        params = {
            'points': c_train_par_population,
            'custom': carac == common.challenge.deux_caracteristiques_custom,
            'hover': True,
            'droite': droite,
        }
        
        run_js(f"setTimeout(() => window.mathadata.tracer_points_centroides('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

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
    tracer_points_centroides(carac=common.challenge.deux_caracteristiques_custom,droite=True)

def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    tracer_points_droite(id=id, input="number", carac=common.challenge.deux_caracteristiques_custom, initial_hidden=True)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                const params = {{
                    points,
                    custom: true,
                    hover: true,
                    droite: true, 
                }}
                mathadata.tracer_points_centroides('{id}', params)

                // AFFICHER LE GRAPH APRÈS LA SÉLECTION
                setTimeout(() => {{
                    const bigBlock = document.getElementById('{id}-bigBlock');
                    if (bigBlock) {{
                        bigBlock.style.display = 'block';
                    }}
                }}, 100);
            }})
        }}
    ''')

def afficher_customisation_2():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    tracer_points_centroides(id=id, carac=common.challenge.deux_caracteristiques_custom,droite=True, initial_hidden=True)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                const params = {{
                    points,
                    custom: true,
                    hover: true,
                    droite: true, 
                }}
                mathadata.tracer_points_centroides('{id}', params)

                // AFFICHER LE GRAPH APRÈS LA SÉLECTION
                setTimeout(() => {{
                    const scoreContainer = document.getElementById('{id}-score-container');
                    if (scoreContainer) {{
                        scoreContainer.style.display = 'block';
                    }}
                }}, 100);
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
                        labels: {
                            usePointStyle: true,
                            boxWidth: 10,
                        }
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
    const {points, hideClasses, hideCentroids, additionalPoints, segmentHover, hover} = params;

    // Calculate the min and max values for the axes
    const allData = points.flat(2);
    const max = Math.ceil(Math.max(...allData, -1) + 1);
    const min = Math.floor(Math.min(...allData, 1) - 1);

    // Colors for the populations
    const colors = mathadata.classColorCodes.map(c => `rgba(${c}, ${allData.length > 100 ? 0.5 : 1})`);
    const centroidColors = mathadata.classColorCodes.map(c => `rgba(${c}, 0.5)`);

    // Colors for additional points
    const additionalPointsColors = {
        A: 'rgba(255, 2, 10, 0.5)',
        B: 'rgba(10, 255, 2, 0.5)',
        C: 'rgba(2, 150, 225, 0.5)'
    };

    // Calculate centroids
    const centroids = points.map(set => {
        const xMean = set.reduce((sum, [x]) => sum + x, 0) / set.length;
        const yMean = set.reduce((sum, [, y]) => sum + y, 0) / set.length;
        return { x: xMean, y: yMean };
    });

    // Prepare the data for Chart.js
    const datasets = points.map((set, index) => {
        return {
            label: `Images de ${hideClasses ? '?' : (index === 0 ? '2' : '7')}`,
            data: set.map(([x, y]) => ({ x, y, centroids, classLabel: index === 0 ? '2' : '7' })),
            backgroundColor: colors[index],
            borderColor: colors[index],
            pointStyle: 'cross',
            pointRadius: 5,
            order: 1,
        }
    });
    
    // Add centroids to the dataset
    if (!hideCentroids) {
        centroids.forEach((centroid, index) => {
            datasets.push({
                label: `Point moyen de la classe: ${index === 0 ? '2' : '7'}`,
                data: [centroid],
                backgroundColor: colors[index],
                borderColor: 'black',
                pointStyle: 'circle',
                pointRadius: 6,
                borderWidth: 1,
                order: 0,
            });
        });
    }

    // Add additional points
    if (additionalPoints) {
        Object.entries(additionalPoints).forEach(([label, point]) => {
            datasets.push({
                label: `point ${label}`,
                data: [{ x: point[0], y: point[1], label: label }],
                backgroundColor: additionalPointsColors[label],
                borderColor: 'black',
                pointStyle: 'circle',
                pointRadius: 8,
                borderWidth: 2,
                order: 2,
                isAdditionalPoint: true,
            });
        });
    }

    const chartConfig = {
        type: 'scatter',
        data: { datasets },
        options: {
            scales: {
                x: {
                    title: { display: true, text: 'Caractéristique x' },
                    min, max,
                },
                y: {
                    title: { display: true, text: 'Caractéristique y' },
                    min, max,
                }
            },
            aspectRatio: 1,
            maintainAspectRatio: true,
            interaction: {
                mode: 'nearest',
                axis: 'xy',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        usePointStyle: true,
                        boxWidth: 10,
                        filter: function(legendItem, data) {
                            const dataset = data.datasets[legendItem.datasetIndex];
                            return !dataset.isAdditionalPoint;
                        }
                    }
                },
                tooltip: {
                    enabled: !segmentHover,
                    position: 'nearest',
                    callbacks: {
                        label: function(context) {
                            const dataset = context.dataset;
                            const point = dataset.data[context.dataIndex];
                            if (dataset.label.includes('Point moyen')) {
                                return `Point moyen de : ${dataset.label.split(' ').pop()}`;
                            } else if (dataset.isAdditionalPoint) {
                                return `Point ${point.label} \n Quelle est ma classe ?`;
                            } else {
                                if (hideClasses) {
                                    return
                                }
                                if (hideCentroids) {
                                    return `Image de: ${point.classLabel}`;
                                }
                                const distanceTo2 = Math.sqrt((point.x - centroids[0].x) ** 2 + (point.y - centroids[0].y) ** 2).toFixed(2);
                                const distanceTo7 = Math.sqrt((point.x - centroids[1].x) ** 2 + (point.y - centroids[1].y) ** 2).toFixed(2);
                                return `Image de: ${point.classLabel} \n Distance au point moyen 2: ${distanceTo2} \n Distance au point moyen 7: ${distanceTo7}`;
                            }
                        }
                    }
                }
            },
            onHover: (event, chartElements) => {
                if (segmentHover) {
                    const chart = event.chart;

                    // Initialize trackers on the chart object if they don't exist
                    if (typeof chart.lastHoveredElementKey === 'undefined') {
                        chart.lastHoveredElementKey = null;
                    }
                    if (typeof chart.customHoverDetails === 'undefined') {
                        chart.customHoverDetails = null;
                    }

                    if (chartElements && chartElements.length > 0) {
                        const activeElement = chartElements[0];
                        const datasetIndex = activeElement.datasetIndex;
                        const index = activeElement.index;
                        const currentHoveredElementKey = `${datasetIndex}-${index}`;

                        if (chart.lastHoveredElementKey !== currentHoveredElementKey) {
                            chart.lastHoveredElementKey = currentHoveredElementKey;

                            const hoveredElement = activeElement.element;
                            const hoveredPixelX = hoveredElement.x;
                            const hoveredPixelY = hoveredElement.y;
                            const dataHoveredX = chart.scales.x.getValueForPixel(hoveredPixelX);
                            const dataHoveredY = chart.scales.y.getValueForPixel(hoveredPixelY);
                            
                            // Store details for afterDraw to use
                            chart.customHoverDetails = {
                                hoveredPixelX,
                                hoveredPixelY,
                                dataHoveredX,
                                dataHoveredY,
                            };
                            chart.draw(); // Trigger redraw for afterDraw
                        }
                        // If key is the same, do nothing, afterDraw will use existing details
                    } else {
                        // No element is being hovered
                        if (chart.lastHoveredElementKey !== null) {
                            chart.lastHoveredElementKey = null;
                            chart.customHoverDetails = null; // Clear details
                            chart.draw(); // Trigger redraw for afterDraw to clear segments
                        }
                    }
                }
            },
        },
        plugins: [{
            afterDraw: function(chart) {
                const ctx = chart.ctx;

                // Part 1: Draw segments and distances if customHoverDetails exists
                if (chart.customHoverDetails && segmentHover) { // check segmentHover again
                    const { 
                        hoveredPixelX, hoveredPixelY, 
                        dataHoveredX, dataHoveredY, 
                    } = chart.customHoverDetails;

                    ctx.save();
                    centroids.forEach((centroid, idx) => {
                        const centroidPixelX = chart.scales.x.getPixelForValue(centroid.x);
                        const centroidPixelY = chart.scales.y.getPixelForValue(centroid.y);

                        // Draw segment
                        ctx.beginPath();
                        ctx.setLineDash([5, 5]);
                        ctx.moveTo(hoveredPixelX, hoveredPixelY);
                        ctx.lineTo(centroidPixelX, centroidPixelY);
                        ctx.strokeStyle = colors[idx];
                        ctx.lineWidth = 1.5;
                        ctx.stroke();

                        // Calculate distance
                        const distance = Math.sqrt(Math.pow(dataHoveredX - centroid.x, 2) + Math.pow(dataHoveredY - centroid.y, 2));
                        const distanceText = distance.toFixed(2);

                        // Calculate midpoint for text
                        const midSegmentX = (hoveredPixelX + centroidPixelX) / 2;
                        const midSegmentY = (hoveredPixelY + centroidPixelY) / 2;
                        
                        ctx.font = '12px Arial';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle'; // Align text vertically to center for background

                        // Text dimensions for background
                        const textMetrics = ctx.measureText(distanceText);
                        const textWidth = textMetrics.width;
                        const textHeight = parseInt(ctx.font, 10); // Approximate height
                        const padding = 3;

                        let textPosX = midSegmentX + (idx % 2 === 0 ? 20 : -20); // Adjusted base offset X
                        let textPosY = midSegmentY + (idx % 2 === 0 ? -10 : 10); // Adjusted base offset Y

                        // Draw background for text
                        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                        ctx.fillRect(
                            textPosX - textWidth / 2 - padding,
                            textPosY - textHeight / 2 - padding,
                            textWidth + padding * 2,
                            textHeight + padding * 2
                        );
                        
                        // Draw text
                        ctx.fillStyle = colors[idx];
                        ctx.fillText(distanceText, textPosX, textPosY);
                    });
                    ctx.restore();
                }
                
                // Part 2: Existing afterDraw logic for labels A/B/C and centroid labels
                // Dessiner les labels A/B/C
                chart.data.datasets.forEach(dataset => {
                    if (!dataset.isAdditionalPoint) return;
                    
                    const meta = chart.getDatasetMeta(chart.data.datasets.indexOf(dataset));
                    const point = meta.data[0];
                    const label = dataset.data[0].label;
                    
                    ctx.save();
                    ctx.font = 'bold 14px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'bottom';
                    ctx.fillStyle = 'black';
                    ctx.fillText(label, point.x, point.y - 10); // Increased Y offset from -8
                    ctx.restore();
                });
                
                // Dessiner les labels des centroïdes 
                if (!hideCentroids) {
                    centroids.forEach((centroid, index) => {
                        const datasetIndex = points.length + index;
                        if (chart.isDatasetVisible(datasetIndex)) {
                            const meta = chart.getDatasetMeta(datasetIndex);
                            const element = meta.data[0];
                            
                            ctx.save();
                            ctx.font = 'bold 12px Arial';
                            ctx.fillStyle = 'black';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(`point moyen de ${index === 0 ? '2' : '7'}`, element.x + 18, element.y - 3); // Adjusted X and Y offset
                            ctx.restore();
                        }
                    });
                }
            }
        }]
    };

    if (hover) {
        chartConfig.options.plugins.tooltip = {
            enabled: false,
            position: 'nearest',
            external: (context) => {
                mathadata.dataTooltip(context, id)
            }
        }
    }

    window.mathadata.create_chart(id, chartConfig);
};

window.mathadata.tracer_points_droite = function(id, params) {
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
            if (!isFinite(m) || !isFinite(p)) {
                console.error("Erreur : valeur infinie détectée !");
                return;
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

            const min_x = chart.options.scales.x?.min ?? -10;  // Valeur par défaut si undefined
            const max_x = chart.options.scales.x?.max ?? 10;

            let y_min = m * min_x + p;
            let y_max = m * max_x + p;

            // Vérifier si les valeurs sont valides
            if (!isFinite(y_min) || !isFinite(y_max)) {
                console.error("Valeur infinie détectée dans le tracé de la droite !");
                return;
            }

            const data = [{x: min_x, y: y_min}, {x: max_x, y: y_max}];
            chart.data.datasets[2].data = data;
            chart.data.datasets[2].label = `y = ${m}x ${p < 0 ? '-' : '+'} ${Math.abs(p)}`

            chart.update()
            
            label_m.innerHTML =`m = ${displayValue ? m : ''}`
            label_p.innerHTML = `p = ${displayValue ? p : ''}`
        }

        // Function to remove all event listeners by cloning the element
        /*function removeAllEventListeners(element) {
            const clone = element.cloneNode(true);
            element.parentNode.replaceChild(clone, element);
            return clone;
        }*/

        // Remove all event listeners and add the new ones
        slider_m = removeAllEventListeners(slider_m);
        slider_p = removeAllEventListeners(slider_p);

        slider_m.addEventListener("input", update);
        slider_p.addEventListener("input", update);
        update()
    }

       
window.mathadata.tracer_points_2 = function(id, params) {
    if (typeof params === 'string') {
        params = JSON.parse(params);
    }
    const { points, hover, hideClasses } = params;

    const pointsde2 = points[0];
    const pointsde7 = points[1];

    function calculerCentroide(points) {
        const n = points.length;
        if (n === 0) return null;

        let sommeX = 0;
        let sommeY = 0;

        for (const p of points) {
            if (Array.isArray(p)) {
                // Cas où les points sont sous forme [x, y]
                sommeX += p[0];
                sommeY += p[1];
            } else if (typeof p === 'object' && 'x' in p && 'y' in p) {
                // Cas où les points sont sous forme { x, y }
                sommeX += p.x;
                sommeY += p.y;
            } else {
                console.warn("Point au format inattendu :", p);
                return null; // Empêcher des calculs avec des données corrompues
            }
        }

        return [sommeX / n, sommeY / n];
    }



    const centroide2 = calculerCentroide(pointsde2);
    const centroide7 = calculerCentroide(pointsde7);

    //console.log("Point moyen de la classe 2:", centroide2);
    //console.log("Point moyen de la classe 7:", centroide7);

    const allData = points.flat(2);
    const max = Math.ceil(Math.max(...allData, -1) + 1);
    const min = Math.floor(Math.min(...allData, 1) - 1);

    const colors = mathadata.classColorCodes.map(c => `rgba(${c}, 1)`);

    const datasets = points.map((set, index) => {
        return {
            label: `Images de ${hideClasses ? '?' : (index === 0 ? '2' : '7')}`,
            data: set.map(([x, y]) => ({ x, y })),
            backgroundColor: colors[index],
            borderColor: colors[index],
            pointStyle: 'cross',
            pointRadius: 8,
            order: 1,
        };
    });

    if (centroide2) {
        datasets.push({
            label: 'Point moyen de la classe 2',
            data: [{ x: centroide2[0], y: centroide2[1] }],
            backgroundColor: colors[0],
            borderColor: 'black',
            pointStyle: 'circle',
            pointRadius: 6,
            order: 0,
            dragData: false, // Disable dragging for centroids
        });
    }
    if (centroide7) {
        datasets.push({
            label: 'Point moyen de la classe 7',
            data: [{ x: centroide7[0], y: centroide7[1] }],
            backgroundColor: colors[1],
            borderColor: 'black',
            pointStyle: 'circle',
            pointRadius: 6,
            order: 0,
            dragData: false, // Disable dragging for centroids
        });
    }

    if (centroide2 && centroide7) {
        const midX = (centroide2[0] + centroide7[0]) / 2;
        const midY = (centroide2[1] + centroide7[1]) / 2;
        const deltaX = centroide7[0] - centroide2[0];
        const deltaY = centroide7[1] - centroide2[1];

        if (deltaX !== 0) {
            const slope = -deltaX / deltaY;
            const y1 = midY + slope * (min - midX);
            const y2 = midY + slope * (max - midX);

            datasets.push({
                label: 'Médiatrice',
                type: 'line',
                pointStyle: 'line',
                data: [
                    { x: min, y: y1 },
                    { x: max, y: y2 }
                ],
                borderColor: 'green',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
            });

            datasets.push({
                label: 'Zone supérieure (classe 2)',
                type: 'line',
                pointStyle: 'rect',
                data: [
                    { x: min, y: y1 },
                    { x: max, y: y2 }
                ],
                borderColor: 'transparent',
                fill: 'end',
                backgroundColor: 'rgba(0, 0, 255, 0.1)',
            });

            datasets.push({
                label: 'Zone inférieure (classe 7)',
                type: 'line',
                pointStyle: 'rect',
                data: [
                    { x: min, y: y1 },
                    { x: max, y: y2 }
                ],
                borderColor: 'transparent',
                fill: 'origin',
                backgroundColor: 'rgba(255, 255, 0, 0.1)',
            });
        }
    }

    function recalculerCentroides(chart) {
        const pointsde2 = chart.data.datasets[0].data;
        const pointsde7 = chart.data.datasets[1].data;

        console.log('Points après drag - Classe 2:', pointsde2);
        console.log('Points après drag - Classe 7:', pointsde7);    

        const centroide2 = calculerCentroide(pointsde2);
        const centroide7 = calculerCentroide(pointsde7);

        console.log("Nouveaux point moyens:", centroide2, centroide7);

        chart.data.datasets[2].data = centroide2 ? [{ x: centroide2[0], y: centroide2[1] }] : [];
        chart.data.datasets[3].data = centroide7 ? [{ x: centroide7[0], y: centroide7[1] }] : [];

        if (centroide2 && centroide7) {
            const midX = (centroide2[0] + centroide7[0]) / 2;
            const midY = (centroide2[1] + centroide7[1]) / 2;
            const deltaX = centroide7[0] - centroide2[0];
            const deltaY = centroide7[1] - centroide2[1];

            if (deltaX !== 0) {
                const slope = -deltaX / deltaY;
                const y1 = midY + slope * (chart.options.scales.x.min - midX);
                const y2 = midY + slope * (chart.options.scales.x.max - midX);

                const mediatrice = [
                    { x: chart.options.scales.x.min, y: y1 },
                    { x: chart.options.scales.x.max, y: y2 }
                ];

                chart.data.datasets[4].data = mediatrice;
                chart.data.datasets[5].data = mediatrice;
                chart.data.datasets[6].data = mediatrice;
            }
        }

        chart.update('none');
    }
    function calculerDistance(point, centroide) {
        const dx = point.x - centroide[0];
        const dy = point.y - centroide[1];
        return Math.sqrt(dx * dx + dy * dy).toFixed(2);
    }
       
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
                    labels: {
                    usePointStyle: true, // Utiliser le style de point pour les éléments de la légende
                    boxWidth: 10, // Largeur de la boîte de style de point
                    }
                },
                dragData: {
                    dragX: true,  
                    dragY: true,
                    showTooltip: true,
                    onDrag: function(e, datasetIndex, index, value) {
                        const div_id = id;  // Récupère l'ID du canvas
                        const chart = window.mathadata.charts[div_id];  // Récupère le chart par son ID
                        if (chart) {
                            recalculerCentroides(chart);  // Passe le chart directement à la fonction
                        }
                    }

                },
            },
        }
    };
    //console.log(chartConfig.data.datasets)
    window.mathadata.create_chart(id, chartConfig);
};



       
    function isColorDark(color) {
        const rgb = color.match(/\d+/g);
        const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
        return brightness < 128;
    }

    window.mathadata.tracer_points_centroides = function(id, params) {
        if (typeof params === 'string') {
            params = JSON.parse(params);
        }
        mathadata.tracer_points(`${id}-chart`, params);

        const {points, custom, displayValue, droite} = params;
        const chart = window.mathadata.charts[`${id}-chart`]
        // Compute centroids for each class
        const centroids = points.map(set => {
            // Sum the x and y coordinates for each point in the set
            const [sumX, sumY] = set.reduce(
                ([accX, accY], [x, y]) => [accX + x, accY + y],
                [0, 0]
            );
            // Calculate the average (centroid) for the set
            return { x: sumX / set.length, y: sumY / set.length };
        });
        //console.log(centroids)
        // Push each centroid as its own dataset with its coordinates in the label:
        /*centroids.forEach((centroid, index) => {
            const color = `rgb(${window.mathadata.centroidColorCodes[index]})`
            chart.data.datasets.push({
                type: 'scatter',
                label: `Point moyen de la classe ${mathadata.classes[index]}`,
                data: [centroid],
                backgroundColor: color,
                borderColor: 'black',
                borderWidth: 4,
                pointStyle: 'circle',
                pointRadius: 10,
                pointHoverRadius: 7,
            });
        });
        */
    
        if (droite) {
            // calcul de la bissectrice
            const centroid1 = centroids[0];
            const centroid2 = centroids[1];
            const midX = (centroid1.x + centroid2.x) / 2;
            const midY = (centroid1.y + centroid2.y) / 2;
            const m = -(centroid2.x - centroid1.x) / (centroid2.y - centroid1.y); // Slope of the perpendicular bisector
            const p = midY - m * midX; // y-intercept of the perpendicular bisector
            const data = [{x: chart.options.scales.x.min, y: m * chart.options.scales.x.min + p}, {x: chart.options.scales.x.max, y: m * chart.options.scales.x.max + p}];
            chart.data.datasets.push({
                type: 'line',
                label: `Droite médiatrice y = ${Math.round(m * 100) / 100}x ${p < 0 ? '-' : '+'} ${Math.abs(Math.round(p * 100) / 100)}`,
                data: data,
                borderColor: 'black',
                borderWidth: 1,
                pointStyle: 'line',
                pointRadius: 0,
                pointHitRadius: 0,
            });
            if (!isFinite(m) || !isFinite(p)) {
                console.error("Erreur : valeur infinie détectée !");
                return;
            }
            const python = `compute_score_json(${m}, ${p}, custom=${custom ? 'True' : 'False'})`
            mathadata.run_python(python, ({error}) => {
                if (error > 50) {
                    error = 100 - error
                }
                error = Math.round(error * 100) / 100
                const score = document.getElementById(`${id}-score`)
                score.innerHTML = `${error}%`
            })

        }
        chart.options.plugins.legend = {
            display: true,
            labels: {
                usePointStyle: true,
                boxWidth: 10,
            }
        };
        chart.update();
    }
''')

""" /**
 * Interactive Perpendicular Bisector Visualization
 * 
 * This script creates an interactive Chart.js visualization where a user can move a point
 * to explore the concept of perpendicular bisectors. 
 *
 * Features:
 * - Two fixed points (red, blue) define the perpendicular bisector.
 * - A draggable moving point (green) that updates dynamically.
 * - When the moving point is equidistant from both fixed points, a segment of the bisector is revealed.
 * - The connecting lines glow in the equidistant zone.
 * 
 * How It Works:
 * - The script calculates the distances from the moving point to the two fixed points.
 * - If the point is close to equidistant (within a tolerance), the bisector is progressively drawn.
 * - The visualization is updated smoothly for a natural feel.
 * 
 * Technologies Used:
 * - Chart.js for rendering the scatter plot.
 * - Canvas API for custom line drawing and effects.
 * - Mathematical projection for accurate bisector tracing.
 */

// Global constants
const TOLERANCE = 0.05;
const SEGMENT_RADIUS = 0.005;

// -------------------- 1) DEFINE FIXED POINTS & MOVING POINT --------------------
const p1 = { x: 2, y: 3, color: 'red' };
const p2 = { x: 8, y: 7, color: 'blue' };
let movingPoint = { x: 5, y: 5 };

// Midpoint of p1 and p2
const mid = {
  x: (p1.x + p2.x) / 2,
  y: (p1.y + p2.y) / 2
};

// Direction vector from p1 to p2 and its perpendicular
const dx = p2.x - p1.x;
const dy = p2.y - p1.y;
let vx = dy;
let vy = -dx;

// -------------------- 2) PARAMETRIC REPRESENTATION --------------------
// Bisector line: L(t) = mid + t*(vx, vy)
function projectOntoBisector(px, py) {
  const dot = (px - mid.x) * vx + (py - mid.y) * vy;
  const len2 = vx * vx + vy * vy;
  const t = dot / len2;
  return { projX: mid.x + t * vx, projY: mid.y + t * vy, t };
}

function paramToPoint(t) {
  return { x: mid.x + t * vx, y: mid.y + t * vy };
}

// -------------------- 3) HELPER FUNCTION FOR ALMOST EQUIDISTANCE --------------------
function isAlmostEquidistant(x, y) {
  const dRed = Math.hypot(x - p1.x, y - p1.y);
  const dBlue = Math.hypot(x - p2.x, y - p2.y);
  return {
    almost: Math.abs(dRed - dBlue) < TOLERANCE,
    dRed,
    dBlue
  };
}

// -------------------- 4) MANAGE SMALL INTERVALS ALONG t --------------------
let bisectorIntervals = [];

function addInterval(start, end) {
  if (start > end) [start, end] = [end, start];
  bisectorIntervals.push({ start, end });
  bisectorIntervals = unifyIntervals(bisectorIntervals);
}

function unifyIntervals(intervals) {
  if (intervals.length < 2) return intervals;
  intervals.sort((a, b) => a.start - b.start);
  const result = [];
  let current = intervals[0];
  for (let i = 1; i < intervals.length; i++) {
    const next = intervals[i];
    if (next.start <= current.end) {
      current.end = Math.max(current.end, next.end);
    } else {
      result.push(current);
      current = next;
    }
  }
  result.push(current);
  return result;
}

// -------------------- 5) CHART SETUP --------------------
const ctx = document.getElementById('myChart').getContext('2d');
const scatterChart = new Chart(ctx, {
  type: 'scatter',
  data: {
    datasets: [
      {
        label: 'Fixed Point 1',
        data: [{ x: p1.x, y: p1.y }],
        backgroundColor: p1.color,
        pointRadius: 6
      },
      {
        label: 'Fixed Point 2',
        data: [{ x: p2.x, y: p2.y }],
        backgroundColor: p2.color,
        pointRadius: 6
      },
      {
        label: 'Moving Point',
        data: [{ x: movingPoint.x, y: movingPoint.y }],
        backgroundColor: 'green',
        pointRadius: 8
      }
    ]
  },
  options: {
    animation: false,
    scales: {
      x: { type: 'linear', position: 'bottom', title: { display: true, text: 'X Axis' } },
      y: { type: 'linear', title: { display: true, text: 'Y Axis' } }
    },
    plugins: {
      legend: { 
        display: true,
        labels: {
          usePointStyle: true,
          boxWidth: 10
        }
      },
      tooltip: {
        filter: (item) => item.datasetIndex === 2,
        callbacks: {
          title: (items) => {
            const xVal = items[0].parsed.x.toFixed(2);
            const yVal = items[0].parsed.y.toFixed(2);
            return `Moving Point: (${xVal}, ${yVal})`;
          },
          label: (context) => {
            const { almost, dRed, dBlue } = isAlmostEquidistant(context.parsed.x, context.parsed.y);
            let status;
            if (almost) {
              status = 'Almost Equidistant';
            } else if (dRed < dBlue) {
              status = 'Closer to Red';
            } else {
              status = 'Closer to Blue';
            }
            return [
              `Red: ${dRed.toFixed(2)}`,
              `Blue: ${dBlue.toFixed(2)}`,
              status
            ];
          }
        }
      }
    }
  },
  plugins: [{
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const xAxis = chart.scales.x;
      const yAxis = chart.scales.y;

      // 1) Draw connecting lines from the moving point to each fixed point.
      const mx = xAxis.getPixelForValue(movingPoint.x);
      const my = yAxis.getPixelForValue(movingPoint.y);
      const { almost: almostEquidistant } = isAlmostEquidistant(movingPoint.x, movingPoint.y);

      [p1, p2].forEach(fp => {
        const fpX = xAxis.getPixelForValue(fp.x);
        const fpY = yAxis.getPixelForValue(fp.y);
        ctx.save();
        // Apply glow effect if in the equidistant zone.
        if (almostEquidistant) {
          ctx.shadowColor = 'green';
          ctx.shadowBlur = 15;
        } else {
          ctx.shadowColor = 'transparent';
          ctx.shadowBlur = 0;
        }
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(fpX, fpY);
        ctx.strokeStyle = fp.color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      });

      // 2) Draw the discovered segments of the perpendicular bisector (in green).
      const merged = unifyIntervals(bisectorIntervals);
      ctx.save();
      ctx.beginPath();
      merged.forEach((interval) => {
        const startPt = paramToPoint(interval.start);
        const startPx = xAxis.getPixelForValue(startPt.x);
        const startPy = yAxis.getPixelForValue(startPt.y);
        const endPt = paramToPoint(interval.end);
        const endPx = xAxis.getPixelForValue(endPt.x);
        const endPy = yAxis.getPixelForValue(endPt.y);
        ctx.moveTo(startPx, startPy);
        ctx.lineTo(endPx, endPy);
      });
      ctx.strokeStyle = 'green';
      ctx.lineWidth = 4;
      ctx.stroke();
      ctx.restore();
    }
  }]
});

// -------------------- 6) DRAGGING LOGIC --------------------
let dragging = false;
const canvas = document.getElementById('myChart');

function isNearPoint(mouseX, mouseY) {
  const xPixel = scatterChart.scales.x.getPixelForValue(movingPoint.x);
  const yPixel = scatterChart.scales.y.getPixelForValue(movingPoint.y);
  return Math.hypot(mouseX - xPixel, mouseY - yPixel) < 10;
}

canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  if (isNearPoint(e.clientX - rect.left, e.clientY - rect.top)) {
    dragging = true;
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const rect = canvas.getBoundingClientRect();
  const newX = scatterChart.scales.x.getValueForPixel(e.clientX - rect.left);
  const newY = scatterChart.scales.y.getValueForPixel(e.clientY - rect.top);
  movingPoint.x = newX;
  movingPoint.y = newY;
  scatterChart.data.datasets[2].data[0] = { x: newX, y: newY };

  // When almost equidistant, project and record a small interval along the bisector.
  const { almost } = isAlmostEquidistant(newX, newY);
  if (almost) {
    const { t } = projectOntoBisector(newX, newY);
    addInterval(t - SEGMENT_RADIUS, t + SEGMENT_RADIUS);
  }
  scatterChart.update('none');
});

canvas.addEventListener('mouseup', () => dragging = false);
canvas.addEventListener('mouseleave', () => dragging = false);  """


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
        if score < 0.1:
            validation_score_droite_custom()
        else:
            print_error("Continuez à chercher 2 zones pour avoir moins de 10% d'erreur.")

    calculer_score(algorithme, method="2 moyennes custom", parameters=f"m={g_m}, p={g_p}", cb=cb) 


### Attempting jsxgraph
def _build_srcdoc(points, title, centroid_name='M',
                  width=600, height=500,
                  xmin=-3, xmax=7, ymin=-5, ymax=5):
    """
    Build HTML for an iframe srcdoc showing draggable points and midpoint/centroid.
    """
    # JS to create points
    create_pts = []
    for i,(x,y) in enumerate(points):
        create_pts.append(
            f"var P{i}=board.create('point',[{x},{y}],{{name:'{chr(65+i)}',size:4, snapToGrid: true, snapSizeX: 0.1, snapSizeY: 0.1}});"
        )
    # JS for centroid or midpoint
    n = len(points)
    if n==2:
        centroid_js = "var C=board.create('midpoint',[P0,P1],{name:'%s',size:4,face:'cross'});" % centroid_name
    else:
        sumx = '+'.join([f'P{i}.X()' for i in range(n)])
        sumy = '+'.join([f'P{i}.Y()' for i in range(n)])
        centroid_js = (
            "var C=board.create('point',["
            f"function(){{return({sumx})/{n}}},"
            f"function(){{return({sumy})/{n}}}],{{name:'{centroid_name}',size:4,face:'cross'}});"
        )
    pts_js = '\n          '.join(create_pts + [centroid_js])

    # HTML document
    html = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
  <style>html,body{{margin:0;padding:0;overflow:hidden;font-family:sans-serif;}}
    #title{{text-align:center;font-weight:bold;margin:4px 0;}}
    #coords{{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.6);color:#fff;padding:4px 8px;border-radius:4px;font-size:12px;}}

  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraph.css"/>
  <script src="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraphcore.js"></script>
</head><body>
  <div id="title">{title}</div>
  <div id="board" style="width:{width}px;height:{height}px;"></div>
  <div id="coords">Coords</div>
  <script>
    setTimeout(function(){{
      var board=JXG.JSXGraph.initBoard('board',{{boundingbox:[{xmin},{ymax},{xmax},{ymin}],
        axis:true,showNavigation:false,showCopyright:false}});
      {pts_js}
      function upd(){{document.getElementById('coords').innerText=
            'Coordonnées du point moyen : ('+C.X().toFixed(2)+','+C.Y().toFixed(2)+')';}}
      board.on('update',upd);upd();
    }},50);
  </script>
</body></html>
"""
    # Escape quotes for embedding
    return html.replace('"','&quot;')

# Precompute srcdocs
_SRC_DOCS = {
    '2': _build_srcdoc([(-1,1),(2,-2)], "Point moyen pour 2 points", 'M'),
    '3': _build_srcdoc([(-1,1),(2,-2),(-1,-2)], "Point moyen pour 3 points", 'G'),
    '4': _build_srcdoc([(-1,1),(2,-2),(-1,-2),(2,1)], "Point moyen pour 4 points", 'G'),
}


def afficher_manip_points_moyens():
    """
    Display a dropdown to choose among 2/3/4-point canvases and show the selected iframe.
    """
    # Unique IDs for elements
    sel_id = 'sel_' + uuid.uuid4().hex
    reset_id = 'rst_' + uuid.uuid4().hex
    frame_ids = {n: 'frm_' + uuid.uuid4().hex for n in ['2','3','4']}

    # Prebuilt srcdoc strings
    src2 = _SRC_DOCS['2']
    src3 = _SRC_DOCS['3']
    src4 = _SRC_DOCS['4']

    # HTML with three hidden iframes and controls
    html = f'''
<div style="margin-bottom:8px;">
  <select id="{sel_id}">
    <option value="2">2 Points</option>
    <option value="3">3 Points</option>
    <option value="4">4 Points</option>
  </select>
  <button id="{reset_id}">Reinitialiser</button>
</div>
<div id="canvas_container">
  <iframe id="{frame_ids['2']}" srcdoc="{src2}" width="600" height="450" style="border:0;overflow:hidden;display:none;"></iframe>
  <iframe id="{frame_ids['3']}" srcdoc="{src3}" width="600" height="450" style="border:0;overflow:hidden;display:none;"></iframe>
  <iframe id="{frame_ids['4']}" srcdoc="{src4}" width="600" height="450" style="border:0;overflow:hidden;display:none;"></iframe>
</div>
<script>
(function() {{
  var sel = document.getElementById('{sel_id}');
  var btn = document.getElementById('{reset_id}');
  var frames = {{
    '2': '{frame_ids['2']}',
    '3': '{frame_ids['3']}',
    '4': '{frame_ids['4']}'
  }};
  function showCanvas(v) {{
    Object.keys(frames).forEach(function(k) {{
      var f = document.getElementById(frames[k]);
      f.style.display = (k === v ? 'block' : 'none');
    }});
  }}
  sel.addEventListener('change', function() {{ showCanvas(sel.value); }});
  btn.addEventListener('click', function() {{
    var f = document.getElementById(frames[sel.value]);
    // reload iframe
    f.srcdoc = f.srcdoc;
  }});
  // initial display
  showCanvas(sel.value);
}})();
</script>
'''
    display(HTML(html))


def afficher_manip_distance(fixed=False):
    """
    Render the interactive JSXGraph demo via an iframe(srcdoc) so that
    inline <script> tags aren't stripped by JupyterLab's sanitizer.
    """
    box_id   = f"jxgbox_{uuid.uuid4().hex}"
    js_fixed = 'true' if fixed else 'false'

    # Build the full HTML page for the iframe
    page = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Distance par rapport à deux points moyens</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraph.css" />
    <script src="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraphcore.js"></script>
    <style>
      html, body {{ margin:0; padding:0; overflow:hidden; font-family:sans-serif; }}
      #title {{ width:600px; text-align:center; font-size:16px; font-weight:bold; }}
    </style>
  </head>
  <body>
    <div id="title">Distance par rapport à deux points moyens</div>
    <div id="{box_id}" class="jxgbox" style="width:600px; height:600px;"></div>
    <script>
      // Initialize board
      var board = JXG.JSXGraph.initBoard('{box_id}', {{
        boundingbox:[-1,9,9,-1], axis:true, grid:true,
        showNavigation:false, showCopyright:false
      }});
      // Create points
      var C1 = board.create('point',[2,2],{{name:'A', color:'blue', size:4, fixed:true, label:{{fontSize:14,autoPosition:true}}}});
      var C2 = board.create('point',[7,7],{{name:'B', color:'orange', size:4, fixed:true, label:{{fontSize:14,autoPosition:true}}}});
      var P  = board.create('point',[6,5],{{name:'P', color:'green', size:3, fixed:{js_fixed}, snapToGrid:true, snapSizeX:0.5, snapSizeY:0.5, label:{{fontSize:14,autoPosition:true}}}});
      // Hypotenuse segments
      board.create('segment',[P,C1],{{color:'blue', strokeWidth:2}});
      board.create('segment',[P,C2],{{color:'orange', strokeWidth:2}});
      // Auxiliary corner points
      var P1_C1 = board.create('point',[function(){{return 2;}}, function(){{return P.Y();}}],{{visible:false}});
      var P1_C2 = board.create('point',[function(){{return 7;}}, function(){{return P.Y();}}],{{visible:false}});
      // Leg segments for C1 (blue)
      board.create('segment',[P,P1_C1],{{dash:2, color:'blue'}});
      board.create('segment',[P1_C1,C1],{{dash:2, color:'blue'}});
      // Leg segments for C2 (orange)
      board.create('segment',[P,P1_C2],{{dash:2, color:'orange'}});
      board.create('segment',[P1_C2,C2],{{dash:2, color:'orange'}});
      // Length labels
      board.create('text',[
        function(){{return (P.X()+C1.X())/2;}}, function(){{return (P.Y()+C1.Y())/2-0.3;}},
        function(){{return Math.hypot(P.X()-C1.X(), P.Y()-C1.Y()).toFixed(2);}}
      ],{{visible:false, fontSize:14, color:'blue', fontWeight:'bold'}});
      board.create('text',[
        function(){{return (P.X()+2)/2;}}, function(){{return P.Y()+0.2;}},
        function(){{return Math.abs(P.X()-2).toFixed(2);}}
      ],{{fontSize:12, color:'blue'}});
      board.create('text',[
        function(){{return 2-0.4;}}, function(){{return (P.Y()+2)/2;}}, 
        function(){{return Math.abs(P.Y()-2).toFixed(2);}}
      ],{{fontSize:12, color:'blue'}});
      board.create('text',[
        function(){{return (P.X()+C2.X())/2;}}, function(){{return (P.Y()+C2.Y())/2+0.3;}}, 
        function(){{return Math.hypot(P.X()-C2.X(), P.Y()-C2.Y()).toFixed(2);}}
      ],{{visible:false, fontSize:14, color:'orange', fontWeight:'bold'}});
      board.create('text',[
        function(){{return (P.X()+7)/2;}}, function(){{return P.Y()-0.2;}}, 
        function(){{return Math.abs(P.X()-7).toFixed(2);}}
      ],{{fontSize:12, color:'orange'}});
      board.create('text',[
        function(){{return 7+0.2;}}, function(){{return (P.Y()+7)/2;}}, 
        function(){{return Math.abs(P.Y()-7).toFixed(2);}}
      ],{{fontSize:12, color:'orange'}});
      // Legend
      board.create('text',[0.5, 8.85,
        'A : point moyen des 2<br/>B : point moyen des 7<br/>P : caractéristiques d\\'une image'
      ],{{
        anchorX:'left', anchorY:'top', fontSize:14,
        useMathJax:false, cssStyle:'text-align:left;'
      }});
    </script>
  </body>
</html>"""

    # Escape quotes for embedding in the srcdoc attribute
    srcdoc = page.replace('"', '&quot;')

    # Display the iframe
    iframe = (
        f'<iframe srcdoc="{srcdoc}" '
        'style="width:620px; height:620px; border:none;"></iframe>'
    )
    display(HTML(iframe))


### Validation

def check_coordinates(coords, errors):
    if not (isinstance(coords, tuple)):
        errors.append("Les coordonnées doivent être écrites entre parenthèses séparés par une virgule. Exemple : (3, 5)")
        return False
    if len(coords) != 2:
        errors.append("Les coordonnées doivent être composées de deux valeurs séparés par une virgule. Pour les nombres à virgule, utilisez un point '.' et non une virgule")
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
    
    # deux_caracteristiques = common.challenge.deux_caracteristiques
    
    # Proposition Akim : on hardcode les coordonnées pour que ça tombe juste
    # On récupère les coordonnées des points A et B, globale (def dans tracer_2_points)
    global A_true, B_true
    
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

def function_validation_classes(errors, answers):
    classe_point_A = answers.get('classe_point_A')
    classe_point_B = answers.get('classe_point_B')
    classe_point_C = answers.get('classe_point_C')

    # Vérifier que les valeurs sont des entiers
    if not isinstance(classe_point_A, int) or not isinstance(classe_point_B, int) or not isinstance(classe_point_C, int):
        errors.append("Les classes des points doivent être des nombres entiers.")
        return False

    # Vérifier les valeurs attendues
    if classe_point_A != 2:
        errors.append("La classe du point A n'est pas correcte. Réessayez.")
    if classe_point_B != 7:
        errors.append("La classe du point B n'est pas correcte. Réessayez.")
    if classe_point_C != 2:
        errors.append("La classe du point C n'est pas correcte. Ce point n'est pas facile à classer. Il aurait bien pu s'agir d'un 2. Réessayez.")

    return len(errors) == 0

def function_validation_valeurs_deplacement_horizontal(errors, answers):
    x2 = answers.get('deplacement_horizontal_2')
    x3 = answers.get('deplacement_horizontal_3')
    x4 = answers.get('deplacement_horizontal_4')

    # Pour vérifier que les valeurs sont bien des floats ou int
    for var_name, value in [('deplacement_horizontal_2', x2), ('deplacement_horizontal_3', x3), ('deplacement_horizontal_4', x4)]:
        if not isinstance(value, (int, float)):
            errors.append(f"La valeur de {var_name} doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
            return False

    # Comparaison avec tolérance
    if abs(x2 - 1.5) > 1e-2:
        errors.append("deplacement_horizontal_2 n'est pas correct. Bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(x3 - 1) > 1e-2:
        errors.append("deplacement_horizontal_3 n'est pas correct. Pensez à cliquer sur 3 Points en haut à gauche. Puis bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(x4 - 0.75) > 1e-2:
        errors.append("deplacement_horizontal_4 n'est pas correct. Pensez à cliquer sur 4 Points en haut à gauche. Bougez le point B sur la figure et regardez de combien bouge M.")

    return len(errors) == 0

def function_validation_valeurs_deplacement_vertical(errors, answers):
    y2 = answers.get('deplacement_vertical_2')
    y3 = answers.get('deplacement_vertical_3')
    y4 = answers.get('deplacement_vertical_4')

    # Pour vérifier que les valeurs sont bien des floats ou int
    for var_name, value in [('deplacement_vertical_2', y2), ('deplacement_vertical_3', y3), ('deplacement_vertical_4', y4)]:
        if not isinstance(value, (int, float)):
            errors.append(f"La valeur de {var_name} doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
            return False

    # Comparaison avec tolérance
    if abs(y2 - 3) > 1e-2:
        errors.append("deplacement_vertical_2 n'est pas correct. Bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(y3 - 2) > 1e-2:
        errors.append("deplacement_vertical_3 n'est pas correct. Pensez à cliquer sur 3 Points en haut à gauche. Puis bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(y4 - 1.5) > 1e-2:
        errors.append("deplacement_vertical_4 n'est pas correct. Pensez à cliquer sur 4 Points en haut à gauche. Puis bougez le point B sur la figure et regardez de combien bouge M.")

    return len(errors) == 0

validation_execution_2_points = MathadataValidate(success="")
validation_question_2_points = MathadataValidateVariables({
    'A': None,
    'B': None,
}, function_validation=function_validation_2_points)
validation_execution_200_points = MathadataValidate(success="")

# ATTENTION CHALLENGE DÉPENDANT - À FACTORISER
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
    success="C'est la bonne réponse. Un point bleu est plus proche du point moyen des 7 que de celui des 2 : il va donc être classé \"7\". De même pour un point orange. On a donc deux erreurs, ce qui fait 20%.",
    on_success=lambda answers: set_step(2)
)
validation_execution_tracer_points_droite = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_execution_point_droite = MathadataValidate(success="")
validation_question_equation = MathadataValidateVariables({
    'abscisse_M': 20,
    'ordonnee_M': None
}, function_validation=function_validation_equation, on_success=lambda answers: set_step(3))
validation_execution_caracteristiques_custom = MathadataValidate(success="")
validation_score_droite_custom = MathadataValidate(success="Bien joué, vous pouvez continuer à améliorer votre score. Il est possible de descendre à 3% d'erreur.")
validation_execution_scatter_caracteristiques_ripou = MathadataValidate(success="")
# Définir les variables de validation
validation_question_classes = MathadataValidateVariables({
    'classe_point_A': 2,
    'classe_point_B': 7,
    'classe_point_C': 2
}, 
    function_validation=function_validation_classes,
    success="Bien joué ! Ce n'était pas facile de voir que le point C est un 2. Il aurait très bien pu être un 7.",
    on_success=lambda answers: set_step(2))
# Exécuter la validation
validation_execution_classes = MathadataValidate(success="Les classes des points sont correctes; vous pouvez passer à la suite.")
validation_deplacement_horizontal = MathadataValidateVariables(
    {
        'deplacement_horizontal_2': 1.5,
        'deplacement_horizontal_3': 1,
        'deplacement_horizontal_4': 0.75,
    },
    tips=[
        {
            'seconds':20,
            'tip': 'Bougez le point B sur la figure ci-dessus et regardez de combien bouge M.'
        },
        {
            'seconds':40,
            'tip': 'Pensez à changer le nombre de points en haut à gauche.'
        }
    ],
    function_validation=function_validation_valeurs_deplacement_horizontal
)

validation_deplacement_vertical = MathadataValidateVariables(
    {
        'deplacement_vertical_2': 3,
        'deplacement_vertical_3': 2,
        'deplacement_vertical_4': 1
    },
    tips=[
        {
            'seconds': 20,
            'tip': 'Bougez le point B sur la figure ci-dessus et regardez de combien bouge M.'
        },
        {
            'seconds':40,
            'tip': 'Pensez à changer le nombre de points en haut à gauche.'
        }
    ],
    function_validation=function_validation_valeurs_deplacement_vertical
)

validation_execution_afficher_manip_distance = MathadataValidate(success="")

### ATTENTION CHALLENGE DÉPENDANT - À FACTORISER 
validation_distance_point_moyen = MathadataValidateVariables({
    'AP': 5,
    'BP': {
        'value': {
            'min': 2.22,
            'max': 2.25,
            }
        },
    'classe_P': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_P n'a pas la bonne valeur. Vous devez répondre par 2 ou 7."
            }
        ]
    }
},
    success="Bien joué, vous pouvez passer à la suite")

def function_validation_recherche_mediatrice(errors, answers):
    point_1 = answers.get('point_1')
    point_2 = answers.get('point_2')
    point_3 = answers.get('point_3')

    for var_name, value in [('point_1', point_1), ('point_2', point_2), ('point_3', point_3)]:
        if not isinstance(value, tuple):
            errors.append(f"La valeur de {var_name} doit être un couple de deux coordonnées. Ex (2, 3).")
            return False
        for coord in value:
            if not isinstance(coord, (int, float)):
                errors.append(f"La valeur de {var_name} doit être un couple de deux coordonnées. Ex (2, 3).")
                return False

    points = [point_1, point_2, point_3]
    points_name = ['point_1', 'point_2', 'point_3']
    for i in range(3):
        point = points[i]
        point_name = points_name[i]
        x, y = point[0], point[1]

        # on vérifie que les trois points sont différents
        if not (point_1 != point_2 and point_1 != point_3 and point_2 != point_3):
            errors.append('Les trois points doivent être différents')
            return False

        # On vérifie si le point est sur la droite médiatrice
        if x+y-9!=0:
            errors.append(f'le point {point_name} n\'est pas à la même distance de A et B')

    return len(errors) == 0

validation_recherche_mediatrice = MathadataValidateVariables({
    'point_1': None,
    'point_2': None,
    'point_3': None
}, function_validation=function_validation_recherche_mediatrice)

