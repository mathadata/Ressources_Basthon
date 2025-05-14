from IPython.display import display # Pour afficher des DataFrames avec display(df)
import pandas as pd
import os
import sys
# import mplcursors

# Pour accepter réponse élèves QCM
A = 'A'
B = 'B'
C = 'C'

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
            'a': 0.5,
            'b': -1,
            'c': 20
        },  
        'hover': True
    }
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def tracer_points_droite_vecteur(id=None, carac=None, initial_hidden=False, save=True, normal= None, directeur=None, reglage_normal=None):
    if id is None:
        id = uuid.uuid4().hex
    if directeur is None:
        directeur = False
    if reglage_normal is None:
        reglage_normal = False
    
    # Mise en place du conteneur pour le graphique
    display(HTML(f'''
        <!-- Conteneur pour afficher le taux d'erreur -->
        <div id="{id}-score-container"
            style="
            text-align: center;
            font-weight: bold;
            font-size: 2rem;
            {'display:none;' if initial_hidden else ''}
            ">
            Erreur : <span id="{id}-score">...</span>
        </div>

        <!-- Zone canvas pour tracer le graphique -->
        <canvas id="{id}-chart"></canvas>

        <!-- Conteneur pour les champs d'entrée -->
        <div id="{id}-inputs"
            style="
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-direction: row;
            {'display:none;' if initial_hidden else ''}
            ">
            <!-- Cas « directeur » et pas en mode « reglage_normal » -->
            <div style="
                display: {'flex' if (directeur and not reglage_normal) else 'none'};
                flex-direction: row;
                gap: 1rem;
                ">
                <!-- Paramètre ux -->
                <div>
                    <label for="{id}-input-ux" id="{id}-label-ux"></label>
                    <input type="number"
                        id="{id}-input-ux"
                        value="5"
                        step="1"
                        style="width: 50px; height: 25px; font-size: 12px;">
                </div>
                <!-- Paramètre uy -->
                <div>
                    <label for="{id}-input-uy" id="{id}-label-uy"></label>
                    <input type="number"
                        id="{id}-input-uy"
                        value="10"
                        step="1"
                        style="width: 50px; height: 25px; font-size: 12px;">
                </div>
            </div>

            <!-- Cas du mode « reglage_normal » -->
            <div style="
                display: {'flex' if reglage_normal else 'none'};
                flex-direction: row;
                gap: 1rem;
                ">
                <!-- Paramètre a -->
                <div>
                    <label for="{id}-input-a" id="{id}-label-a"></label>
                    <input type="number"
                        id="{id}-input-a"
                        value="10"
                        step="1"
                        style="width: 50px; height: 25px; font-size: 12px;">
                </div>
                <!-- Paramètre b -->
                <div>
                    <label for="{id}-input-b" id="{id}-label-b"></label>
                    <input type="number"
                        id="{id}-input-b"
                        value="-5"
                        step="1"
                        style="width: 50px; height: 25px; font-size: 12px;">
                </div>
            </div>

            <!-- Paramètre x_A -->
            <div>
                <label for="{id}-input-xA" id="{id}-label-xA"></label>
                <input type="number"
                    id="{id}-input-xA"
                    value="50"
                    step="1"
                    style="width: 50px; height: 25px; font-size: 12px;">
            </div>
            <!-- Paramètre y_A -->
            <div>
                <label for="{id}-input-yA" id="{id}-label-yA"></label>
                <input type="number"
                    id="{id}-input-yA"
                    value="50"
                    step="1"
                    style="width: 50px; height: 25px; font-size: 12px;">
            </div>
        </div>
    '''))


    if normal is None:
        normal = False

    if not initial_hidden:
        if carac is None:
            carac = common.challenge.deux_caracteristiques
        
        c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

        params = {
            'points': c_train_par_population,
            'custom': carac == common.challenge.deux_caracteristiques_custom,
            'hover': True,
            'displayValue': False,
            'save': save,
            'directeur': directeur,
            'normal': normal,
            'reglageNormal': reglage_normal,
        }
    
        run_js(f"setTimeout(() => window.mathadata.tracer_points_droite_vecteur('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

def produit_scalaire_exercice():
    display(HTML("""<iframe scrolling="no" title="Produit scalaire et Classification" src="https://www.geogebra.org/material/iframe/id/gmtxwxat/width/3000/height/800/border/888888/sfsb/true/smb/false/stb/false/stbh/false/ai/false/asb/false/sri/false/rc/false/ld/true/sdz/true/ctl/false" width="3000px" height="800px" style="border:0px;"> </iframe>"""))


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

   


def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    tracer_points_droite_vecteur(id=id, carac=common.challenge.deux_caracteristiques_custom, initial_hidden=True, save=False, normal=True, reglage_normal=True)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                const params = {{
                    points,
                    custom: true,
                    hover: true,
                    save: false,
                    'normal': true,
                    'reglageNormal':true,
                }};
                mathadata.tracer_points_droite_vecteur('{id}', params);

                // AFFICHER LE GRAPH APRÈS LA SÉLECTION
                document.getElementById('{id}-score-container').style.display = 'block';
                document.getElementById('{id}-inputs').style.display = 'flex';
            }})
        }}
    ''')

def update_custom():
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques_custom)
    return json.dumps(c_train_par_population, cls=NpEncoder)

def estim_2d(a, b, c, k):
    return common.challenge.classes[0] if (b == 0 and k[0] > -c/a) or (b != 0 and a*k[0] + b*k[1] + c > 0) else common.challenge.classes[1]

def _classification(a, b, c, c_train):
    r_est_train = np.array([
        estim_2d(a, b, c, k)
        for k in c_train
    ])
    return r_est_train
    
def erreur_lineaire(a, b, c, c_train):
    r_est_train = _classification(a, b, c, c_train)
    erreurs = (r_est_train != common.challenge.r_train).astype(int)
    return 100*np.mean(erreurs)
 
# JS

run_js('''
    if(localStorage.getItem('ux') !== null && localStorage.getItem('uy') !== null && localStorage.getItem('a') !== null && localStorage.getItem('b') !== null && localStorage.getItem('xA') !== null && localStorage.getItem('yA') !== null) {
        const ux = parseFloat(localStorage.getItem('ux'));
        const uy = parseFloat(localStorage.getItem('uy'));
        const a = parseFloat(localStorage.getItem('a'));
        const b = parseFloat(localStorage.getItem('b'));
        const y = parseFloat(localStorage.getItem('yA'));
        const x = parseFloat(localStorage.getItem('xA'));
        mathadata.run_python (`set_vector_parameters(${ux}, ${uy}, ${a}, ${b}, ${x}, ${y})`);
    }

    function findIntersectionPoints(a, b, c, x_min, x_max) {
        const y_min = x_min;
        const y_max = x_max;
        const points = [];

        // Handle vertical line (b === 0): x = -c/a
        if (b === 0) {
            const x = -c / a;
            if (x >= x_min && x <= x_max) {
                points.push({ x, y: y_min }, { x, y: y_max });
            }
        }
        // Handle horizontal line (a === 0): y = -c/b
        else if (a === 0) {
            const y = -c / b;
            if (y >= y_min && y <= y_max) {
                points.push({ x: x_min, y }, { x: x_max, y });
            }
        }
        // General case
        else {
            // Intersect with vertical boundaries: x = x_min and x = x_max
            let y = (-a * x_min - c) / b;
            if (y >= y_min && y <= y_max) points.push({ x: x_min, y });
            y = (-a * x_max - c) / b;
            if (y >= y_min && y <= y_max) points.push({ x: x_max, y });

            // Intersect with horizontal boundaries: y = y_min and y = y_max
            let x = (-b * y_min - c) / a;
            if (x >= x_min && x <= x_max) points.push({ x, y: y_min });
            x = (-b * y_max - c) / a;
            if (x >= x_min && x <= x_max) points.push({ x, y: y_max });
        }

        // Remove duplicates
        const uniquePoints = points.filter((pt, i, arr) =>
            arr.findIndex(p => p.x === pt.x && p.y === pt.y) === i
        );

        return uniquePoints.slice(0, 2).sort((p1, p2) => p1.x - p2.x); // Return at most two intersection points
    }

    function getLineEquationStr(a, b, c) {
        return `${a}x ${b < 0 ? '-' : '+'} ${Math.abs(b)}y ${c < 0 ? '-' : '+'} ${Math.abs(c)} = 0`
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
            let {m, p, a, b, c} = line;
            let label;
            
            if (m !== undefined && p !== undefined) {
                label = `y = ${m}x ${p < 0 ? '-' : '+'} ${Math.abs(p)}`;
                a = m;
                b = -1;
                c = p;
            } else if (a !== undefined && b !== undefined && c !== undefined) {
                label = getLineEquationStr(a, b, c);
            }

            const lineData = findIntersectionPoints(a, b, c, min, max); 
            
            datasets.push({
                label,
                type: 'line',
                data: lineData,
                pointRadius: 0,
                pointHitRadius: 0,
                borderColor: 'black',
                borderWidth: 1,
            });

            datasets.push({
                label: 'Zone de la classe  r^ = 2',
                type: 'line',
                data: lineData,
                pointsRadius: 0,
                pointHitRadius: 0,
                borderColor: 'transparent',
                fill: 'end',
                backgroundColor: `rgba(${mathadata.classColorCodes[0]}, 0.1)`,
            });

            datasets.push({
                label: 'Zone de la classe  r^ = 7',
                type: 'line',
                data: lineData,
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
            plugins: [],
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
    }

window.mathadata.tracer_points_droite_vecteur = function(id, params) {
    mathadata.tracer_points(`${id}-chart`, params);
    if (typeof params === 'string') {
        params = JSON.parse(params);
    }
    const {custom, displayValue, save, normal, directeur, reglageNormal} = params;
    const chart = window.mathadata.charts[`${id}-chart`]

    const start_ux = 50
    const start_uy = 10

    // add vector u
    chart.data.datasets.push({
        type: 'line',
        data: [],
        borderColor: 'red',
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 0,
        label: '\u20D7u',
        hidden: !params.directeur,
    }); 

    // add line
    chart.data.datasets.push({
        type: 'line',
        data: [],
        borderColor: 'black',
        borderWidth: 1,
        pointRadius: 0,
        pointHitRadius: 0,
        label: 'ax + by + c = 0',
    });
   
    // add point A
    chart.data.datasets.push({
        data: [{ x: 50, y: 50 }],
        backgroundColor: 'black',
        borderColor: 'black',
        pointStyle: 'cross',
        pointRadius: 10,
        pointHoverRadius: 10,
        borderWidth: 3,
        hoverBorderWidth: 3,
        label: 'A',
        z: 10,
    });

     //Ajout du vecteur normal
    chart.data.datasets.push({
        type: 'line',
        data: [],
        borderColor: 'blue',
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 0,
        label: '\u20D7n',
        hidden: !params.normal,
    });

    let input_ux = document.getElementById(`${id}-input-ux`)
    let input_uy = document.getElementById(`${id}-input-uy`)
    const label_ux = document.getElementById(`${id}-label-ux`)
    const label_uy = document.getElementById(`${id}-label-uy`)
    let input_a = document.getElementById(`${id}-input-a`)
    let input_b = document.getElementById(`${id}-input-b`)
    const label_a = document.getElementById(`${id}-label-a`)
    const label_b = document.getElementById(`${id}-label-b`)
    let input_xA = document.getElementById(`${id}-input-xA`);
    let input_yA = document.getElementById(`${id}-input-yA`);
    const label_xA = document.getElementById(`${id}-label-xA`);
    const label_yA = document.getElementById(`${id}-label-yA`);
    const score = document.getElementById(`${id}-score`)

    if (chart.config.plugins[chart.config.plugins.length - 1]?.afterDraw === undefined) {
        chart.config.plugins.push({
            afterDraw: function(chart) {
                const ctx = chart.ctx;
                let a,b,ux,uy;
                if (reglageNormal) {
                    a = parseFloat(input_a.value);
                    b = parseFloat(input_b.value);
                    ux = b;
                    uy = -a;
                } else {
                    ux = parseFloat(input_ux.value);
                    uy = parseFloat(input_uy.value);
                    a = uy;
                    b = -ux;
                }
                const xA = parseFloat(input_xA.value);
                const yA = parseFloat(input_yA.value);

                if (chart.isDatasetVisible(2)) {
                    const meta = chart.getDatasetMeta(2);
                    const data = meta.data;
                    
                    const x1 = data[0].x;
                    const y1 = data[0].y;
                    const x2 = data[1].x;
                    const y2 = data[1].y;
                    const dx = x2 - x1;
                    const dy = y2 - y1;

                    // Draw vector u arrow
                    const headlen = 10;
                    const angle = Math.atan2(dy, dx);

                    ctx.beginPath();
                    ctx.moveTo(x2, y2);
                    ctx.lineTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
                    ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
                    ctx.lineTo(x2, y2);
                    ctx.fillStyle = 'red';
                    ctx.fill();

                    if (meta._parsed) {
                        ctx.save();
                        ctx.font = '18px Arial';
                        ctx.fillStyle = 'red';
                        ctx.textAlign = 'left';
                        ctx.fillText(`\u20D7u(${ux}, ${uy})`, x2 + 10, y2);
                        ctx.restore(); 
                    }
                }

                if (chart.isDatasetVisible(3)) {
                    const meta = chart.getDatasetMeta(3);
                    const data = meta.data;
                    let lineParam_a, lineParam_b, lineParam_c;
                    lineParam_a = a;
                    lineParam_b = b;
                    lineParam_c = -a * xA - b * yA;  // La droite passe par A
                    const x1 = data[0].x;
                    const y1 = data[0].y;
                    const x2 = data[1].x;
                    const y2 = data[1].y;
                    const dx = x2 - x1;
                    const dy = y2 - y1;

                    const angle = Math.atan2(dy, dx);
                    
                    const textX = Math.min(x2 + 20, chart.chartArea.right - 5);
                    const textY = Math.min(y2 + 20, chart.chartArea.bottom - 5);
                    
                    ctx.save();
                    ctx.translate(textX, textY);
                    ctx.rotate(angle);
                    ctx.font = '16px Arial';
                    ctx.fillStyle = 'black';
                    ctx.textAlign = 'right';
                    ctx.fillText(getLineEquationStr(lineParam_a, lineParam_b, lineParam_c), 0, 0);
                    ctx.restore();
                }

                if (chart.isDatasetVisible(4)) {
                    const meta = chart.getDatasetMeta(4);
                    const data = meta.data;
                    const x = data[0].x;
                    const y = data[0].y;
                    ctx.save();
                    ctx.font = '18px Arial';
                    ctx.fillStyle = 'black';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(`A(${xA}, ${yA})`, x + 10, y);
                    ctx.restore();
                }

                // AJOUT : dessiner vecteur normal n
                if (chart.isDatasetVisible(5)) {
                    const meta = chart.getDatasetMeta(5);
                    const data = meta.data;
                    if (data.length >= 2) {
                        const x1 = data[0].x;
                        const y1 = data[0].y;
                        const x2 = data[1].x;
                        const y2 = data[1].y;
                        const dx = x2 - x1;
                        const dy = y2 - y1;
                        const headlen = 10;
                        const angle = Math.atan2(dy, dx);

                        ctx.beginPath();
                        ctx.moveTo(x2, y2);
                        ctx.lineTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
                        ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
                        ctx.lineTo(x2, y2);
                        ctx.fillStyle = 'blue';
                        ctx.fill();
                        
                        // Étiquette n
                        ctx.save();
                        ctx.font = '18px Arial';
                        ctx.fillStyle = 'blue';
                        ctx.textAlign = 'left';
                        ctx.fillText(`\u20D7n(${a}, ${b})`, x2 + 10, y2);
                        ctx.restore();
                    }
                }
            }
        });
    }

    chart.config.options.plugins.legend = {
        labels: {
            filter(legendItem, chartData) {
                const { datasetIndex } = legendItem;
                return datasetIndex !== 2 && datasetIndex !== 3 && datasetIndex !== 4 && datasetIndex !== 5;
            }
        }
    };

    let exec = null;
    const update = () => {
       let a, b, ux, uy // a et b sont les coordonnées du vecteur normal n
       if(reglageNormal) {
            a = parseFloat(input_a.value);
            b = parseFloat(input_b.value);
            ux = -b;
            uy = a;
       }else {
            ux = parseFloat(input_ux.value);
            uy = parseFloat(input_uy.value);
            a = uy;
            b = -ux;
       }

        const xA = parseFloat(input_xA.value);
        const yA = parseFloat(input_yA.value);

        if (isNaN(a) || isNaN(b) || isNaN(xA) || isNaN(yA)) {
            return;
        }

        const min_x = chart.options.scales.x.min;
        const max_x = chart.options.scales.x.max;
        
        const vectorData = [{x: start_ux, y: start_uy}, {x: start_ux + ux, y: start_uy + uy}];

        let lineParam_a, lineParam_b, lineParam_c;
        lineParam_a = a;
        lineParam_b = b;
        lineParam_c = -a * xA - b * yA;  // La droite passe par A
        

        const lineData = findIntersectionPoints(lineParam_a, lineParam_b, lineParam_c, min_x, max_x);

        chart.data.datasets[2].data = vectorData;
        chart.data.datasets[2].label = `\u20D7u(${ux}, ${uy})`;

        chart.data.datasets[3].data = lineData;
        chart.data.datasets[3].label = getLineEquationStr(lineParam_a, lineParam_b, lineParam_c);

        chart.data.datasets[4].data = [{ x: xA, y: yA }];
       
        // *** AJOUT CONDITIONNEL : calcul des coordonnées du vecteur normal n ***
        if (normal) {
            const u_length = Math.sqrt(ux * ux + uy * uy);
            const n_length = u_length > 0 ? u_length : 30;

            chart.data.datasets[5].data = [
                { x: xA, y: yA },
                { 
                    x: xA + (u_length > 0 ? (uy / u_length) * n_length : 0), 
                    y: yA - (u_length > 0 ? (ux / u_length) * n_length : 0) 
                }
            ];
        }


        chart.update()

        if (save) {
            localStorage.setItem('a', a);
            localStorage.setItem('b', b);
            localStorage.setItem('ux', ux);
            localStorage.setItem('uy', uy);
            localStorage.setItem('xA', xA);
            localStorage.setItem('yA', yA);
        }

        if (ux=== 0 && uy=== 0) {
            return;
        }

        const python = `compute_score_json(${lineParam_a}, ${lineParam_b}, ${lineParam_c}, custom=${custom ? 'True' : 'False'})`
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
            mathadata.run_python(`set_vector_parameters(${ux}, ${uy}, ${a}, ${b}, ${xA}, ${yA})`);
        }, 200)
        
        
        label_a.innerHTML = `a = ${displayValue ? a : ''}`
        label_b.innerHTML = `b = ${displayValue ? b : ''}`    
        label_ux.innerHTML = `u<sub>x</sub> = ${displayValue ? ux : ''}`
        label_uy.innerHTML = `u<sub>y</sub> = ${displayValue ? uy : ''}`
        label_xA.innerHTML = `x<sub>A</sub> = ${displayValue ? xA : ''}`;
        label_yA.innerHTML = `y<sub>A</sub> = ${displayValue ? yA : ''}`;
        
    }

    if (save) {
        const saved_a = localStorage.getItem('a');
        const saved_b = localStorage.getItem('b');
        const saved_ux = localStorage.getItem('ux');
        const saved_uy = localStorage.getItem('uy');
        const saved_xA = localStorage.getItem('xA');
        const saved_yA = localStorage.getItem('yA');

        if (saved_a !== null) input_a.value = saved_a;
        if (saved_b !== null) input_b.value = saved_b;
        if (saved_ux !== null) input_ux.value = saved_ux;
        if (saved_uy !== null) input_uy.value = saved_uy;
        if (saved_xA !== null) input_xA.value = saved_xA;
        if (saved_yA !== null) input_yA.value = saved_yA;
    }

    function removeAllEventListeners(element) {
        const clone = element.cloneNode(true);
        element.parentNode.replaceChild(clone, element);
        return clone;
    }

    input_ux = removeAllEventListeners(input_ux);
    input_uy = removeAllEventListeners(input_uy);
    input_a = removeAllEventListeners(input_a);
    input_b = removeAllEventListeners(input_b);
    input_xA = removeAllEventListeners(input_xA);
    input_yA = removeAllEventListeners(input_yA);

    input_ux.addEventListener("input", update);
    input_uy.addEventListener("input", update);
    input_a.addEventListener("input", update);
    input_b.addEventListener("input", update);
    input_xA.addEventListener("input", update);
    input_yA.addEventListener("input", update);

    update();
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

g_a = None
g_b = None
g_c = None

def compute_score(a, b, c, custom=False):
    if custom:
        carac = common.challenge.deux_caracteristiques_custom
    else:
        carac = common.challenge.deux_caracteristiques
        
    c_train = compute_c_train(carac, common.challenge.d_train)
    error = erreur_lineaire(a, b, c, c_train)
    return error

def compute_score_json(a, b, c, custom=False):
    global g_a, g_b, g_c
    g_a = a
    g_b = b
    g_c = c
    
    error = compute_score(a, b, c, custom)
    return json.dumps({'error': error})

ux = None
uy = None
a = None
b = None

##Point A
xA = None
yA = None


def set_vector_parameters(ux_val, uy_val, a_val, b_val, xA_val, yA_val):
    global a, b, xA, yA, g_a, g_b, g_c, ux, uy
    ux = ux_val
    uy = uy_val
    a = a_val
    b = b_val
    xA = xA_val
    yA = yA_val

    if a and b:
        g_a = a      
        g_b = b
        g_c = -a * xA - b * yA



def calculer_score_droite():
    global g_a, g_b, g_c
    
    deux_caracteristiques = common.challenge.deux_caracteristiques
    base_score = compute_score(g_a, g_b, g_c)
    
    if base_score <= 50:
        above = common.challenge.classes[0]
        below = common.challenge.classes[1]
    else:
        above = common.challenge.classes[1]
        below = common.challenge.classes[0]

    if base_score >= 8 and base_score <= 92:
        print_error("Vous pourrez passer à la suite quand vous aurez un pourcentage d'erreur de moins de 8%.")

    deux_caracteristiques = common.challenge.deux_caracteristiques
    def algorithme(d):
        k = deux_caracteristiques(d)
        if (g_b == 0 and k[0] > -g_c/g_a) or (g_b != 0 and g_a*k[0] + g_b*k[1] + g_c > 0):
            return above
        else:
            return below

    def cb(score):
        if score < 0.08:
            validation_score_droite()

    calculer_score(algorithme, method="2 moyennes", parameters=f"a={g_a}, b={g_b}, c={g_c}", cb=cb) 

def calculer_score_droite_normal():
    global g_a, g_b, g_c
    
    deux_caracteristiques = common.challenge.deux_caracteristiques
    base_score = compute_score(g_a, g_b, g_c)
    
    if base_score <= 50:
        above = common.challenge.classes[0]
        below = common.challenge.classes[1]
    else:
        above = common.challenge.classes[1]
        below = common.challenge.classes[0]

    if base_score >= 8 and base_score <= 92:
        print_error("Vous pourrez passer à la suite quand vous aurez un pourcentage d'erreur de moins de 8%.")

    deux_caracteristiques = common.challenge.deux_caracteristiques
    def algorithme(d):
        k = deux_caracteristiques(d)
        if (g_b == 0 and k[0] > -g_c/g_a) or (g_b != 0 and g_a*k[0] + g_b*k[1] + g_c > 0):
            return above
        else:
            return below
    def cb(score):
        if score < 0.08:
            validation_score_droite_normal()

    calculer_score_2(algorithme, method="2 moyennes", parameters=f"a={g_a}, b={g_b}, c={g_c}", cb=cb) 


def calculer_score_droite_normal_2custom():
    global g_a, g_b, g_c
    
    deux_caracteristiques = common.challenge.deux_caracteristiques
    base_score = compute_score(g_a, g_b, g_c)
    
    if base_score <= 50:
        above = common.challenge.classes[0]
        below = common.challenge.classes[1]
    else:
        above = common.challenge.classes[1]
        below = common.challenge.classes[0]

    if base_score >= 8 and base_score <= 92:
        print_error("Vous pourrez passer à la suite quand vous aurez un pourcentage d'erreur de moins de 8%.")

    deux_caracteristiques = common.challenge.deux_caracteristiques
    def algorithme(d):
        k = deux_caracteristiques(d)
        if (g_b == 0 and k[0] > -g_c/g_a) or (g_b != 0 and g_a*k[0] + g_b*k[1] + g_c > 0):
            return above
        else:
            return below
    def cb(score):
        if score < 0.08:
            validation_score_droite_normal_2custom()

    calculer_score_2(algorithme, method="2 moyennes", parameters=f"a={g_a}, b={g_b}, c={g_c}", cb=cb) 

def calculer_score_custom_droite():
    global g_a, g_b, g_c
    if compute_score(g_a, g_b, g_c, custom=True) <= 50:
        above = common.challenge.classes[0]
        below = common.challenge.classes[1]
    else:
        above = common.challenge.classes[1]
        below = common.challenge.classes[0]
    
    deux_caracteristiques = common.challenge.deux_caracteristiques_custom
    def algorithme(d):
        k = deux_caracteristiques(d)
        if (g_b == 0 and k[0] > -g_c/g_a) or (g_b != 0 and g_a*k[0] + g_b*k[1] + g_c > 0):
            return above
        else:
            return below
        
    def cb(score):
        if score < 0.06:
            validation_score_droite_custom()
        else:
            print_error("Continuez à chercher 2 zones pour avoir moins de 6% d'erreur. N'oubliez pas de mettre à jour les valeurs de a, b et y après avoir défini votre zone.")

    calculer_score(algorithme, method="2 moyennes custom", parameters=f"a={g_a}, b={g_b}, c={g_c}", cb=cb) 
 
### Validation

def check_coordinates(coords, errors):
    if not (isinstance(coords, tuple)):
        errors.append("Les coordonnées doivent être écrites entre parenthèses séparés par une virgule. Exemple : (3, 5)")
        return False
    if len(coords) != 2:
        errors.append("Les coordonnées doivent être composées de deux valeurs séparés par une virgule. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
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

def is_n_vers_le_haut(n):
    return (n[0]<=0)



def function_validation_normal(errors, answers):
    n = answers['n']
    if not (isinstance(n, tuple) and len(n) == 2 and all(isinstance(x, (int, float)) for x in n)):
        errors.append("Écrivez les coordonnées du vecteur entre parenthèses séparées par une virgule. Pour les valeurs non entières, utilisez un point. Exemple : (3.5 , 5)")
        return False

    if n[0]*a+n[1]*b!=0:
        errors.append("Ce n'est pas une réponse correcte. Un vecteur n orthogonal à un autre vecteur u (a, b) vérifie n.u=0. Une possibilité est le vecteur (-b,a).")
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
validation_execution_tracer_points_droite_vecteur = MathadataValidate(success="")
validation_execution_tracer_points_droite_vecteur_2 = MathadataValidate(success="")
validation_execution_tracer_points_droite_vecteur_3 = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_score_droite_normal = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_score_droite_normal_2custom = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_execution_point_droite = MathadataValidate(success="")
validation_question_normal = MathadataValidateVariables({
    'n': None
}, function_validation=function_validation_normal,success="C'est une bonne réponse. Le vecteur n est orthogonal au vecteur directeur.")

validation_execution_produit_scalaire_exercice = MathadataValidate(success="")


## Attention la réponse dépend du choix fait dans geogebra pour le vecteur normal
n_geogebra = (4, -8)
A_geogebra = (20, 30)
M1_geogebra = (40, 30)
M2_geogebra = (25, 35)
AM1_geogebra = (M1_geogebra[0] - A_geogebra[0], M1_geogebra[1] - A_geogebra[1])
AM2_geogebra = (M2_geogebra[0] - A_geogebra[0], M2_geogebra[1] - A_geogebra[1])


def function_validation_question_produit_scalaire(errors, answers):
    produit_scalaire = answers['produit_scalaire']
    if not isinstance(produit_scalaire, (int, float)):
        errors.append("Le produit scalaire doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False
    if produit_scalaire==Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if produit_scalaire!=function_calcul_produit_scalaire(n_geogebra,AM1_geogebra):
        errors.append("Ce n'est pas la bonne valeur. Relis la définition du produit scalaire et vérifie tes calculs.")
        return False 
    return True

validation_question_produit_scalaire = MathadataValidateVariables({
    'produit_scalaire': None
}, function_validation=function_validation_question_produit_scalaire, tips=[
    {
      'trials': 1,
      'tip': 'Vous devez calculer le produit scalaire entre n(4, -8) et AM(?, ?) avec A(20, 30) et M(40, 30)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (40-20, 30-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(4, -8) et AM(20, 0).'
    },
    {
      'trials': 4,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par 4*20 + (-8)*0'
    },
     {
      'trials': 5,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par 4*20 + (-8)*0 = 80'
    }
])

validation_execution_caracteristiques_custom = MathadataValidate(success="")
validation_execution_scatter_caracteristiques_ripou = MathadataValidate(success="")


def function_validation_question_produit_scalaire_2(errors, answers):
    produit_scalaire = answers['produit_scalaire']
    if not isinstance(produit_scalaire, (int, float)):
        errors.append("Le produit scalaire doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False
    if produit_scalaire==Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if produit_scalaire!=function_calcul_produit_scalaire(n_geogebra,AM2_geogebra):
        errors.append("Ce n'est pas la bonne valeur. Relis la définition du produit scalaire et vérifie tes calculs.")
        return False 
    return True

validation_question_produit_scalaire_2 = MathadataValidateVariables({
    'produit_scalaire': None
}, function_validation=function_validation_question_produit_scalaire_2, tips=[
    {
      'trials': 1,
      'tip': 'Vous devez calculer le produit scalaire entre n(4, -8) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(4, -8) et AM(5, 5).'
    },
    {
      'trials': 4,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par 4*5 + (-8)*5'
    },
     {
      'trials': 5,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par  4*5 + (-8)*5 = -20'
    }
])



validation_question_produit_scalaire_2 = MathadataValidateVariables({
    'produit_scalaire': -20
}, tips=[
    {
      'trials': 1,
      'tip': 'Vous devez calculer le produit scalaire entre n(4,-8) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(4, -8) et AM(5, 5).'
    },
    {
      'trials': 4,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par 4*5 + (-8)*5 = -20'
    }
  ])


validation_score_droite_custom = MathadataValidate(success="Bien joué, vous pouvez continuer à améliorer votre score. Il est possible de descendre à 3% d'erreur.")


# ajout louis


def function_calcul_produit_scalaire(u,v):
    return u[0]*v[0] + u[1]*v[1]

def function_calcul_produit_vectoriel(u,v):
    return u[0]*v[1] - u[1]*v[0]


def function_sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    

validation_question_produit_scalaire_louis = MathadataValidateVariables({
    'produit_scalaire': 10
}, tips=[
    {
      'trials': 1,
      'tip': 'Vous devez calculer le produit scalaire entre n(-2, 4) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(-2, 4) et AM(5, 5).'
    },
    {
      'trials': 4,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par -2*5 + 4*5 = 10'
    }
  ])

xM=40
yM=20

def function_validation_question_decouverte_vecteur_normal(errors, answers):
    n = answers['n']
    if not check_coordinates(n, errors):
        return False
    if function_calcul_produit_vectoriel(n,(a,b)) != 0:
        if abs(n[0])==abs( a) and abs(n[1])==abs(b):
            errors.append("Il y a une erreur de signe dans ta réponse. Relis la propriété juste au-dessus et lis l'équation de la droite sur le graphique.")
        else:
            errors.append("Ce n'est pas une réponse correcte. Relis la propriété juste au-dessus et lis l'équation de la droite sur le graphique.")
        return False
    if n[0] != a or n[1] != b:
        errors.append("C'est bien un vecteur normal ! Mais nous cherchons le vecteur n = (a, b), avec a et b correspondant à l'équation de la droite." )
        return False
    return True
    
    

validation_question_decouverte_vecteur_normal= MathadataValidateVariables({
    'n': None,
    
}, function_validation=function_validation_question_decouverte_vecteur_normal,
tips= [{
      'trials': 1,
      'seconds': 30,
      'tip': 'Sur le graphique l\'équation de la droite est donnée par ax + by + c = 0. Lis les valeurs de a et b pour trouver un vecteur normal n.'
    },
    {
      'trials': 2,
      'seconds': 50,
      'tip': 'Sur le graphique l\'équation de la droite est donnée par ax + by + c = 0. Un vecteur normal possible est n = (a, b).'
    }
    ])

def function_validation_produit_n_u(errors, answers):
    produit_scalaire_n_u = answers['produit_scalaire_n_u']
    if produit_scalaire_n_u is Ellipsis:
         errors.append("Tu n'as pas remplacé les ...")
         return False
    if not isinstance(produit_scalaire_n_u, (int, float)):
            errors.append("Le résultat doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
            return False
    if produit_scalaire_n_u!=0:
        errors.append("La réponse est fausse. Relis la question.")
        return False
    return True

validation_question_produit_n_u = MathadataValidateVariables({
    'produit_scalaire_n_u': 0
}, function_validation=function_validation_produit_n_u, success="Bravo, c'est la bonne réponse. Le produit scalaire entre tout vecteur normal n et tout vecteur directeur u vaut bien 0 car ils sont orthogonaux ! ",
tips= [
    {
      'trials': 1,
      'seconds': 3,
      'tip': 'Tu dois calculer le produit scalaire entre le vecteur n (que tu as donné à la cellule précédente) et le vecteur u que tu as réglé dans le graphique.'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Tu dois calculer le produit scalaire entre le vecteur n (celui que tu as donné à la cellule précédente) et le vecteur u que tu as réglé dans le graphique. Le produit scalaire est donné par n_x*u_x< + n_y*u_y.'
    },
    {
      'seconds': 60,
      'trials': 5,
      'tip': 'Tu n\'as pas réussi cette question soit du fait d\'un bug de l\'activité soit du fait d\'une erreur de ta part. Mais tu peux poursuivre l\'activité',
      'validate': True # Unlock the next cells
    }
    ]
    )


# Sens du vecteur normal : 
# Attention dépend du choix de caractéristiques !! 
def angle_vecteur(vecteur):
    angle_rad = np.arctan2(vecteur[1], vecteur[0])  # y, x
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 360  # Pour que l'angle soit toujours entre 0 et 360°

## WIP 
def function_validation_question_classe_direction_n(errors, answers):
    classe_sens_vecteur_normal = answers['classe_sens_vecteur_normal']
    classe_sens_oppose_vecteur_normal = answers['classe_sens_oppose_vecteur_normal']
    n=(a,b)
    if classe_sens_vecteur_normal is Ellipsis or classe_sens_oppose_vecteur_normal is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if angle_vecteur((12,10))<=angle_vecteur(n)<angle_vecteur((12,10))+180:
        if classe_sens_vecteur_normal != 2:
            errors.append("La réponse fournie pour \'classe_sens_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
        if classe_sens_oppose_vecteur_normal != 7:
            errors.append("La réponse fournie pour \'classe_sens_oppose_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
    else:
        if classe_sens_vecteur_normal != 7:
            errors.append("La réponse fournie pour \'classe_sens_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
        if classe_sens_oppose_vecteur_normal != 2:
            errors.append("La réponse fournie pour \'classe_sens_oppose_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False    
    return True

validation_question_classe_direction_n = MathadataValidateVariables({
'classe_sens_vecteur_normal' : None,
'classe_sens_oppose_vecteur_normal' : None
}, function_validation=function_validation_question_classe_direction_n,
tips= [{
      'trials': 1,
      'seconds': 30,
      'tip': 'Ta réponse n\'est pas cohérente avec le graphique et le sens du vecteur normal n. Relis la question et regarde le graphique.'
    },
    ])



M_retourprobleme=(40,20)

def function_validation_normal_2a(errors, answers):
    vec = answers['vecteur_AM']
    x_A = answers['x_A']
    y_A = answers['y_A']
    x_M = answers['x_M']
    y_M = answers['y_M']
    
    if check_coordinates(vec, errors) and (Ellipsis,Ellipsis,Ellipsis,Ellipsis)==(x_A, y_A, x_M, y_M):
        if vec !=(M_retourprobleme[0]-xA,M_retourprobleme[1]-yA):
            errors.append("Les coordonnées du vecteur AM ne sont pas correctes. Reprends les calculs.")
            return False
        return True
      
    if not all(isinstance(coord, (int, float)) for coord in (x_A, y_A, x_M, y_M)):
        errors.append("Les coordonnées des points doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False
    if not check_coordinates(vec, errors):
        return False
    if (xM, yM) != (x_M, y_M):
        errors.append("Les coordonnées du point M ne sont pas correctes. Vérifie la position de M dans l'énoncé.")
        return False
    if (xA, yA) != (x_A, y_A):
        errors.append("Les coordonnées du point A ne sont pas correctes. Vérifie la position de A sur le graphique")
        return False
    if vec != (xM - xA, yM - yA):
        errors.append("Ce n'est pas une réponse correcte. Retrouve la formule pour obtenir les coordonnées d'un vecteur à partir des coordonnées de deux points.")
        return False

    return len(errors)==0     

validation_question_normal_2a = MathadataValidateVariables({
    'vecteur_AM': None,
    'x_A' : None,
    'y_A' : None,
    'x_M' : None,
    'y_M' : None
}, function_validation=function_validation_normal_2a,
tips= [{
      'trials': 2,
      'seconds': 60,
      'tip': 'Pour calculer un vecteur à partir des coordonnées de deux points, il faut soustraire les coordonnées du point de départ (A) de celles du point d\'arrivée (M). Par exemple, pour le vecteur AM, on fait (xM - xA, yM - yA).'
    }
    ])




def function_validation_normal_2b(errors, answers):
    valeur = answers['produit_scalaire_n_AM']
    vec = get_variable('vecteur_AM')
    
    if not isinstance(valeur, (int, float)):
        errors.append("Le résultat doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False

    if valeur != function_calcul_produit_scalaire((a,b), vec):
        errors.append("La réponse n'est pas correcte. ")
        return False

    return True     

validation_question_normal_2b = MathadataValidateVariables({
    'produit_scalaire_n_AM': None
}, function_validation=function_validation_normal_2b,tips= [{
      'trials': 1,
      'seconds': 60,
      'tip': 'Note sur un papier les coordonnées du vecteur n, les coordonnées du vecteur AM puis effectue le produit scalaire.'
    },{
      'trials': 2,
      'seconds': 60,
      'tip': 'Note sur un papier les coordonnées du vecteur n(a,b), les coordonnées du vecteur AM(e,f) puis effectue le produit scalaire ae+bf.'
    }
    ])


def function_validation_normal_2c(errors, answers):
    produit_scalaire= get_variable('produit_scalaire_n_AM')
    reponse= answers['reponse']
    if reponse is Ellipsis :
        errors.append("Tu n'as pas remplacé les ...")
        return False
    
    if reponse not in (A, B, C):  
        errors.append(" La réponse ne peut être que A, B ou C.")
        return False  
    conditions = {
    A: produit_scalaire > 0,
    B: produit_scalaire < 0,
    C: produit_scalaire == 0
    }
    if conditions.get(reponse, False):    
        return True
    errors.append("Ta réponse n'est pas correcte. Reprends l'énoncé")
    return False


validation_question_normal_2c = MathadataValidateVariables({
    'reponse': None,
  
}, function_validation=function_validation_normal_2c)

def function_validation_normal_2d(errors, answers):
    classe_de_M =answers['classe_de_M'] 
    if classe_de_M is Ellipsis :
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if classe_de_M!=7:
        errors.append("La réponse est incorecte. Relis au dessus comment déterminer la classe d'un point à l'aide d'un produit scalaire.")
        return False
    return True 

validation_question_normal_2d = MathadataValidateVariables({
  'classe_de_M': None
}, function_validation=function_validation_normal_2d,success="Bravo ! Tu es arrivé·e au bout du processus de Classification d'une image inconnue.")




#Zone custom 
A_2 = (7, 2)       # <- coordonnées du point A1
B_2 = (9, 25)     # <- coordonnées du point B1


A_1 = (14, 2)     # <- coordonnées du point A2
B_1 = (23, 10)     # <- coordonnées du point B2

 
def affichage_zones_custom_2(A1, B1, A2, B2,normal=False,trace=False):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)
    if trace:
        tracer_points_droite_vecteur(carac=common.challenge.deux_caracteristiques_custom, save=False, normal=normal, reglage_normal=normal,directeur= not normal)


validation_question_2cara_comprehension = MathadataValidateVariables({
    'reponse': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_points_bleus n'a pas la bonne valeur. Vous devez répondre par 2 ou 7."
            },
            {
                'value': 7,
                'if': "Il y a plus de pixels blanc (valeur élevée) dans la zone rouge pour le 2. La moyenne du niveau de gris sera donc plus élevée pour le 2"
                },
        ]
    }
})

def calculer_score_custom_droite_2cara():
    global g_a, g_b, g_c
    if compute_score(g_a, g_b, g_c, custom=True) <= 50:
        above = common.challenge.classes[0]
        below = common.challenge.classes[1]
    else:
        above = common.challenge.classes[1]
        below = common.challenge.classes[0]
    
    deux_caracteristiques = common.challenge.deux_caracteristiques_custom
    def algorithme(d):
        k = deux_caracteristiques(d)
        if (g_b == 0 and k[0] > -g_c/g_a) or (g_b != 0 and g_a*k[0] + g_b*k[1] + g_c > 0):
            return above
        else:
            return below
        
    def cb(score):
        if score < 0.11:
            validation_score_droite_custom_2cara()
        else:
            print_error("Continuez à chercher 2 zones pour avoir moins de 11% d'erreur. N'oubliez pas de mettre à jour les valeurs de a, b et y après avoir défini votre zone.")

    calculer_score(algorithme, method="2 moyennes custom", parameters=f"a={g_a}, b={g_b}, c={g_c}", cb=cb) 
validation_score_droite_custom = MathadataValidate(success="Bien joué, vous pouvez continuer à améliorer votre score.")