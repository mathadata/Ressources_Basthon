import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common

notebook_id = 5

common.start_analytics_session(notebook_id)

if not sequence:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))

def compute_histogramme(caracteristique):
    c_train = compute_c_train(caracteristique, common.challenge.d_train)
    data = {}
    for i in range(len(c_train)):
        c = c_train[i]
        k = int(c / 2) * 2
        if k not in data:
            data[k] = [0,0]
        
        if common.challenge.r_train[i] == common.challenge.classes[0]:
            data[k][0] += 1
        else:
            data[k][1] += 1

    return data

def afficher_histogramme(div_id=None, seuil=None, caracteristique=None, legend=False):
    if div_id is None:
        div_id = uuid.uuid4().hex
        display(HTML(f'<canvas id="{div_id}"></canvas>'))
        
    if not caracteristique:
        caracteristique = common.challenge.caracteristique
        
    data = compute_histogramme(caracteristique)

    run_js(f"setTimeout(() => window.mathadata.displayHisto('{div_id}', '{json.dumps(data)}', {'true' if legend else 'false'}, {seuil if seuil is not None else 'undefined'}, true), 100)")

def animation_histogramme(id=None, carac=None):
    if id is None:
        id = uuid.uuid4().hex
        display(HTML(f'''
            <div id="{id}" style="display: flex; gap: 2rem; height: 300px; width: 100%;">
                <div id="{id}-data" style="width: 300px; height: 300px;"></div>
                <div style="height: 300px; flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: space-between">
                    <div style="flex: 1; display: flex; justify-content: center; align-items: center;">Carcatéristique x&nbsp;=&nbsp;<span id="{id}-x">Calcul...</span></div>
                    <canvas id="{id}-histo"></canvas>
                    <canvas id="{id}-chart"></canvas>
                </div>
            </div>
        '''))

    if carac is None:
        carac = common.challenge.caracteristique

    set = common.challenge.d_train
    c_train = compute_c_train(carac, set)
    params = {
        'data': set,
        'c_train': c_train,
        'labels': [0 if r == common.challenge.classes[0] else 1 for r in common.challenge.r_train],
    }
    run_js(f"setTimeout(() => window.mathadata.animation_histogramme('{id}', '{json.dumps(params, cls=NpEncoder)}'), 100)")

def calculer_score_hist_seuil():
    if not has_variable('t'):
        return
    
    t = get_variable('t')
    if t is Ellipsis:
        return
    
    r_petite_caracteristique = common.challenge.r_petite_caracteristique
    r_grande_caracteristique = common.challenge.r_grande_caracteristique
    caracteristique = common.challenge.caracteristique

    def algorithme(d):
        k = caracteristique(d)
        if k <= t:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique

    def cb(score):
        validation_score_seuil_optim()
        set_step(3)
    
    calculer_score(algorithme, method="moyenne ref hist", parameters=f"t={t}", cb=cb) 


def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
        <canvas id="{id}-histo"></canvas>
    '''))
    common.challenge.display_custom_selection(id)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (res) => {{
                window.mathadata.displayHisto('{id}-histo', res, true, undefined, true)
            }})
        }}
    ''')

def update_custom():
    return json.dumps(compute_histogramme(common.challenge.caracteristique_custom), cls=NpEncoder)


run_js('''
window.mathadata.displayHisto = (div_id, data, with_legend, seuil, with_axes_legend) => {
    if (typeof data === 'string')
        data = JSON.parse(data)

    const data_1 = Object.entries(data).map(([key, v]) => ({x: parseInt(key) + 1, y: v[0]}))
    const data_2 = Object.entries(data).map(([key, v]) => ({x: parseInt(key) + 1, y: v[1]}))
    const config = {
        type: 'bar',
        data: {
            datasets: [
                {
                    label: `Nombre d'images de ${with_legend ? 2 : '?'}`,
                    data: data_1,
                    backgroundColor: window.mathadata.classColors[0],
                    borderColor: window.mathadata.classColors[0],
                    borderWidth: 1
                },
                {
                    label: `Nombre d'images de ${with_legend ? 7 : '?'}`,
                    data: data_2,
                    backgroundColor: window.mathadata.classColors[1],
                    borderColor: window.mathadata.classColors[1],
                    borderWidth: 1
                }
            ]
        },
        options: {
            scales: {
                x: {
                    type: 'linear',
                    offset: false,
                    grid: {
                        offset: false
                    },
                    ticks: {
                        stepSize: 2,
                    },
                    beginAtZero: true,
                    title: {
                        display: with_axes_legend,
                        text: 'Caractéristique x'
                    }
                }, 
                y: {
                    suggestedMax: 5,
                    title: {
                        display: with_axes_legend,
                        text: "Nombre d'images"
                    },
                },
            },
            barPercentage: 0.9,  // Adjust bar width
            categoryPercentage: 1.0,  // Adjust bar spacing
            grouped: false,
            borderSkipped: 'middle',
            interaction: {
                intersect: false,
                mode: 'index',
            },
        },
    };
    
    if (seuil !== undefined) {
        // Ensure the label for the vertical line is correctly formatted.
        config.data.datasets.push({
            type: 'scatter',
            data: [{x: seuil, y: 0}, {x: seuil, y: 100}],
            showLine: true,
            label: `x = ${seuil}`,
        });
    }
    window.mathadata.create_chart(div_id, config)
}

window.mathadata.animation_histogramme = function(id, params) {
    params = JSON.parse(params)
    const {data, c_train, labels, t} = params

    const max = Math.ceil(Math.max(...c_train) + 1)
    const min = Math.min(0, Math.floor(Math.min(...c_train) - 1))

    mathadata.tracer_droite_carac(`${id}-chart`, params)
    const chart = window.mathadata.charts[`${id}-chart`]

    let i = 0
    chart.data.datasets[1].pointRadius = function(context) {
        var index = context.dataIndex;
        return i === index ? 8 : 4;            
    }
    chart.update()

    window.mathadata.displayHisto(`${id}-histo`, '{}', true, t)
    const histo = window.mathadata.charts[`${id}-histo`]

    const getInitData = () => Array(max - min + 1).fill(0).map((_, i) => ({x: min + i, y: 0}))
    histo.data.datasets[0].data = getInitData()
    histo.data.datasets[1].data = getInitData()
    histo.options.scales.y.suggestedMax = 5
    histo.options.aspectRatio = 5
    histo.update()

    const length = max - min + 1
    
    function updateImage() {
        window.mathadata.affichage(`${id}-data`, data[i])
        clearCarac()
        setTimeout(() => {
            updateCarac()
        }, delay)
    }
    
    function setCarac(x) {
        document.getElementById(`${id}-x`).innerHTML = x
    }
    
    function updateCarac() {
        setCarac(c_train[i].toFixed(2))
        setTimeout(() => {
            updateChart()
        }, delay)
    }
    
    function clearCarac() {
        setCarac('Calcul...')
    }
    
    function updateChart() {
        chart.data.datasets[1].data.push({x: c_train[i], y: 0})
        chart.update()
    
        const histoIndex = Math.floor(c_train[i]) - min
        histo.data.datasets[labels[i]].data[histoIndex].y++
        histo.update()

        i++
        if (i < data.length) {
            if (delay > 0.1) {
                delay = delay *= 0.8 // decrease the delay progressively
                setTimeout(() => {
                    updateImage()
                }, delay)
            } else {
                endAnimation()
            }
        } else {
            // to set all points to normal size
            setTimeout(() => chart.update(), delay)
        }
    }

    function endAnimation() {
        // speed up to display all points
        chart.data.datasets[1].pointRadius = 4
        function updateDisplay() {
            // step 10 by 10 then 100 by 100 until the end
            let end_i
            if (i < 100) {
                end_i = i + 10
            } else {
                end_i = i + 100
            }
            end_i = Math.min(end_i, data.length)

            // update data
            for (i; i < end_i; i++) {
                chart.data.datasets[1].data.push({ x: c_train[i], y: 0 });
                const histoIndex = Math.floor(c_train[i]) - min;
                histo.data.datasets[labels[i]].data[histoIndex].y++;
            }
        
            // update display
            window.mathadata.affichage(`${id}-data`, data[end_i - 1]);
            setCarac(c_train[end_i - 1].toFixed(2));
            chart.update('none');
            histo.update('none');

            if (i < data.length) {
                requestAnimationFrame(updateDisplay);
            }
        }

        requestAnimationFrame(updateDisplay);
    }
        
    let delay = 1000  // Initial delay
    updateImage()
}
''')

### ----- CELLULES VALIDATION ----


### Pour les checks d'execution des cellules sans réponse attendue:
validation_execution_animation_histogramme = MathadataValidate(success="")
validation_execution_afficher_histogramme = MathadataValidate(success="")
validation_score_seuil_optim = MathadataValidate(success="")
validation_execution_caracteristique_custom = MathadataValidate(success="")
validation_score_zone_custom = MathadataValidate(success="")
validation_execution_afficher_customisation = MathadataValidate(success="")
