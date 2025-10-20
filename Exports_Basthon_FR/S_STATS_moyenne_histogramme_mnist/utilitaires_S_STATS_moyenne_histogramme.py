import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common


def compute_histogramme(caracteristique):
    c_train = compute_c_train(caracteristique, common.challenge.d_train)
    data = {}
    for i in range(len(c_train)):
        c = c_train[i]
        if not np.isnan(c):
            k = int(c / 2) * 2
            if k not in data:
                data[k] = [0, 0]

            if common.challenge.r_train[i] == common.challenge.classes[0]:
                data[k][0] += 1
            else:
                data[k][1] += 1

    return data


def afficher_histogramme(div_id=None, seuil=None, caracteristique=None, legend=False, aspect_ratio=None, max_y=None):
    if div_id is None:
        div_id = uuid.uuid4().hex
        display(HTML(f'<canvas id="{div_id}"></canvas>'))

    if not caracteristique:
        caracteristique = common.challenge.caracteristique

    data = compute_histogramme(caracteristique)

    params = {
        'with_legend': legend,
        'with_axes_legend': True,
        'seuil': seuil,
        'aspect_ratio': aspect_ratio,
        'max_y': max_y,
    }

    run_js(
        f"setTimeout(() => window.mathadata.displayHisto('{div_id}', '{json.dumps(data)}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def animation_histogramme(id=None, carac=None):
    if id is None:
        id = uuid.uuid4().hex
        display(HTML(f'''
            <div id="{id}" style="display: flex; gap: 2rem; height: 300px; width: 100%;">
                <div id="{id}-data" style="width: 300px; height: 300px;"></div>
                <div style="height: 300px; flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: space-around; height: 100%; gap: 1rem;">
                    <p style="text-align: center">Caractéristique x&nbsp;=&nbsp;<span id="{id}-x">Calcul...</span></p>
                    <div style="flex: 1; width: 100%">
                        <canvas id="{id}-histo"></canvas>
                        <canvas id="{id}-chart"></canvas>
                    </div>
                </div>
            </div>
        '''))

    if carac is None:
        carac = common.challenge.caracteristique

    d_train = common.challenge.d_train
    r_train = common.challenge.r_train
    c_train = compute_c_train(carac, d_train)
    labels = [0 if r == common.challenge.classes[0] else 1 for r in r_train]

    params = {
        'c_train': c_train,
        'labels': labels,
    }

    run_js(
        f"setTimeout(() => window.mathadata.animation_histogramme('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


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

    calculer_score(algorithme, method="moyenne ref hist", parameters=f"t={t}", cb=cb)


def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
        <canvas id="{id}-histo" style="margin-top: 1rem;"></canvas>
    '''))
    common.challenge.display_custom_selection(id)

    params = {
        'with_legend': True,
        'with_axes_legend': True,
    }

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (res) => {{
                window.mathadata.displayHisto('{id}-histo', res, '{json.dumps(params, cls=NpEncoder)}')
            }})
        }}
    ''')


def update_custom():
    return json.dumps(compute_histogramme(common.challenge.caracteristique_custom), cls=NpEncoder)


run_js('''
window.mathadata.displayHisto = (div_id, data, params) => {
    if (typeof data === 'string')
        data = JSON.parse(data)

    if (typeof params === 'string')
        params = JSON.parse(params)
        
    const {with_legend, with_axes_legend, seuil, aspect_ratio, max_y} = params

    const data_1 = Object.entries(data).map(([key, v]) => ({x: parseInt(key) + 1, y: v[0]}))
    const data_2 = Object.entries(data).map(([key, v]) => ({x: parseInt(key) + 1, y: v[1]}))
    const config = {
        type: 'bar',
        data: {
            datasets: [
                {
                    label: `Nombre ${mathadata.data('de', {plural: true})} de ${with_legend ? mathadata.challenge.strings['classes'][0] : '?'}`,
                    data: data_1,
                    backgroundColor: window.mathadata.classColors[0],
                    borderColor: window.mathadata.classColors[0],
                    borderWidth: 1
                },
                {
                    label: `Nombre ${mathadata.data('de', {plural: true})} de ${with_legend ? mathadata.challenge.strings['classes'][1] : '?'}`,
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
                    max: max_y,
                    title: {
                        display: with_axes_legend,
                        text: `Nombre ${mathadata.data('de', {plural: true})}`
                    },
                },
            },
            aspectRatio: aspect_ratio,
            barPercentage: 0.9,  // Adjust bar width
            categoryPercentage: 1.0,  // Adjust bar spacing
            grouped: false,
            borderSkipped: 'middle',
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        title: (context) => {
                            return `Nombre ${mathadata.data('de', {plural: true})} avec x entre ${context[0].parsed.x - 1} et ${context[0].parsed.x + 1}`
                        },
                        label: (context) => {
                            return `${with_legend ? `${mathadata.classes[context.datasetIndex]}: ` : ''}${context.parsed.y}`
                        },
                    },
                },
            },
        },
    };
    
    if (seuil !== undefined && seuil !== null) {
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

window.mathadata.animation_histogramme = function(id, paramsJson) {
    const params = JSON.parse(paramsJson)
    const {c_train, labels, t} = params

    const max = Math.ceil(Math.max(...c_train) + 1)
    const min = Math.min(0, Math.floor(Math.min(...c_train) - 1))

    mathadata.tracer_droite_carac(`${id}-chart`, params)
    const chart = window.mathadata.charts[`${id}-chart`]

    let i = 0
    const pointRadius = (datasetIndex) => {
        return function(context) {
            return labels[i] === datasetIndex && context.dataIndex === chart.data.datasets[datasetIndex].data.length - 1 ? 8 : 4
        }
    }
    
    chart.data.datasets[0].pointRadius = pointRadius(0)
    chart.data.datasets[1].pointRadius = pointRadius(1)
    chart.update()

    window.mathadata.displayHisto(`${id}-histo`, '{}', {
        with_legend: true,
        seuil: t,
    })
    const histo = window.mathadata.charts[`${id}-histo`]

    const getInitData = () => Array(max - min + 1).fill(0).map((_, i) => ({x: min + i, y: 0}))
    histo.data.datasets[0].data = getInitData()
    histo.data.datasets[1].data = getInitData()
    histo.options.scales.y.suggestedMax = 5
    histo.options.aspectRatio = 5
    histo.update()

    const length = max - min + 1
    
    function updateImage() {
        mathadata.run_python(`get_data(${i})`, (data) => {
            window.mathadata.affichage(`${id}-data`, data)
        })
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

    let blockSize = 1
    
    function updateChart() {
        let j = i
        while (j < c_train.length && j < i + blockSize) {
            const datasetIndex = labels[j]
            chart.data.datasets[datasetIndex].data.push({x: c_train[j], y: 0})
    
            const histoIndex = Math.floor(c_train[j]) - min
            histo.data.datasets[labels[j]].data[histoIndex].y++
            j++
        }
        
        chart.update()
        histo.update()

        i = j
        if (i < c_train.length) {
            if (delay > 0.1) {
                // On commence par diminuer le delay progressivement
                delay = delay *= 0.8
            } else {
                // A partir d'un certain point, on ajoute plusieurs points à la fois
                blockSize += 1
                
                // Remplace la fonction qui affiche le dernier point en plus gros par une valeur fixe pour faire moins de calculs
                chart.data.datasets[0].pointRadius = 4
                chart.data.datasets[1].pointRadius = 4
            }

            setTimeout(() => {
                updateImage()
            }, delay)

        } else {
            // affiche la dernière image
            mathadata.run_python(`get_data(${i} - 1)`, (data) => {
                window.mathadata.affichage(`${id}-data`, data)
                setCarac(c_train[i - 1].toFixed(2))
            })
        
            // to set all points to normal size
            setTimeout(() => chart.update(), delay)
        }
    }

    let delay = 2000  // Initial delay
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


def on_success_question_hist_1(answers):
    if common.challenge.carac_explanation:
        pretty_print_success(common.challenge.carac_explanation)
    else:
        pretty_print_success("Bravo, c'est la bonne réponse !")


validation_question_hist_1 = MathadataValidateVariables({
    'r_histogramme_orange': {
        'value': common.challenge.classes[1],
        'errors': [
            {
                'value': {
                    'in': common.challenge.classes,
                },
                'else': f"r_histogramme_orange n'a pas la bonne valeur. Tu dois répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
            }
        ]
    },
    'r_histogramme_bleu': {
        'value': common.challenge.classes[0],
        'errors': [
            {
                'value': {
                    'in': common.challenge.classes,
                },
                'else': f"r_histogramme_bleu n'a pas la bonne valeur. Tu dois répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
            }
        ]
    }
}, 
 success="", on_success=on_success_question_hist_1)


def get_names_and_values_question_hist_seuil():
    data = compute_histogramme(common.challenge.caracteristique)
    # Find the intersection point where the histograms cross
    # data is a dictionary with keys like '0', '2', '4' and values like [10, 20]
    # We need to find where the order between first and second number changes

    # Convert keys to integers and sort them
    sorted_keys = sorted(data.keys())

    # Find where the histograms cross (where the sign of the difference changes)
    intersection_point = None
    for i in range(len(sorted_keys) - 1):
        current_key = sorted_keys[i]
        next_key = sorted_keys[i + 1]

        # Check if the sign of the difference changes
        current_diff = data[current_key][0] - data[current_key][1]
        next_diff = data[next_key][0] - data[next_key][1]

        if (current_diff > 0 and next_diff <= 0) or (current_diff < 0 and next_diff >= 0):
            # Found the intersection between these two points
            intersection_point = sorted_keys[i + 1]
            break
    else:
        raise ValueError("Il y a eu un problème lors de la recherche du point d'intersection des histogrammes.")

    return {
        't': {
            'value': intersection_point,
            'errors': [
                {
                    'value': {
                        'max': intersection_point - 4,
                    },
                    'if': "Ton seuil t est trop bas. Regarde ou les 2 histogrammes se croisent pour trouver le meilleur seuil."
                },
                {
                    'value': {
                        'min': intersection_point + 4,
                    },
                    'if': "Ton seuil t est trop haut. Regarde ou les 2 histogrammes se croisent pour trouver le meilleur seuil."
                },
                {
                    'value': {
                        'min': intersection_point - 4,
                        'max': intersection_point + 4,
                    },
                    'if': f'Tu te rapproches mais ce n\'est pas le meilleur seuil. Il doit y avoir plus de {common.challenge.r_petite_caracteristique} que de {common.challenge.r_grande_caracteristique} qui ont une caractéristique x inférieure ou égale à t et inversement pour x supérieur à t.'
                },

            ]
        }
    }


validation_question_hist_seuil = MathadataValidateVariables(
    get_names_and_values=get_names_and_values_question_hist_seuil, success="Bravo, ton seuil est maintenant optimal !")
