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


def afficher_histogramme(div_id=None, seuil=None, caracteristique=None, legend=False, aspect_ratio=None, max_y=None, show_seuil_slider=False, show_score=False):
    create_new = div_id is None
    if create_new:
        div_id = uuid.uuid4().hex

    if not caracteristique:
        caracteristique = common.challenge.caracteristique

    data = compute_histogramme(caracteristique)

    # Valeur initiale du seuil (utilisée pour initialiser le slider)
    seuil_initial = seuil
    if seuil_initial is None and show_seuil_slider:
        seuil_initial = 0
        
    # Préparation des données pour le calcul de score dynamique en JS
    c_train_list = []
    labels_list = []
    if show_score:
        c_train_full = compute_c_train(caracteristique, common.challenge.d_train)
        c_train_list = c_train_full.tolist()
        # Mapping des classes vers 0 et 1 pour simplifier le JS
        labels_list = [0 if r == common.challenge.classes[0] else 1 for r in common.challenge.r_train]

    params = {
        'with_legend': legend,
        'with_axes_legend': True,
        'seuil': seuil_initial,
        'aspect_ratio': aspect_ratio,
        'max_y': max_y,
        'show_score': show_score,
        'c_train': c_train_list,
        'labels': labels_list
    }

    if create_new:
        run_js(
            f"mathadata.add_observer('{div_id}', () => window.mathadata.displayHisto('{div_id}', '{json.dumps(data)}', '{json.dumps(params, cls=NpEncoder)}'))")
        
        # Construire le HTML avec ou sans slider selon le paramètre
        slider_html = ''
        if show_seuil_slider:
            slider_html = f'''
                <div id="{div_id}-slider-container" style="padding: 1rem; display: flex; flex-direction: column; align-items: center; gap: 0.5rem;">
                    <label for="{div_id}-slider" style="font-weight: bold;">Seuil t = <span id="{div_id}-seuil-value">{seuil_initial}</span></label>
                    <input type="range" id="{div_id}-slider" min="0" max="78" value="{seuil_initial}" step="1" style="width: 100%; max-width: 600px;">
                </div>
            '''
            
        score_html = ''
        if show_score:
            score_html = f'''
                <div id="{div_id}-score-container" style="text-align: center; padding: 0.5rem; font-size: 1.1rem; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #eee; margin-bottom: 0.5rem;">
                    <strong>Taux d'erreur :</strong> <span id="{div_id}-score-value">--</span>%
                </div>
            '''
        
        display(HTML(f'''
            <div id="{div_id}-wrapper" style="display: flex; flex-direction: column; gap: 1rem;">
                {score_html}
                <canvas id="{div_id}"></canvas>
                {slider_html}
            </div>
        '''))
    else:
        run_js(f"window.mathadata.displayHisto('{div_id}', '{json.dumps(data)}', '{json.dumps(params, cls=NpEncoder)}')")


def animation_histogramme(id=None, carac=None):
    create_new = id is None
    if create_new:
        id = uuid.uuid4().hex

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

    if create_new:
        run_js(
            f"mathadata.add_observer('{id}', () => window.mathadata.animation_histogramme('{id}', '{json.dumps(params, cls=NpEncoder)}'))")
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
    else:
        run_js(f"window.mathadata.animation_histogramme('{id}', '{json.dumps(params, cls=NpEncoder)}'))")


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

    calculer_score(algorithme, cb=cb)


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


# Générateur d'histogrammes interactifs (classification visuelle)
def generer_histogramme_classif(div_id=None, dataset=None, caracteristique=None, aspect_ratio=None):
    
    if caracteristique is None:
        caracteristique = common.challenge.caracteristique
    
    indices_dataset = None
    if dataset is None:
        c_train = compute_c_train(caracteristique, common.challenge.d_train)
        indices_valides = np.where((c_train > 25) & (c_train < 37))[0]
        if len(indices_valides) >= 10:
            indices_dataset = indices_valides[:10]
            dataset = common.challenge.d_train[indices_dataset]
        else:
            print(f"Pas assez d'images dans l'intervalle utilisable! Seulement {len(indices_valides)} disponibles")
            return
    else:
        # Trouver les indices du dataset dans d_train
        indices_dataset = []
        for d in dataset:
            for i, d_train_item in enumerate(common.challenge.d_train):
                if np.array_equal(d, d_train_item):
                    indices_dataset.append(i)
                    break
    
    create_new = div_id is None
    if create_new:
        div_id = uuid.uuid4().hex

    # Calculer les caractéristiques pour le dataset
    c_train = compute_c_train(caracteristique, dataset)
    
    # Obtenir les labels (classes) pour le dataset
    labels = []
    for idx in indices_dataset:
        if common.challenge.r_train[idx] == common.challenge.classes[0]:
            labels.append(0)
        else:
            labels.append(1)
    
    bins = [24, 26, 28, 30, 32, 34, 36, 38]
    max_counts = {k: [0, 0] for k in bins}  # [classe 0, classe 1]
    
    # Calculer max_counts séparément pour chaque classe
    for i, c in enumerate(c_train):
        if not np.isnan(c):
            k = int(c / 2) * 2
            # S'assurer que k est dans notre plage [24, 38]
            if k >= 24 and k <= 38:
                class_idx = labels[i]
                max_counts[k][class_idx] = max_counts[k][class_idx] + 1
    
    # Initialiser les données à 0 pour tous les bins (format [classe 0, classe 1])
    data = {k: [0, 0] for k in bins}

    # Convertir le dataset en liste pour le JSON
    dataset_list = [d.tolist() for d in dataset]

    params = {
        'with_legend': True,
        'with_axes_legend': True,
        'seuil': None,
        'aspect_ratio': aspect_ratio,
        'max_y': 10,
        'x_min': 24,
        'x_max': 38,
        'max_counts': max_counts,
        'labels': labels,  # Passer les labels au JavaScript
        'c_train': c_train.tolist(),  # Passer les valeurs de caractéristiques
        'dataset': dataset_list,  # Passer les images
    }

    # Créer un wrapper avec une div pour l'image au-dessus du canvas
    display(HTML(f'''
        <div id="{div_id}-wrapper" style="display: flex; flex-direction: column; gap: 1rem; align-items: center;">
            <div id="{div_id}-image-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; padding: 1rem; border: 1px solid #ddd; border-radius: 5px; width: 50%;">
                <div id="{div_id}-image" style="width: 150px; height: 150px; aspect-ratio: 1;"></div>
                <div id="{div_id}-carac-info" style="text-align: center;">
                    <p id="{div_id}-carac-text" style="margin: 0; font-size: 1.5rem;">Caractéristique x = <span id="{div_id}-carac-value">Calcul...</span></p>
                    <p id="{div_id}-feedback" style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;"></p>
                </div>
            </div>
            <div id="{div_id}-wrap" style="width: 100%;"><canvas id="{div_id}"></canvas></div>
        </div>
    '''))
    run_js(f"window.mathadata.generateInteractiveHistoClassif('{div_id}', '{json.dumps(data)}', '{json.dumps(params, cls=NpEncoder)}')")


run_js('''
window.mathadata.displayHisto = (div_id, data, params) => {
    if (typeof data === 'string')
        data = JSON.parse(data)

    if (typeof params === 'string')
        params = JSON.parse(params)
        
    const {with_legend, with_axes_legend, seuil, aspect_ratio, max_y, show_score, c_train, labels} = params

    // Variable pour contrôler l'épaisseur de la ligne de seuil (en pixels)
    const SEUIL_LINE_WIDTH = 4;

    const data_1 = Object.entries(data).map(([key, v]) => ({x: parseInt(key) + 1, y: v[0]}))
    const data_2 = Object.entries(data).map(([key, v]) => ({x: parseInt(key) + 1, y: v[1]}))
    
    // Variable pour stocker la valeur actuelle du seuil (utilisée par le plugin)
    let currentSeuilValue = seuil !== undefined && seuil !== null ? seuil : null;

    // Fonction pour créer la configuration du chart
    const createConfig = () => {
        // Plugin pour dessiner la ligne de seuil verticale
        const seuilLinePlugin = {
            id: 'seuilLinePlugin',
            afterDraw: (chart) => {
                if (currentSeuilValue === null || currentSeuilValue === undefined) return;
                
                const ctx = chart.ctx;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                
                if (!xScale || !yScale) return;
                
                // Calculer la position x en pixels pour la valeur du seuil
                const xPixel = xScale.getPixelForValue(currentSeuilValue);
                
                // Obtenir les limites de l'axe Y en pixels (de haut en bas de la zone de dessin)
                const yTop = yScale.top;
                const yBottom = yScale.bottom;
                
                // Dessiner la ligne verticale qui s'étend sur toute la hauteur
                ctx.save();
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
                ctx.lineWidth = SEUIL_LINE_WIDTH;
                ctx.beginPath();
                ctx.moveTo(xPixel, yTop);
                ctx.lineTo(xPixel, yBottom);
                ctx.stroke();

                // Afficher le texte explicatif de part et d'autre de la ligne si t > 18
                if (currentSeuilValue > 18) {
                    ctx.font = 'bold 12px Arial';
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
                    
                    // Texte gauche : x < t => classe 1 (petite carac)
                    ctx.textAlign = 'right';
                    // On récupère le nom de la classe 1 au pluriel
                    const nomClasse1 = mathadata.classe(1, {plural: true});
                    const textLeft = `x < t : considérés comme des ${nomClasse1}`;
                    ctx.fillText(textLeft, xPixel - 8, yTop + 20);
                    
                    // Texte droite : x > t => classe 0 (grande carac)
                    ctx.textAlign = 'left';
                    // On récupère le nom de la classe 0 au pluriel
                    const nomClasse0 = mathadata.classe(0, {plural: true});
                    const textRight = `x > t : considérés comme des ${nomClasse0}`;
                    ctx.fillText(textRight, xPixel + 8, yTop + 20);
                }

                ctx.restore();
            }
        };

        const config = {
            type: 'bar',
            data: {
                datasets: [
                    {
                        label: `Nombre ${with_legend ? mathadata.classe(0, {article: 'de', plural: true, alt: true}) : 'de ?'}`,
                        data: data_1,
                        backgroundColor: window.mathadata.classColors[0],
                        borderColor: window.mathadata.classColors[0],
                        borderWidth: 1
                    },
                    {
                        label: `Nombre ${with_legend ? mathadata.classe(1, {article: 'de', plural: true, alt: true}) : 'de ?'}`,
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
                                return `Nombre ${mathadata.data({article: 'de', plural: true})} avec x entre ${context[0].parsed.x - 1} et ${context[0].parsed.x + 1}`
                            },
                            label: (context) => {
                                return `${with_legend ? `${mathadata.classe(context.datasetIndex)}: ` : ''}${context.parsed.y}`
                            },
                        },
                    },
                },
            },
            plugins: [seuilLinePlugin],
        };
        
        return config;
    };

    // Créer le chart initial
    const config = createConfig();
    window.mathadata.create_chart(div_id, config);
    const chart = window.mathadata.charts[div_id];

    // Configurer le slider si il existe
    const slider = document.getElementById(`${div_id}-slider`);
    const seuilValueEl = document.getElementById(`${div_id}-seuil-value`);
    
    if (slider && seuilValueEl) {
        // Fonction pour mettre à jour le chart avec le nouveau seuil
        const updateSeuil = (newSeuil) => {
            // Mettre à jour la variable globale du seuil
            currentSeuilValue = parseFloat(newSeuil);
            
            // Mettre à jour l'affichage de la valeur
            if (seuilValueEl) {
                seuilValueEl.textContent = newSeuil;
            }

            // Mise à jour du score si activé
            if (show_score && c_train && labels) {
                let errors = 0;
                const total = c_train.length;
                for (let i = 0; i < total; i++) {
                    if (c_train[i] !== null) {
                        // Logique adaptée à MNIST : La classe 0 est le '2' (grande caractéristique moyenne), La classe 1 est le '7' (petite caractéristique moyenne) 
                        // Donc si x <= t (petite valeur), on prédit la classe 1. Sinon, on prédit la classe 0.
                        const pred = c_train[i] <= currentSeuilValue ? 1 : 0;
                        if (pred !== labels[i]) {
                            errors++;
                        }
                    }
                }
                const errorRate = ((errors / total) * 100).toFixed(1);
                const scoreEl = document.getElementById(`${div_id}-score-value`);
                if (scoreEl) {
                    scoreEl.textContent = errorRate;
                    
                    // Feedback visuel simple sur la couleur
                    const container = document.getElementById(`${div_id}-score-container`);
                    if (container) {
                        if (errorRate < 15) container.style.backgroundColor = '#d4edda'; // Vert clair
                        else if (errorRate < 32) container.style.backgroundColor = '#fff3cd'; // Jaune clair
                        else container.style.backgroundColor = '#f8d7da'; // Rouge clair
                    }
                }
            }
            
            // Synchroniser avec la variable Python 't' (comme dans themes/geo/utilitaires.py)
            mathadata.run_python(`t = ${newSeuil}`);
            
            // Mettre à jour le chart (le hook afterDraw redessinera automatiquement la ligne)
            chart.update();
        };

        // Écouter les changements du slider
        slider.addEventListener('input', (e) => {
            const newSeuil = parseFloat(e.target.value);
            updateSeuil(newSeuil);
        });

        // Initialiser la valeur affichée et synchroniser avec Python
        const initialSeuil = seuil !== undefined && seuil !== null ? seuil : 0;
        if (seuilValueEl) {
            seuilValueEl.textContent = initialSeuil;
        }
        // Appel initial pour calculer le score et synchroniser
        updateSeuil(initialSeuil);
    } else {
        // Si pas de slider, s'assurer que la ligne est dessinée avec la valeur initiale
        if (currentSeuilValue !== null && currentSeuilValue !== undefined) {
            chart.update();
        }
    }
}

// Générateur d'histogramme interactif avec boutons + / - par bin (mono-classe) — chart vierge
window.mathadata = window.mathadata || {};
window.mathadata.generateInteractiveHisto = function(div_id, data, params) {
    try {
        if (typeof params === 'string') { try { params = JSON.parse(params) } catch(_) { params = {} } }
        params = params || {}
        const aspect = params.aspect_ratio || 2.0

        // Bins fixes: 0,2,...,88 => centres 1..89
        const keys = Array.from({length: 45}, (_,i)=> i*2)
        const centers = keys.map(k => k + 1)
        const emptyDataset = centers.map(x => ({ x, y: 0 }))

        // Construire un chart vierge directement (sans displayHisto)
        const config = {
            type: 'bar',
            data: { datasets: [{
                label: `Nombre ${window.mathadata.data ? window.mathadata.data('de', {plural: true}) : ''}`,
                data: emptyDataset,
                backgroundColor: 'rgba(100, 149, 237, 0.7)',
                borderColor: 'rgba(100, 149, 237, 1)',
                borderWidth: 1
            }]},
            options: {
                aspectRatio: aspect,
                scales: {
                    x: { type: 'linear', min: 0, max: 90, ticks: { stepSize: 2 }, title: { display: true, text: 'Caractéristique x' } },
                    y: { min: 0, max: 500, ticks: { stepSize: 5 }, title: { display: true, text: `Nombre ${window.mathadata.data ? window.mathadata.data('de', {plural: true}) : ''}` } }
                },
                plugins: { legend: { display: false } },
                barPercentage: 0.9,
                categoryPercentage: 1.0,
                grouped: false,
                borderSkipped: 'middle',
                interaction: { intersect: false, mode: 'index' }
            }
        }

        if (window.mathadata.create_chart) {
            window.mathadata.create_chart(div_id, config)
        } else if (window.Chart) {
            const ctx = document.getElementById(div_id).getContext('2d')
            window.mathadata.charts = window.mathadata.charts || {}
            window.mathadata.charts[div_id] = new Chart(ctx, config)
        } else {
            console.error('[mathadata] Chart.js indisponible')
            return
        }

        const chart = window.mathadata.charts[div_id]

        //  wrapper + calque de contrôle
        const canvas = document.getElementById(div_id);
        if (!canvas) { console.warn(`[mathadata] canvas #${div_id} introuvable`); return; }
        let wrapper = canvas.parentNode;
        if (!wrapper || !(wrapper instanceof HTMLElement) || !wrapper.id || wrapper.id !== `${div_id}-wrap`) {
            const newWrap = document.createElement('div');
            newWrap.id = `${div_id}-wrap`;
            newWrap.style.display = 'block';
            newWrap.style.position = 'relative';
            canvas.parentNode.insertBefore(newWrap, canvas);
            newWrap.appendChild(canvas);
            wrapper = newWrap;
        }
        wrapper.style.position = wrapper.style.position || 'relative';

        // nettoyer anciens calques
        const oldLayer = wrapper.querySelector('.interactive-histo-controls');
        if (oldLayer) oldLayer.remove();

        const layer = document.createElement('div');
        layer.className = 'interactive-histo-controls';
        layer.style.position = 'absolute';
        layer.style.inset = '0';
        layer.style.pointerEvents = 'none';
        layer.style.zIndex = '20';
        wrapper.appendChild(layer);

        // utilitaire création bouton
         const mkBtn = (bg, title, text) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.title = title;
            btn.textContent = text;
            btn.style.position = 'absolute';
            btn.style.width = '15px';
            btn.style.height = '15px';
            btn.style.borderRadius = '999px';
            btn.style.border = '1px solid rgba(0,0,0,0.2)';
             btn.style.boxShadow = '0 1px 3px rgba(0,0,0,0.2)';
            btn.style.background = bg;
            btn.style.color = '#fff';
            btn.style.fontWeight = '700';
             btn.style.fontSize = '10px';
             btn.style.lineHeight = '10px';
            btn.style.pointerEvents = 'auto';
            btn.style.cursor = 'pointer';
            btn.style.display = 'flex';
            btn.style.alignItems = 'center';
            btn.style.justifyContent = 'center';
            btn.style.transition = 'transform 0.12s ease, background-color 0.12s ease';
            btn.onmousedown = () => btn.style.transform = 'scale(0.88)';
            btn.onmouseup = () => btn.style.transform = 'scale(1)';
            btn.onmouseleave = () => btn.style.transform = 'scale(1)';
            btn.setAttribute('aria-label', title || text);
            return btn;
        };

        // indexation x -> index dataset
        const indexByX = {};
        chart.data.datasets[0].data.forEach((pt, i) => { indexByX[String(pt.x)] = i; });

        // Optimisation update (regroupement via RAF)
        let updatePending = false;
        function scheduleChartUpdate() {
            if (!updatePending) {
                updatePending = true;
                requestAnimationFrame(() => {
                    try { chart.update(); } catch(e) { console.warn('[mathadata] chart.update() failed', e) }
                    updatePending = false;
                });
            }
        }

        // Fonction qui place les contrôles dynamiquement
        function placeControls() {
            try {
                // vérifier échelles prêtes
                const xScale = chart.scales && (chart.scales.x || chart.scales['x-axis-0'] || chart.scales['x']);
                const yScale = chart.scales && (chart.scales.y || chart.scales['y-axis-0'] || chart.scales['y']);
                if (!xScale || !yScale) return;

                // nettoyage
                layer.innerHTML = '';

                // déterminer densité et anti-chevauchement
                const availableWidth = xScale.width || canvas.clientWidth || (wrapper.clientWidth || 0);
                const nBins = keys.length || 1;
                const minSpacing = availableWidth / nBins;
                const overlap = minSpacing < 36; // seuil

                keys.forEach(k => {
                    const binCenter = k + 1;
                    const xPx = xScale.getPixelForValue ? xScale.getPixelForValue(binCenter) : ( ( (binCenter - chart.options.scales.x.min) / (chart.options.scales.x.max - chart.options.scales.x.min) ) * availableWidth );
                    const baseY = yScale.getPixelForValue ? yScale.getPixelForValue(yScale.min) : (canvas.clientHeight - 30);

                    const plusBtn = mkBtn('rgba(100,149,237,0.95)', `+1 image`, '+');
                    const minusBtn = mkBtn('rgba(220,68,68,0.95)', `-1 image`, '−');

                    // Décalages ajustés selon densité
                    const offset = overlap ? 8 : 12;
                    // position horizontale (on corrige pour que le bouton soit centré)
                    plusBtn.style.left = `${Math.round(xPx - offset)}px`;
                    minusBtn.style.left = `${Math.round(xPx + (overlap ? 4 : 6))}px`;
                    // position verticale : légèrement au-dessus de l'axe X
                    const axisTop = Math.round(baseY - 26);
                    plusBtn.style.top = `${axisTop}px`;
                    minusBtn.style.top = `${axisTop}px`;

                    // clic + / -
                    plusBtn.onclick = (e) => {
                        e.stopPropagation(); e.preventDefault();
                        const idx = indexByX[String(binCenter)];
                        if (idx == null) return;
                        const val = chart.data.datasets[0].data[idx];
                        val.y = (val.y || 0) + 1;
                        // synchroniser data source (mono-classe : data[k][0])
                        if (Array.isArray(data[k])) data[k][0] = (data[k][0]||0) + 1;
                        else data[k] = (data[k]||0) + 1;
                        scheduleChartUpdate();
                    };
                    minusBtn.onclick = (e) => {
                        e.stopPropagation(); e.preventDefault();
                        const idx = indexByX[String(binCenter)];
                        if (idx == null) return;
                        const val = chart.data.datasets[0].data[idx];
                        val.y = Math.max(0, (val.y || 0) - 1);
                        if (Array.isArray(data[k])) data[k][0] = Math.max(0, (data[k][0]||0) - 1);
                        else data[k] = Math.max(0, (data[k]||0) - 1);
                        scheduleChartUpdate();
                    };

                    layer.appendChild(plusBtn);
                    layer.appendChild(minusBtn);
                });
            } catch (err) {
                console.error('[mathadata] placeControls() erreur :', err);
            }
        }

        // placer initialement
        placeControls();

         // Axes fixes => pas de resize ni rebinds; placer une fois
         placeControls();

        // fin try principal
    } catch (err) {
        console.error('[mathadata] Erreur dans generateInteractiveHisto :', err);
    }
};

// Générateur d'histogramme interactif avec boutons + uniquement (classification visuelle)
window.mathadata.generateInteractiveHistoClassif = function(div_id, data, params) {
    // Parsing des données
    if (typeof data === 'string') { 
        try { data = JSON.parse(data) } 
        catch(_) { 
            console.error('[mathadata] Erreur parsing data:', _);
            return;
        } 
    }
    if (typeof params === 'string') { 
        try { params = JSON.parse(params) } 
        catch(_) { 
            console.error('[mathadata] Erreur parsing params:', _);
            return;
        } 
    }
        params = params || {}
        const aspect = params.aspect_ratio || 2.0
        const x_min = params.x_min || 24
        const x_max = params.x_max || 38
        const max_y = params.max_y || 10
        const max_counts = params.max_counts || {}
        const labels = params.labels || []
        const c_train = params.c_train || []
        const dataset = params.dataset || []
        
    // Vérifier que les éléments DOM existent (attendre qu'ils soient disponibles)
        const imageContainer = document.getElementById(`${div_id}-image`)
        const caracValueEl = document.getElementById(`${div_id}-carac-value`)
        const feedbackEl = document.getElementById(`${div_id}-feedback`)
    
    if (!imageContainer || !caracValueEl || !feedbackEl) {
        // Les éléments DOM ne sont pas encore disponibles, réessayer après un court délai
                                setTimeout(() => {
            window.mathadata.generateInteractiveHistoClassif(div_id, data, params);
        }, 100);
        return;
    }
        
        // État de validation step-by-step
        let currentImageIndex = 0
        const totalImages = dataset.length
        let errorTimeout = null; // Pour annuler le timeout d'erreur si on clique sur le bon bin
        
        // Fonction pour afficher l'image courante
        function displayCurrentImage() {
            if (currentImageIndex >= totalImages) {
                // Toutes les images sont validées
            if (imageContainer) imageContainer.innerHTML = '<p style="text-align: center; margin: auto;">Toutes les images ont été validées !</p>'
            if (caracValueEl) caracValueEl.textContent = 'Terminé'
            if (feedbackEl) {
                feedbackEl.textContent = ''
                feedbackEl.style.color = 'green'
            }
                return
            }
            
            const currentImage = dataset[currentImageIndex]
            const currentCarac = c_train[currentImageIndex]
            const currentLabel = labels[currentImageIndex]
            
            // Afficher l'image
            if (window.mathadata.affichage) {
                window.mathadata.affichage(`${div_id}-image`, currentImage, {})
            }
            
            // Afficher la caractéristique
        if (caracValueEl) caracValueEl.textContent = currentCarac.toFixed(2)
        if (feedbackEl) {
            feedbackEl.textContent = ''
            feedbackEl.style.color = ''
        }
        }
        
        // Afficher la première image
        displayCurrentImage()

        // Bins fixes: 24, 26, 28, 30, 32, 34, 36, 38 => centres 25, 27, 29, 31, 33, 35, 37, 39
        const keys = Array.from({length: 8}, (_,i)=> x_min + i*2)
        const centers = keys.map(k => k + 1)
        
        // Créer deux datasets séparés pour chaque classe
        const initialDataset0 = centers.map(x => {
            const binKey = x - 1; // Le bin correspondant (k)
            const binKeyStr = String(binKey);
            // Les données sont maintenant au format [classe 0, classe 1]
            const dataValue = data[binKeyStr] || data[binKey] || [0, 0];
            const initialValue = Array.isArray(dataValue) ? dataValue[0] : 0;
            return { x, y: initialValue };
        })
        
        const initialDataset1 = centers.map(x => {
            const binKey = x - 1; // Le bin correspondant (k)
            const binKeyStr = String(binKey);
            const dataValue = data[binKeyStr] || data[binKey] || [0, 0];
            const initialValue = Array.isArray(dataValue) ? dataValue[1] : 0;
            return { x, y: initialValue };
        })

        // Construire le chart avec 2 datasets
        const config = {
            type: 'bar',
            data: { 
                datasets: [
                    {
                        label: `Nombre ${window.mathadata.classe ? window.mathadata.classe(0, {article: 'de', plural: true, alt: true}) : 'de ?'}`,
                        data: initialDataset0,
                        backgroundColor: window.mathadata.classColors ? window.mathadata.classColors[0] : 'rgba(100, 149, 237, 0.7)',
                        borderColor: window.mathadata.classColors ? window.mathadata.classColors[0] : 'rgba(100, 149, 237, 1)',
                        borderWidth: 1
                    },
                    {
                        label: `Nombre ${window.mathadata.classe ? window.mathadata.classe(1, {article: 'de', plural: true, alt: true}) : 'de ?'}`,
                        data: initialDataset1,
                        backgroundColor: window.mathadata.classColors ? window.mathadata.classColors[1] : 'rgba(255, 165, 0, 0.7)',
                        borderColor: window.mathadata.classColors ? window.mathadata.classColors[1] : 'rgba(255, 165, 0, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                aspectRatio: aspect,
                scales: {
                    x: { 
                        type: 'linear', 
                        min: x_min - 1, 
                        max: x_max + 1, 
                        // Utiliser afterBuildTicks pour ne garder que les ticks pairs
                        // Cela garantit que les labels et gridlines sont parfaitement alignés
                        afterBuildTicks: function(scale) {
                            // Filtrer pour ne garder que les ticks pairs (24, 26, 28, 30, 32, 34, 36, 38)
                            scale.ticks = scale.ticks.filter(function(tick) {
                                return tick.value % 2 === 0;
                            });
                        },
                        ticks: { 
                            stepSize: 1, // Pas de 1 pour générer tous les ticks, puis on filtre dans afterBuildTicks
                            // Les labels seront automatiquement alignés avec les gridlines
                            // car on ne garde que les ticks pairs dans afterBuildTicks
                            align: 'center', // Centrer les labels sur les ticks
                            autoSkip: false, // Ne pas sauter automatiquement de ticks
                            maxRotation: 0, // Pas de rotation
                            minRotation: 0
                        },
                        grid: {
                            // Les gridlines seront automatiquement alignées avec les ticks pairs
                            // car afterBuildTicks a filtré les ticks
                            color: 'rgba(0, 0, 0, 0.1)', // Couleur uniforme pour tous les ticks pairs
                            drawOnChartArea: true,
                            drawTicks: true,
                            offset: false // Pas d'offset pour un alignement parfait
                        },
                        offset: false, // Pas d'offset sur l'axe pour un alignement parfait
                        title: { display: true, text: 'Caractéristique x' } 
                    },
                    y: { 
                        min: 0, 
                        max: max_y, 
                        ticks: { stepSize: 1 }, 
                        title: { display: true, text: `Nombre ${window.mathadata.data ? window.mathadata.data('de', {plural: true}) : ''}` } 
                    }
                },
                plugins: { legend: { display: true } },
                barPercentage: 0.9,
                categoryPercentage: 1.0,
                grouped: false,
                borderSkipped: 'middle',
                interaction: { intersect: false, mode: 'index' }
            }
        }

        if (window.mathadata.create_chart) {
            window.mathadata.create_chart(div_id, config)
        } else if (window.Chart) {
            const ctx = document.getElementById(div_id).getContext('2d')
            window.mathadata.charts = window.mathadata.charts || {}
            window.mathadata.charts[div_id] = new Chart(ctx, config)
        } else {
            console.error('[mathadata] Chart.js indisponible')
            return
        }

        const chart = window.mathadata.charts[div_id]

        // --- Préparer wrapper + calque de contrôle ---
        const canvas = document.getElementById(div_id);
        if (!canvas) { console.warn(`[mathadata] canvas #${div_id} introuvable`); return; }
        let wrapper = canvas.parentNode;
        if (!wrapper || !(wrapper instanceof HTMLElement) || !wrapper.id || wrapper.id !== `${div_id}-wrap`) {
            const newWrap = document.createElement('div');
            newWrap.id = `${div_id}-wrap`;
            newWrap.style.display = 'block';
            newWrap.style.position = 'relative';
            canvas.parentNode.insertBefore(newWrap, canvas);
            newWrap.appendChild(canvas);
            wrapper = newWrap;
        }
        wrapper.style.position = wrapper.style.position || 'relative';

        // nettoyer anciens calques
        const oldLayer = wrapper.querySelector('.interactive-histo-controls');
        if (oldLayer) oldLayer.remove();

        const layer = document.createElement('div');
        layer.className = 'interactive-histo-controls';
        layer.style.position = 'absolute';
        layer.style.inset = '0';
        layer.style.pointerEvents = 'none';
        layer.style.zIndex = '20';
        wrapper.appendChild(layer);

        // Taille des boutons (modifiable ici)
        const BUTTON_SIZE = 20; // en pixels
        const BUTTON_FONT_SIZE = 20; // en pixels
        const BUTTON_VERTICAL_OFFSET = 15; // Position verticale : distance depuis l'axe X (en pixels, positif = en dessous, négatif = au-dessus)

        // utilitaire création bouton
        const mkBtn = (bg, title, text) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.title = title;
            btn.textContent = text;
            btn.style.position = 'absolute';
            btn.style.width = BUTTON_SIZE + 'px';
            btn.style.height = BUTTON_SIZE + 'px';
            btn.style.borderRadius = '999px';
            btn.style.border = '1px solid rgba(0,0,0,0.2)';
            btn.style.boxShadow = '0 1px 3px rgba(0,0,0,0.2)';
            btn.style.background = bg;
            btn.style.color = '#fff';
            btn.style.fontWeight = '700';
            btn.style.fontSize = BUTTON_FONT_SIZE + 'px';
            btn.style.lineHeight = BUTTON_SIZE + 'px';
            btn.style.pointerEvents = 'auto';
            btn.style.cursor = 'pointer';
            btn.style.display = 'flex';
            btn.style.alignItems = 'center';
            btn.style.justifyContent = 'center';
            btn.style.transition = 'transform 0.12s ease, background-color 0.12s ease';
            btn.onmousedown = () => btn.style.transform = 'scale(0.88)';
            btn.onmouseup = () => btn.style.transform = 'scale(1)';
            btn.onmouseleave = () => btn.style.transform = 'scale(1)';
            btn.setAttribute('aria-label', title || text);
            return btn;
        };

        // indexation x -> index dataset
        const indexByX = {};
        chart.data.datasets[0].data.forEach((pt, i) => { indexByX[String(pt.x)] = i; });
        
        // Créer un mapping bin -> classe pour déterminer quelle classe incrémenter
        // Pour chaque bin, trouver la première image qui correspond à ce bin
        const binToClass = {};
        for (let imgIdx = 0; imgIdx < c_train.length; imgIdx++) {
            const c = c_train[imgIdx];
            if (!isNaN(c)) {
                const k = Math.floor(c / 2) * 2;
                if (k >= x_min && k <= x_max) {
                    const binKeyStr = String(k);
                    if (!binToClass.hasOwnProperty(binKeyStr)) {
                        binToClass[binKeyStr] = labels[imgIdx];
                    }
                }
            }
        }

    // Fonction qui place les contrôles dynamiquement
        function placeControls() {
                // vérifier échelles prêtes
                const xScale = chart.scales && (chart.scales.x || chart.scales['x-axis-0'] || chart.scales['x']);
                const yScale = chart.scales && (chart.scales.y || chart.scales['y-axis-0'] || chart.scales['y']);
                if (!xScale || !yScale) return;

                // nettoyage
                layer.innerHTML = '';

                // Placer les boutons + directement sur les positions des nombres impairs (centres des bins)
                // Les nombres impairs (25, 27, 29, 31, 33, 35, 37, 39) correspondent aux centres des bins
                for (let i = 0; i < keys.length; i++) {
                    const currentBin = keys[i];
                    const binCenter = currentBin + 1; // Nombre impair : centre du bin (25, 27, 29, etc.)
                    
                    // Position exacte du nombre impair (centre du bin) sur l'axe X
                    const xPx = xScale.getPixelForValue ? xScale.getPixelForValue(binCenter) : 
                        ((binCenter - chart.options.scales.x.min) / (chart.options.scales.x.max - chart.options.scales.x.min)) * (xScale.width || canvas.clientWidth);
                    
                    // Position en dessous de l'axe X (à la place de la graduation du nombre impair)
                    const baseY = yScale.getPixelForValue ? yScale.getPixelForValue(yScale.min) : (canvas.clientHeight - 30);
                    const axisBottom = Math.round(baseY + BUTTON_VERTICAL_OFFSET);

                    // Déterminer quelle classe incrémenter pour ce bin
                    const binKeyStr = String(currentBin);
                    const classIdx = binToClass[binKeyStr] !== undefined ? binToClass[binKeyStr] : 0;
                    
                    // Bouton uniforme vert pour ne pas donner d'indication
                    const plusBtn = mkBtn('rgba(76,175,80,0.95)', `+1 image`, '+');
                    plusBtn.style.left = `${Math.round(xPx - BUTTON_SIZE / 2)}px`; // Centrer le bouton
                    plusBtn.style.top = `${axisBottom}px`;

                    // clic + avec vérification de limite et validation step-by-step
                    plusBtn.onclick = (e) => {
                        e.stopPropagation(); 
                        e.preventDefault();
                        
                        // Si toutes les images sont validées, ne rien faire
                        if (currentImageIndex >= totalImages) return;
                        
                        // Déterminer quel bin incrémenter (le bin correspondant au centre)
                        const idx = indexByX[String(binCenter)];
                        if (idx == null) return;
                        
                        // Récupérer les informations de l'image courante
                        const currentImageCarac = c_train[currentImageIndex]
                        const currentImageLabel = labels[currentImageIndex]
                        
                        // Déterminer le bin de l'image courante
                        const currentImageBin = Math.floor(currentImageCarac / 2) * 2
                        const currentImageBinStr = String(currentImageBin)
                        
                        // Vérifier si le bouton cliqué correspond au bon bin
                        const isCorrectBin = currentImageBinStr === binKeyStr
                        
                        // Si le bin correspond, on valide (peu importe la classe mappée au bouton)
                        // car avec la validation step-by-step, on s'assure que l'utilisateur clique
                        // sur le bon bouton pour l'image courante
                        if (isCorrectBin) {
                            // Annuler le timeout d'erreur s'il existe (si on avait cliqué sur le mauvais bin avant)
                            if (errorTimeout !== null) {
                                clearTimeout(errorTimeout)
                                errorTimeout = null
                            }
                            
                            // Validation correcte - utiliser la classe de l'image courante
                            const datasetIdx = currentImageLabel;
                            const val = chart.data.datasets[datasetIdx].data[idx];
                            const currentCount = val.y || 0;
                            
                            // Les clés JSON sont des strings, donc on doit convertir
                            const maxCountArray = max_counts[binKeyStr] || max_counts[currentBin] || [0, 0];
                            const maxCount = Array.isArray(maxCountArray) ? maxCountArray[datasetIdx] : maxCountArray;
                            
                            // Vérifier si on peut encore incrémenter
                            if (currentCount < maxCount) {
                                val.y = currentCount + 1;
                                // Mettre à jour data au format [classe 0, classe 1]
                                const dataValue = data[binKeyStr] || data[currentBin] || [0, 0];
                                const newDataValue = Array.isArray(dataValue) ? [...dataValue] : [0, 0];
                                newDataValue[datasetIdx] = (newDataValue[datasetIdx] || 0) + 1;
                                data[binKeyStr] = newDataValue;
                        // Mise à jour immédiate
                        chart.update('none'); // 'none' pour une mise à jour sans animation
                                
                                // Afficher le feedback positif immédiatement (remplace le message d'erreur si présent)
                        if (feedbackEl) {
                                feedbackEl.textContent = 'Très bien !'
                                feedbackEl.style.color = 'green'
                        }
                                
                                // Passer à l'image suivante après un court délai
                                setTimeout(() => {
                                    currentImageIndex++
                                    displayCurrentImage()
                                }, 1500)
                            }
                        } else {
                            // Validation incorrecte
                            // Annuler le timeout précédent s'il existe
                            if (errorTimeout !== null) {
                                clearTimeout(errorTimeout)
                            }
                            
                    if (feedbackEl) {
                        feedbackEl.textContent = 'Mauvaise caractéristique, réessayez !'
                        feedbackEl.style.color = 'red'
                        
                        // Effacer le message après 8 secondes
                        errorTimeout = setTimeout(() => {
                            if (feedbackEl && feedbackEl.textContent === 'Mauvaise caractéristique, réessayez !') {
                                feedbackEl.textContent = ''
                            }
                            errorTimeout = null
                        }, 8000)
                    }
                        }
                    };

                    layer.appendChild(plusBtn);
            }
        }

        // Placer les contrôles après que le chart soit prêt
        setTimeout(() => {
            placeControls();
        }, 100);
};



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


# Alias pour l'exercice avec le slider et le taux d'erreur affiché
validation_afficher_histo_score_seuil_optim = validation_question_hist_seuil
