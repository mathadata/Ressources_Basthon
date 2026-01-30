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
    create_new = div_id is None
    if create_new:
        div_id = uuid.uuid4().hex

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

    if create_new:
        run_js(
            f"mathadata.add_observer('{div_id}', () => window.mathadata.displayHisto('{div_id}', '{json.dumps(data)}', '{json.dumps(params, cls=NpEncoder)}'))")
        display(HTML(f'<canvas id="{div_id}"></canvas>'))
    else:
        run_js(f"window.mathadata.displayHisto('{div_id}', '{json.dumps(data)}', '{json.dumps(params, cls=NpEncoder)}'))")


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
            <div id="{id}" style="display: flex; gap: 2rem; height: 300px; width: 100%; align-items: center;">
                <div id="{id}-data" style="width: 300px; height: 300px; display: flex; align-items: center; justify-content: center;"></div>
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
def generer_histogramme_classif(div_id=None, dataset=None, caracteristique=None, aspect_ratio=None, checkpoint_id='generer_histogramme_classif'):
    
    if caracteristique is None:
        caracteristique = common.challenge.caracteristique
    
    indices_dataset = None
    if dataset is None:
        c_train = compute_c_train(caracteristique, common.challenge.d_train)
        r_train = common.challenge.r_train
        classes = common.challenge.classes
        
        # Fonction helper pour trouver n indices respectant les critères
        def get_indices(min_val, max_val, target_class, count, exclude_indices=[]):
            candidates = np.where(
                (c_train >= min_val) & 
                (c_train < max_val) & 
                (r_train == target_class) & 
                (~np.isin(np.arange(len(c_train)), exclude_indices))
            )[0]
            if len(candidates) < count:
                print(f"Attention: Pas assez de candidats pour {target_class} dans [{min_val}, {max_val}[. Demandé: {count}, Trouvé: {len(candidates)}")
                return candidates.tolist()
            return candidates[:count].tolist()

        selected_indices = []
        
        # 1 image de 7 (classe 1) entre 24 et 26
        selected_indices.extend(get_indices(24, 26, classes[1], 1, selected_indices))
        
        # 2 images de 7 (classe 1) entre 26 et 28
        selected_indices.extend(get_indices(26, 28, classes[1], 2, selected_indices))
        
        # 3 images de 7 (classe 1) entre 28 et 30
        selected_indices.extend(get_indices(28, 30, classes[1], 3, selected_indices))
        
        # 1 image de 2 (classe 0) entre 28 et 30
        selected_indices.extend(get_indices(28, 30, classes[0], 1, selected_indices))
        
        # 1 image de 7 (classe 1) entre 30 et 32
        selected_indices.extend(get_indices(30, 32, classes[1], 1, selected_indices))
        
        # 3 images de 2 (classe 0) entre 30 et 32
        selected_indices.extend(get_indices(30, 32, classes[0], 3, selected_indices))
        
        # 2 images de 2 (classe 0) entre 32 et 34
        selected_indices.extend(get_indices(32, 34, classes[0], 2, selected_indices))
        
        # 1 image de 2 (classe 0) entre 34 et 36
        selected_indices.extend(get_indices(34, 36, classes[0], 1, selected_indices))
        
        if len(selected_indices) > 0:
            # Mélanger les indices pour que l'ordre de validation soit aléatoire
            np.random.shuffle(selected_indices)
            indices_dataset = selected_indices
            dataset = common.challenge.d_train[indices_dataset]
        else:
            print("Impossible de générer le dataset par défaut.")
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
        'checkpoint_id': checkpoint_id,  # ID du checkpoint pour la sauvegarde
    }

    # Créer un wrapper avec une div pour l'image au-dessus du canvas
    display(HTML(f'''
        <div id="{div_id}-wrapper" style="display: flex; flex-direction: column; gap: 1rem; align-items: center;">
            <div id="{div_id}-image-container" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; padding: 1rem; border: 1px solid #ddd; border-radius: 5px; width: 50%;">
                <div id="{div_id}-image" style="width: 150px; height: 150px; aspect-ratio: 1;"></div>
                <div id="{div_id}-carac-info" style="text-align: center;">
                    <p id="{div_id}-carac-text" style="margin: 0; font-size: 1.5rem;">Caractéristique x = <span id="{div_id}-carac-value">Calcul...</span></p>
                    <p id="{div_id}-feedback" style="margin: 0.5rem 0 0 0; font-size: 1.5rem;"></p>
                </div>
            </div>
            <div id="{div_id}-wrap" style="width: 100%;"><canvas id="{div_id}"></canvas></div>
            <p style="text-align: center;  margin: 0.5rem 0; color: #333;">
                Cliquez sur l'intervalle correspondant à la caractéristique de chaque image pour construire l'histogramme
            </p>
        </div>
    '''))
    run_js(f"window.mathadata.add_observer('{div_id}', () => mathadata.generateInteractiveHistoClassif('{div_id}', '{json.dumps(data)}', '{json.dumps(params, cls=NpEncoder)}'))")


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
                        autoSkip: false,
                        includeBounds: false,
                        maxRotation: 0,
                        align: 'center',
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
    const checkpointId = params.checkpoint_id || 'generer_histogramme_classif'

    // Vérifier que les éléments DOM existent (attendre qu'ils soient disponibles)
    const imageContainer = document.getElementById(`${div_id}-image`)
    const caracValueEl = document.getElementById(`${div_id}-carac-value`)
    const feedbackEl = document.getElementById(`${div_id}-feedback`)

        
        // État de validation step-by-step
        let currentImageIndex = 0
        const totalImages = dataset.length
        
        // Fonction pour afficher l'image courante
        function displayCurrentImage() {
            if (currentImageIndex >= totalImages) {
                // Toutes les images sont validées
                if (imageContainer) imageContainer.innerHTML = ''
                if (caracValueEl) caracValueEl.textContent = ''
                if (feedbackEl) {
                    feedbackEl.textContent = `🎉 Bravo ! Tu as construit un petit histogramme en ajoutant les images une par une !`
                    feedbackEl.style.color = 'green'

                    // Sauvegarder le checkpoint
                    window.mathadata.checkpoints.save(checkpointId)

                    mathadata.pass_breakpoint()
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
                feedbackEl.textContent = `Image ${currentImageIndex + 1} sur ${totalImages}`
                feedbackEl.style.color = '#666'
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
        const BUTTON_SIZE = 24; // en pixels
        const BUTTON_FONT_SIZE = 16; // Taille de la police dans le bouton (en pixels)
        const BUTTON_VERTICAL_OFFSET = 8; // Position verticale : distance depuis l'axe X (en pixels, positif = en dessous, négatif = au-dessus)

        // utilitaire création bouton
        const mkBtn = (bg, title, text) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.title = title;
            btn.innerHTML = `<p style="vertical-align: middle;">${text}</p>`;
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
                        
                        // Validation step-by-step: vérifier que le bin cliqué correspond à l'image affichée
                        if (isCorrectBin) {
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
                                chart.update();

                                // Afficher le feedback positif
                                if (feedbackEl) {
                                    feedbackEl.textContent = "✓ Bravo !"
                                    feedbackEl.style.color = 'green'
                                }

                                // Passer à l'image suivante immédiatement
                                currentImageIndex++
                                displayCurrentImage()
                            }
                        } else {
                            // Validation incorrecte
                            if (feedbackEl) {
                                feedbackEl.textContent = "✗ L'image n'appartient pas à cet intervalle, réessaie !"
                                feedbackEl.style.color = 'red'
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

        // Vérifier si cet histogramme a déjà été complété
        const isAlreadyCompleted = window.mathadata.checkpoints.check(checkpointId);

        // Si déjà complété, afficher le message et passer le breakpoint
        if (isAlreadyCompleted) {
            if (feedbackEl) {
                feedbackEl.textContent = '✓ Tu as déjà complété cet histogramme précédemment. Tu peux continuer.'
                feedbackEl.style.color = 'green'
                feedbackEl.style.fontStyle = 'italic'
            }
            mathadata.pass_breakpoint()
        }
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

    // Créer des bins de largeur 2: le bin i représente les valeurs [2i, 2i+2[
    // La barre est centrée sur 2i+1 (ex: bin 0 centré sur 1, bin 1 centré sur 3, etc.)
    const numBins = Math.ceil((max - min) / 2)
    const getInitData = () => Array(numBins).fill(0).map((_, i) => ({x: min + i * 2 + 1, y: 0}))
    histo.data.datasets[0].data = getInitData()
    histo.data.datasets[1].data = getInitData()
    histo.options.scales.y.suggestedMax = 5
    histo.options.aspectRatio = 5
    histo.update()

    // Configurer les graduations du chart aussi
    chart.options.scales.x.ticks = {
        stepSize: 2,
        autoSkip: false,
        includeBounds: false,
        maxRotation: 0,
        align: 'center',
    };
    chart.update()

    const length = max - min + 1

    // Configuration de l'accélération (même stratégie que animateClassification)
    const initialAnimDuration = 3000;  // Début: 3 secondes par point
    const minAnimDuration = 100;        // Min: 100ms pour animation rapide
    const accelerationFactor = 1.1;    // Facteur d'accélération

    let isAnimationRunning = true;
    let isAnimationEnding = false;
    const totalItems = c_train.length;
    const animContainer = document.getElementById(id);

    function setCarac(x) {
        document.getElementById(`${id}-x`).innerHTML = x
    }

    function clearCarac() {
        setCarac('Calcul...')
    }

    function updateImage() {
        if (!isAnimationRunning || i >= totalItems) return;

        mathadata.run_python(`get_data(${i})`, (data) => {
            window.mathadata.affichage(`${id}-data`, data)
        })
        clearCarac()
    }

    function updateCarac() {
        if (!isAnimationRunning || i >= totalItems) return;
        setCarac(c_train[i].toFixed(2))
    }
    
    function showPlusOne(histoIndex, datasetIndex) {
        // Obtenir le canvas de l'histogramme
        const histoCanvas = document.getElementById(`${id}-histo`);
        const rect = histoCanvas.getBoundingClientRect();

        // Calculer la position approximative de la barre
        const chartArea = histo.chartArea;
        const xScale = histo.scales.x;
        const yScale = histo.scales.y;

        // Position x de la barre (utiliser la valeur x centrée sur les bins de largeur 2)
        const barX = xScale.getPixelForValue(min + histoIndex * 2 + 1);
        // Position y en haut de la barre actuelle
        const currentValue = histo.data.datasets[datasetIndex].data[histoIndex].y;
        const barY = yScale.getPixelForValue(currentValue);

        // Créer l'élément +1
        const plusOne = document.createElement('div');
        plusOne.textContent = '+1';
        plusOne.style.position = 'absolute';
        plusOne.style.left = (rect.left + barX) + 'px';
        plusOne.style.top = (rect.top + barY - 10) + 'px';
        plusOne.style.color = datasetIndex === 0 ? 'rgba(80,80,255,1)' : 'rgba(255, 165, 0, 1)';
        plusOne.style.fontWeight = 'bold';
        plusOne.style.fontSize = '14px';
        plusOne.style.pointerEvents = 'none';
        plusOne.style.zIndex = '50';
        plusOne.style.transition = 'all 1500ms ease-out';
        plusOne.style.opacity = '1';

        document.body.appendChild(plusOne);

        // Animer le +1
        requestAnimationFrame(() => {
            plusOne.style.transform = 'translateY(-20px)';
            plusOne.style.opacity = '0';
        });

        // Supprimer après l'animation
        setTimeout(() => {
            plusOne.remove();
        }, 1500);
    }

    function updateCharts() {
        const datasetIndex = labels[i]
        chart.data.datasets[datasetIndex].data.push({x: c_train[i], y: 0})

        // Calculer le bin de largeur 2: valeur 23.5 -> bin 11 (car 23.5 est dans [22, 24[)
        // Pour une valeur v et min=0: bin = floor(v/2), index = floor((v-min)/2)
        const histoIndex = Math.floor((c_train[i] - min) / 2)
        histo.data.datasets[labels[i]].data[histoIndex].y++


        chart.update()
        histo.update()

        if (!isAnimationEnding) {
            // Animation +1 a coté du bins de l'histogramme
            showPlusOne(histoIndex, datasetIndex);
        }
    }

    function processData() {
        if (!isAnimationRunning || i >= totalItems) return;

        const animDuration = Math.max(minAnimDuration, initialAnimDuration / Math.pow(accelerationFactor, i));

        // Détection du passage en mode avance rapide
        if (animDuration <= minAnimDuration && !isAnimationEnding) {
            isAnimationEnding = true;
            endAnimation();
        }

        // Animation en 3 phases de temps total animDuration
        updateImage()

        setTimeout(() => {
            updateCarac()
        }, animDuration / 3)
            
        setTimeout(() => {
            updateCharts()
        }, 2 * animDuration / 3)

        setTimeout(() => {
            i++
            if (i < totalItems) {
                processData()
            }
        }, animDuration);
    }

    function endAnimation() {
        const fadeTime = 2000; // 2 secondes pour le fondu

        // Créer un overlay noir au dessus de l'animation
        const overlay = document.createElement('div');
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0)';
        overlay.style.opacity = '0';
        overlay.style.zIndex = '100';
        overlay.style.transition = `opacity ${fadeTime}ms ease`;
        overlay.style.pointerEvents = 'none';

        // Ajouter l'overlay au conteneur principal
        animContainer.style.position = 'relative';
        animContainer.appendChild(overlay);

        requestAnimationFrame(() => {
            overlay.style.opacity = '1';

            setTimeout(() => {
                // Stopper l'animation quand le fond noir est complètement opaque
                isAnimationRunning = false;

                // Attendre la fin de la dernière animation
                setTimeout(() => {
                    // Traiter le reste des données rapidement sans afficher
                    while (i < totalItems) {
                        const datasetIndex = labels[i]
                        chart.data.datasets[datasetIndex].data.push({x: c_train[i], y: 0})
                        const histoIndex = Math.floor((c_train[i] - min) / 2)
                        histo.data.datasets[labels[i]].data[histoIndex].y++
                        i++
                    }

                    // Mise à jour finale
                    chart.data.datasets[0].pointRadius = 4
                    chart.data.datasets[1].pointRadius = 4
                    chart.update()
                    histo.update()

                    // Afficher la dernière donnée
                    mathadata.run_python(`get_data(${totalItems - 1})`, (data) => {
                        window.mathadata.affichage(`${id}-data`, data)
                        setCarac(c_train[totalItems - 1].toFixed(2))
                    })

                    // Enlever le fond noir pour montrer l'état final
                    overlay.style.opacity = '0';
                    setTimeout(() => {
                        overlay.remove();
                    }, fadeTime);

                }, minAnimDuration);
            }, fadeTime);
        });
    }

    // Démarrer l'animation
    processData()
}
''')

### ----- CELLULES VALIDATION ----


### Pour les checks d'execution des cellules sans réponse attendue:
validation_execution_animation_histogramme = MathadataValidate(success="")
validation_execution_afficher_histogramme = MathadataValidate(success="")
validation_execution_reafficher_histogramme = MathadataValidate(success="")
validation_score_seuil_optim = MathadataValidate(success="")
validation_execution_caracteristique_custom = MathadataValidate(success="")
validation_score_zone_custom = MathadataValidate(success="")
validation_execution_afficher_customisation = MathadataValidate(success="")

def qcm_caracteristique():
    create_qcm({
    'question': "Qu'est ce qu'une caractéristique ?",
    'choices': [
        f"Un nombre qui donne une information sur {data('le')} permettant de {ac_fem('la', 'le')} classer",
        f"La classe {data('du', alt=True)} : {classe(0)} ou {classe(1)}",
        "Un algorithme",
        "Le dernier single d'Aya",
    ],
    'answer_index': 0,
    'multiline': True,
    })

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
