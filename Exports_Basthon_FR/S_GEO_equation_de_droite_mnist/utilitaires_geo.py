from IPython.display import display  # Pour afficher des DataFrames avec display(df)
import pandas as pd
import os
import sys

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    # If sklearn is not available, we can still run the code but without logistic regression
    LogisticRegression = None
    print("Warning: sklearn is not available. Logistic regression will not be used.")

# import mplcursors

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common

A_true = None
B_true = None


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
        'x1': 15,  # c_train[0][0],
        'y1': 20,  # c_train[0][1],
        'x2': 25,  # c_train[1][0],
        'y2': 30,  # c_train[1][1],
    }

    # On stocke en global pour la fonction validation
    global A_true, B_true
    A_true = [params['x1'], params['y1']]
    B_true = [params['x2'], params['y2']]

    run_js(f"setTimeout(() => window.mathadata.tracer_2_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

    df = pd.DataFrame()
    labels = ['Point A :', 'Point B :']
    df.index = labels
    # df.index.name = 'Point'
    df['$r$'] = [f'${r[0]}$', f'${r[1]}$']
    df['$x$'] = ['$?$', '$?$']
    df['$y$'] = ['$?$', '$?$']
    display(df)
    return


def tracer_200_points(nb=200):
    id = uuid.uuid4().hex
    display(HTML(f'''
        <canvas id="{id}-chart"></canvas>
    '''))

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques,
                                                      d_train=common.challenge.d_train[0:nb],
                                                      r_train=common.challenge.r_train[0:nb])
    params = {
        'points': c_train_par_population,
        'hideClasses': True,
    }

    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def update_custom():
    c_train_par_population = compute_c_train_by_class(
        fonction_caracteristique=common.challenge.deux_caracteristiques_custom)
    return json.dumps(c_train_par_population, cls=NpEncoder)


def estim_2d(a, b, c, k, above=None, below=None):
    if above is None:
        above = common.challenge.r_grande_caracteristique
    if below is None:
        below = common.challenge.r_petite_caracteristique
    return below if (b == 0 and k[0] > -c / a) or (b != 0 and a * k[0] + b * k[1] + c > 0) else above


def _classification(a, b, c, c_train, above=None, below=None):
    r_est_train = np.array([
        estim_2d(a, b, c, k, above, below)
        for k in c_train
    ])
    return r_est_train


def erreur_lineaire(a, b, c, c_train, above=None, below=None):
    r_est_train = _classification(a, b, c, c_train, above, below)
    erreurs = (r_est_train != common.challenge.r_train).astype(int)
    return 100 * np.mean(erreurs)


# regreession logistique toute simple
def compute_best_score_logistic(c_train, r_train, r_audessus=None, r_endessous=None):
    """
    c_train: np.array de shape (n, 2) avec n le nombre de points d'entraînement
    r_audessus: valeur de la réponse prédite pour les points au-dessus de la droite
    r_endessous: valeur de la réponse prédite pour les points en-dessous de la droite
    r_train: np.array de shape (n,) avec les réponses atendues des points d'entraînement

    Retourne un tuple (erreur, [a, b, c]) où:
    - erreur est le pourcentage d'erreur de classification
    - [a, b, c] sont les coefficients de la droite ax + by + c = 0
    """

    clf = LogisticRegression(max_iter=10000)
    clf.fit(c_train, r_train)
    predictions = clf.predict(c_train)
    accuracy = np.mean(predictions == r_train)
    # Récupération des paramètres de la droite
    # La régression logistique donne w0*x + w1*y + b = 0
    w = clf.coef_[0]  # [w0, w1]
    b = clf.intercept_[0]  # biais

    # Paramètres de la droite ax + by + c = 0
    a, b_coef, c = w[0], w[1], b

    return 1 - accuracy, [a, b_coef, c]


from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def compute_best_score_svm_hard(c_train, r_train):
    """
    MÉTHODE RECOMMANDÉE: SVM avec marge dure (C très grand)
    Trouve l'hyperplan optimal qui sépare les classes avec la marge maximale.
    """
    # C très grand = marge dure, trouve l'optimum global pour données séparables
    svm = SVC(kernel='linear', C=1e6)
    svm.fit(c_train, r_train)

    predictions = svm.predict(c_train)
    accuracy = np.mean(predictions == r_train)

    # Récupération des paramètres de la droite ax + by + c = 0
    w = svm.coef_[0]
    b = svm.intercept_[0]

    return 1 - accuracy, [w[0], w[1], b]


def compute_best_score_lda(c_train, r_train):
    """
    Alternative: Linear Discriminant Analysis
    Optimal pour données gaussiennes, très rapide.
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(c_train, r_train)

    predictions = lda.predict(c_train)
    accuracy = np.mean(predictions == r_train)

    # Récupération des paramètres (plus complexe pour LDA)
    # Approximation via les moyennes des classes
    class_means = np.array([c_train[r_train == i].mean(axis=0) for i in [0, 1]])
    w = class_means[1] - class_means[0]  # Direction de séparation
    midpoint = (class_means[0] + class_means[1]) / 2
    b = -np.dot(w, midpoint)

    return 1 - accuracy, [w[0], w[1], b]


def compute_best_score_perceptron(c_train, r_train):
    """
    Perceptron: trouve une solution optimale pour données linéairement séparables
    Très rapide, garantit convergence si séparable.
    """
    perceptron = Perceptron(max_iter=1000, tol=1e-3)
    perceptron.fit(c_train, r_train)

    predictions = perceptron.predict(c_train)
    accuracy = np.mean(predictions == r_train)

    # Récupération des paramètres
    w = perceptron.coef_[0]
    b = perceptron.intercept_[0]

    return 1 - accuracy, [w[0], w[1], b]


def compute_best_score_multiple_init(c_train, r_train):
    """
    Régression logistique avec plusieurs initialisations aléatoires
    Rapide et souvent suffisant.
    """

    best_error = float('inf')
    best_line = None

    # Tester plusieurs initialisations
    for random_state in [42, 123, 456, 789, 999]:
        clf = LogisticRegression(max_iter=2000, random_state=random_state,
                                 solver='lbfgs', C=1e6)
        clf.fit(c_train, r_train)

        predictions = clf.predict(c_train)
        accuracy = np.mean(predictions == r_train)
        error = 1 - accuracy

        if error < best_error:
            best_error = error
            w = clf.coef_[0]
            b = clf.intercept_[0]
            best_line = [w[0], w[1], b]

    return best_error, best_line


def compute_best_score_fast(c_train, r_train):
    """
    FONCTION PRINCIPALE: Combine les meilleures méthodes rapides
    """
    methods = [
        ("SVM marge dure", compute_best_score_svm_hard),
        ("Perceptron", compute_best_score_perceptron),
        ("LDA", compute_best_score_lda),
        ("LogReg multi-init", compute_best_score_multiple_init)
    ]

    best_error = float('inf')
    best_line = None
    best_method = None

    for method_name, method_func in methods:
        try:
            error, line = method_func(c_train, r_train)
            if error < best_error:
                best_error = error
                best_line = line
                best_method = method_name
        except Exception as e:
            print(f"Erreur avec {method_name}: {e}")
            continue

    return best_error, best_line


# SOLUTION ULTRA-RAPIDE pour notebook peu performant
def compute_best_score_ultra_fast(c_train, r_train):
    """
    Version ultra-rapide: juste SVM avec marge dure
    Dans 90% des cas, c'est suffisant pour trouver l'optimum.
    """
    svm = SVC(kernel='linear', C=1e6)
    svm.fit(c_train, r_train)

    predictions = svm.predict(c_train)
    accuracy = np.mean(predictions == r_train)

    w = svm.coef_[0]
    b = svm.intercept_[0]

    return 1 - accuracy, [w[0], w[1], b]


def get_best_score(method='utra-fast'):
    """
    Retourne le meilleur score de classification linéaire pour les données d'entraînement.
    """
    r_train = common.challenge.r_train
    r_audessus = common.challenge.r_grande_caracteristique
    r_endessous = common.challenge.r_petite_caracteristique
    d_train = common.challenge.d_train

    car = common.challenge.deux_caracteristiques_custom

    c_train = np.array([car(d) for d in d_train])

    methods = {'ultra-fast': compute_best_score_ultra_fast,
               'fast': compute_best_score_fast,
               'svm_hard': compute_best_score_svm_hard,
               'perceptron': compute_best_score_perceptron,
               'lda': compute_best_score_lda,
               'multiple_init': compute_best_score_multiple_init,
               'logistic': compute_best_score_logistic,
               }

    if method not in methods:
        raise ValueError(f"Méthode '{method}' non reconnue. Choisissez parmi {list(methods.keys())}.")
    compute_best_score = methods[method]

    erreur, params = compute_best_score(c_train, r_train)

    erreur = round(100 * erreur, 2)  # Convertir en pourcentage et arrondir à 2 décimales
    print(f"Meilleur score possible avec ces caractéristiques : {erreur}%")
    print(
        f"Paramètres de la meilleure droite : a={round(params[0], 2)}, b={round(params[1], 2)}, c={round(params[2], 2)}")
    return erreur, params


# JS

run_js('''
    if (localStorage.getItem('input-values')) {
        mathadata.run_python(`set_input_values('${localStorage.getItem("input-values")}')`)
    }

    const calculateCentroids = function(points) {
        return points.map((set, index) => {
          const sums = set.reduce((acc, [x, y]) => {
            acc.x += x;
            acc.y += y;
            return acc;
          }, {x: 0, y: 0});

          return {
            x: sums.x / set.length,
            y: sums.y / set.length,
          }
        })
    }

    const calculateMedianLine = function(centroidCoords) {
        const midX = (centroidCoords[0].x + centroidCoords[1].x) / 2;
        const midY = (centroidCoords[0].y + centroidCoords[1].y) / 2;
        const deltaX = centroidCoords[1].x - centroidCoords[0].x;
        const deltaY = centroidCoords[1].y - centroidCoords[0].y;

        // Calculate parameters a, b, c for the perpendicular bisector (médiatrice)
        // The equation of the line is ax + by + c = 0.
        // The vector (deltaX, deltaY) is the direction vector of the segment joining the two centroids.
        // The perpendicular bisector is normal to this segment.
        // So, the normal vector to the bisector is (deltaX, deltaY).
        // Thus, a = deltaX and b = deltaY for the bisector's equation.
        // The bisector passes through the midpoint (midX, midY).
        // So, a*midX + b*midY + c = 0
        // deltaX * midX + deltaY * midY + c = 0
        // c = -(deltaX * midX + deltaY * midY)

        const a = deltaX;
        const b = deltaY;
        const c = -(deltaX * midX + deltaY * midY);

        return {a, b, c}
    }
      
    window.mathadata.findIntersectionPoints = function(a, b, c, x_min, x_max, force_vertical) {
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
            if (force_vertical || (y >= y_min && y <= y_max)) {
                points.push({ x: x_min, y });
            }
            y = (-a * x_max - c) / b;
            if (force_vertical || (y >= y_min && y <= y_max)) {
                points.push({ x: x_max, y });
            }

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

    window.mathadata.getLineEquationStr = function(a, b, c) {
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

    // fonction générique pour les plots géométrie 2D
    window.mathadata.tracer_points = function(id, params) {
      if (typeof params === 'string') {
          params = JSON.parse(params);
      }

      const {points, droite, vecteurs, centroides, additionalPoints, hideClasses, hover, inputs, labels, displayValue, save, custom, compute_score, drag, force_origin, equation_hide } = params;
        // points: tableau des données en entrée sous forme de coordonnées (deux éléments, les points des 2 et les points des 7) [[[x,y],[x,y],...] , [[x,y],[x,y],...]]
        // droite: la droite à afficher (objet)
        // vecteurs: vecteurs à afficher pour le bouger (normal ou directeur)
        // centroides: bool: afficher les centroides
        // additionalPoints: tableau de points additionnels
        // hideClasses: bool: (défault: false) pour afficher la légende au dessus du graphe
        // hover: objet par défault, peux être appelé comme booleen pour un affichage hover selon son type
        // drag: bool: autorise à bouger les points
        // force_origin: bool: force le cadre à l'origine
        // custom: bool: pour le calcul du score, est-ce qu'on utilise la caractèristique custom ?
        // compute_score: bool: affichage du score
        // displayValue: bool: relatif à l'affichage des valeurs des inputs externes
        // equation_hide: bool: masque l'equation de la droite
      const computeScore = () => {
        const {a, b, c} = values;
        if (a === undefined || b === undefined || c === undefined) {
          return;
        }
        if (a === 0 && b === 0) {
          return;
        }
        
        const python = `compute_score_json(${a}, ${b}, ${c}, custom=${custom ? 'True' : 'False'})`
        mathadata.run_python(python, ({error}) => {
          if (error > 50) {
              error = 100 - error
          }
          error = Math.round(error * 100) / 100
          if (document.getElementById(`${id}-score`)) {
            document.getElementById(`${id}-score`).innerHTML = `${error}%`
          }
        })
      }
    
      // Values for all datasets, updated with inputs
      const values = {}
      const plugins = [];

      // Colors for the populations
      const colors = droite ? mathadata.classColorCodes.map(c => `rgb(${c})`) : mathadata.classColors;
      
      // Prepare the data for Chart.js
      
      // Points (dataset 0 and 1)
      const datasets = points.map((set, index) => {
          return {
              label: `${mathadata.data('', {plural: true, uppercase: true})} de ${hideClasses ? '?' : (index === 0 ? mathadata.challenge.strings.classes[0] : mathadata.challenge.strings.classes[1])}`,
              data: [],
              backgroundColor: colors[index],
              borderColor: colors[index],
              pointStyle: 'cross',
              pointRadius: 5,
              order: 1,
          }
      });
      
      let max, min;
      let droiteDatasetIndex;
      let centroid1DatasetIndex, centroid2DatasetIndex;

        // when caracteristique changes
      const updatePoints = (points, params) => {
        if (points) {
            const allData = points.flat(2);
            max = Math.ceil(Math.max(...allData) + 1);
            min = Math.floor(Math.min(...allData) - 1);
            if (force_origin) {
            min = Math.min(min, 0);
            }
            points.forEach((set, index) => {
            datasets[index].data = set.map(([x, y]) => ({ x, y }))
            })
        } else { // pour appeler depuis le callback dragData sans changer les coordonnées des points
            points = datasets.slice(0, 2).map(d => d.data).map(d => d.map(({x, y}) => [x, y]))
        }

        if (centroid1DatasetIndex) {
          const centroidCoords = calculateCentroids(points)
          centroidCoords.forEach(({x, y}, index) => {
            datasets[centroid1DatasetIndex + index].data = [{x, y}]
          })

          if (droite) {
            const coeffs = calculateMedianLine(centroidCoords)
            values.a = coeffs.a
            values.b = coeffs.b
            values.c = coeffs.c
            mathadata.run_python(`set_input_values('${JSON.stringify(values)}')`)
          }
        }

        if (droiteDatasetIndex) {
            const lineData = mathadata.findIntersectionPoints(values.a, values.b, values.c, min, max);  
            datasets[droiteDatasetIndex].data = lineData; 
            if (droite?.avec_zones) {
                const lineDataVertical = mathadata.findIntersectionPoints(values.a, values.b, values.c, min, max, true);  
                datasets[droiteDatasetIndex + 1].data = lineDataVertical; 
                datasets[droiteDatasetIndex + 2].data = lineDataVertical; 
            }
        }

        const chart = mathadata.charts[`${id}-chart`]
        if (chart) {
          chart.options.scales.x.min = min
          chart.options.scales.x.max = max
          chart.options.scales.y.min = min
          chart.options.scales.y.max = max
          chart.update(params?.animate === false ? 'none' : undefined)
        }

        if (compute_score) {
            computeScore()
        }
      }

      // initialisation
      updatePoints(points)

        // rend la fonction updatePoints accessible via l'objet chart
      plugins.push({
        beforeInit(chart, args, options) {
            chart.updatePoints = updatePoints;
        }
      })

      const start_ux = Math.round((max + min) / 2 / 10) * 10
      const start_uy = Math.round((min + 10) / 10) * 10
      let uDatasetIndex, nDatasetIndex, aDatasetIndex;
      if (vecteurs) {
        const {normal, directeur} = vecteurs;
        const vectorParams = []

        if (directeur) {
          // add vector u
          datasets.push({
              type: 'line',
              data: [],
              borderColor: 'red',
              borderWidth: 2,
              pointRadius: 0,
              pointHitRadius: 0,
              label: '\u20D7u',
          }); 
          vectorParams.push({
            datasetIndex: datasets.length - 1,
            color: 'red',
            label: '\u20D7u',
            id: 'directeur',
          })
          uDatasetIndex = datasets.length - 1;
        }

        if (normal) {
          // add vector n
          datasets.push({
              type: 'line',
              data: [],
              borderColor: 'blue',
              borderWidth: 2,
              pointRadius: 0,
              pointHitRadius: 0,
              label: '\u20D7n',
          });
          vectorParams.push({
            datasetIndex: datasets.length - 1,
            color: 'blue',
            label: '\u20D7n',
            id: 'normal',
          })
          nDatasetIndex = datasets.length - 1;
        }

        // add point A
        datasets.push({
            data: [],
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
        aDatasetIndex = datasets.length - 1;

        plugins.push({
          afterDatasetsDraw: function(chart) {
            const ctx = chart.ctx;
            const datasets = chart.data.datasets;

            vectorParams.forEach(({datasetIndex, color, label, id}, index) => {
              if (chart.isDatasetVisible(datasetIndex)) {
                const meta = chart.getDatasetMeta(datasetIndex);
                const data = meta.data;
                if (data?.length !== 2) {
                  return;
                }

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
                ctx.fillStyle = color;
                ctx.fill();

                ctx.save();
                ctx.font = '18px Arial';
                ctx.fillStyle = color;
                ctx.textAlign = 'left';
                // TODO
                ctx.fillText(`${label}(${values[id === 'directeur' ? 'ux' : 'nx']}, ${values[id === 'directeur' ? 'uy' : 'ny']})`, x2 + 10, y2);
                ctx.restore(); 
              }
            });

            if (chart.isDatasetVisible(aDatasetIndex)) {
                const meta = chart.getDatasetMeta(aDatasetIndex);
                const data = meta.data;
                if (data?.length !== 1) {
                  return;
                }
                
                const x = Math.round(chart.scales.x.getValueForPixel(data[0].x) * 100) / 100;
                const y = Math.round(chart.scales.y.getValueForPixel(data[0].y) * 100) / 100;
                ctx.save();
                ctx.font = '18px Arial';
                ctx.fillStyle = 'black';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(`A(${values['xa']}, ${values['ya']})`, data[0].x + 10, data[0].y);
                ctx.restore();
            }
          }
        })
      }
      
      let centroidCoords;
      if (centroides) {
        centroidCoords = calculateCentroids(points)

        centroidCoords.forEach(({x, y}, index) => {
          datasets.push({
              label: `Point moyen de la classe ${mathadata.challenge.strings.classes[index]}`,
              data: [{ x, y }],
              backgroundColor: colors[index],
              borderColor: 'black',
              pointStyle: 'circle',
              pointRadius: 6,
              order: 0,
          });
        })
        centroid1DatasetIndex = datasets.length - 2;
        centroid2DatasetIndex = datasets.length - 1;

        plugins.push({
            afterDraw: function(chart) {
                const ctx = chart.ctx;

                // Dessiner les labels des centroïdes 
                centroidCoords.forEach((centroid, index) => {
                    const datasetIndex = points.length + index;
                    if (chart.isDatasetVisible(datasetIndex)) {
                        const meta = chart.getDatasetMeta(datasetIndex);
                        const element = meta.data[0];
                        
                        ctx.save();
                        ctx.font = 'bold 12px Arial';
                        ctx.fillStyle = 'black';
                        ctx.textAlign = 'left';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(`point moyen de ${index === 0 ? mathadata.challenge.strings.classes[0] : mathadata.challenge.strings.classes[1]}`, element.x + 18, element.y - 3); // Adjusted X and Y offset
                        ctx.restore();
                    }
                });

                // Draw segments and distances if customHoverDetails exists
                if (hover?.type === 'distance' && chart.customHoverDetails) {
                    const { 
                        hoveredPixelX, hoveredPixelY, 
                        dataHoveredX, dataHoveredY, 
                    } = chart.customHoverDetails;

                    ctx.save();
                    centroidCoords.forEach((centroid, idx) => {
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
                        
                        ctx.font = 'bold 14px Arial';
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
            }
        })
      }

      if (droite) {
          let {mode, m, p, a, b, c, avec_zones} = droite;

          let label;
          let showLegend = true;
          if (equation_hide) {
            showLegend = false;
          }
            
          if (m !== undefined && p !== undefined) {
              label = `y = ${m}x ${p < 0 ? '-' : '+'} ${Math.abs(p)}`;
              a = m;
              b = -1;
              c = p;
          } else if (a !== undefined && b !== undefined && c !== undefined) {
              label = mathadata.getLineEquationStr(a, b, c);
          } else if (centroidCoords) {
            const params = calculateMedianLine(centroidCoords)
            a = params.a
            b = params.b
            c = params.c
            label = 'Médiatrice';
            showLegend = false;
          }

          values.m = m
          values.p = p
          values.a = a
          values.b = b
          values.c = c

          mathadata.run_python(`set_input_values('${JSON.stringify(values)}')`)

          const lineData = mathadata.findIntersectionPoints(a, b, c, min, max); 
            
          // droite
          datasets.push({
              label,
              type: 'line',
              data: lineData,
              pointRadius: 0,
              pointHitRadius: 0,
              borderColor: 'black',
              borderWidth: 1,
          });
          droiteDatasetIndex = datasets.length - 1;
          if (showLegend) {
            const lineDatasetIndex = datasets.length - 1;
            plugins.push({
              afterDatasetsDraw: function(chart) {
                const ctx = chart.ctx;
                const datasets = chart.data.datasets;
                if (chart.isDatasetVisible(lineDatasetIndex)) {
                    const meta = chart.getDatasetMeta(lineDatasetIndex);
                    const data = meta.data;
                    if (data?.length !== 2) {
                      return;
                    }
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
                    
                    let equation
                    if ((values.m !== undefined && values.p !== undefined) && mode !== 'cartesienne') {
                        equation = `y = ${values.m}x ${values.p < 0 ? '-' : '+'} ${Math.abs(values.p)}`
                    } else {
                        equation = mathadata.getLineEquationStr(values.a, values.b, values.c)
                    }
                    ctx.fillText(equation, 0, 0);
                    ctx.restore();
                }
              }
            })
          }

          // zones
          if (avec_zones) {
            const lineDataVertical = mathadata.findIntersectionPoints(a, b, c, min, max, true);  
            datasets.push({
                label: `Zone supérieure (${mathadata.challenge.strings.r_grande_caracteristique})`,
                type: 'line',
                data: lineDataVertical,
                pointsRadius: 0,
                pointHitRadius: 0,
                borderColor: 'transparent',
                fill: 'end',
                backgroundColor: `rgba(${mathadata.classColorCodes[mathadata.challenge.strings.classes.indexOf(mathadata.challenge.strings.r_grande_caracteristique)]}, 0.1)`,
            });

            datasets.push({
                label: `Zone inférieure (${mathadata.challenge.strings.r_petite_caracteristique})`,
                type: 'line',
                data: lineDataVertical,
                pointsRadius: 0,
                pointHitRadius: 0,
                borderColor: 'transparent',
                fill: 'origin',
                backgroundColor: `rgba(${mathadata.classColorCodes[mathadata.challenge.strings.classes.indexOf(mathadata.challenge.strings.r_petite_caracteristique)]}, 0.1)`,
            });
          }

          if (compute_score) {
            computeScore()
          }
        }

      if (additionalPoints) {
        const additionalPointsColors = {
            A: 'rgba(255, 2, 10, 0.5)',
            B: 'rgba(10, 255, 2, 0.5)',
            C: 'rgba(2, 150, 225, 0.5)'
        };

        const additionalPointsDatasetIndex = datasets.length;
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

        plugins.push({
            afterDatasetsDraw: function(chart) {
                const ctx = chart.ctx;
                const datasets = chart.data.datasets;
                
                // Dessiner les labels A/B/C
                Object.keys(additionalPoints).forEach((label, index) => {
                    const meta = chart.getDatasetMeta(additionalPointsDatasetIndex + index);
                    const point = meta.data[0];
                    
                    ctx.save();
                    ctx.font = 'bold 14px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'bottom';
                    ctx.fillStyle = 'black';
                    ctx.fillText(label, point.x, point.y - 10);
                    ctx.restore();
                });
            }
        })
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
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            usePointStyle: true,
                            boxWidth: 10,
                            filter: function(legendItem, data) {
                                return legendItem.datasetIndex <= 1;
                            }
                        }
                    }
                },
            },
            plugins,
        };

        if (hover) {
            const {type} = hover
            if (type === 'distance') {
                chartConfig.options.plugins.tooltip = {
                    enabled: false,
                    position: 'nearest',
                }
                chartConfig.options.onHover = (event, chartElements) => {
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
                };
            } else {
                chartConfig.options.plugins.tooltip = {
                    enabled: false,
                    position: 'nearest',
                    external: (context) => {
                        mathadata.dataTooltip(context, id)
                    }
                }
            }
        }

        if (drag) {
            datasets.forEach((dataset, index) => {
                if (index >= 2) {
                    dataset.dragData = false
                }
            })
            chartConfig.options.plugins.dragData = {
                dragX: true,  
                dragY: true,
                showTooltip: true,
                onDrag: function(e, datasetIndex, index, value) {
                    console.log('drag', e, datasetIndex, index, value)
                    updatePoints(null, {animate: false})
                }
            }
        }
        
        window.mathadata.create_chart(`${id}-chart`, chartConfig);

        if (inputs) {
          // On suppose que les inputs ont un élément html correspondant avec l'id {id}-input-{key}
          const inputElements = {}
          
          const update = () => {
            // Recupération des valeurs des inputs
            const newValues = {}
            Object.keys(inputElements).forEach((key) => {
              let val = inputElements[key].value
              if (typeof val === 'string') {
                  val = parseFloat(val)
              }
              if (!isNaN(val)) {
                newValues[key] = val
                if (displayValue) {
                  const label = document.getElementById(`${id}-label-${key}`)
                  if (label) {
                    label.textContent = `${key} = ${val}`
                  }
                }
              }
            })

            // Remplissage des valeurs avec les equivalences vecteur, droite, etc
            if (newValues.m !== undefined && newValues.p !== undefined) {
              newValues.a = newValues.m
              newValues.b = -1
              newValues.c = newValues.p
            }
            
            if (newValues.ux !== undefined && newValues.uy !== undefined) {
            console.log('pass', newValues)
              newValues.nx = -newValues.uy
              newValues.ny = newValues.ux
            }

            if (newValues.nx !== undefined && newValues.ny !== undefined) {
              newValues.ux = newValues.ny
              newValues.uy = -newValues.nx

              if (newValues.xa !== undefined && newValues.ya !== undefined) {
                newValues.a = newValues.nx
                newValues.b = newValues.ny
                newValues.c = -newValues.nx * newValues.xa - newValues.ny * newValues.ya
              }
            }
            
            Object.assign(values, newValues)
            const values_json = JSON.stringify(values)
            mathadata.run_python(`set_input_values('${values_json}')`)
            if (save) {
                localStorage.setItem(`input-values`, values_json)
            }

            // Update des datasets
            if (uDatasetIndex) {
              datasets[uDatasetIndex].data = [{ x: start_ux, y: start_uy }, { x: start_ux + values.ux, y: start_uy + values.uy }]
            }
            if (nDatasetIndex) {
              datasets[nDatasetIndex].data = [{ x: values.xa, y: values.ya }, { x: values.xa + values.nx, y: values.ya + values.ny }]
            }
            if (aDatasetIndex) {
              datasets[aDatasetIndex].data = [{ x: values.xa, y: values.ya }]
            }
            if (droiteDatasetIndex) {
              datasets[droiteDatasetIndex].data = mathadata.findIntersectionPoints(values.a, values.b, values.c, min, max); 
            }

            mathadata.charts[`${id}-chart`].update()

            // Update du score
            if (compute_score) {
              computeScore()
            }
          }

          // Initialisation point A
          if (inputs.xa !== undefined && inputs.ya !== undefined) {
              const init_a = Math.round((min + 10) / 10) * 10
              values.xa = init_a
              values.ya = init_a
          }
          
          if (save) {
            const savedValues = localStorage.getItem(`input-values`)
            if (savedValues) {
              Object.assign(values, JSON.parse(savedValues))
            }
          }

              

          Object.keys(inputs).forEach((key) => {
              inputElements[key] = document.getElementById(`${id}-input-${key}`)
              if (values[key] !== undefined) {
                inputElements[key].value = values[key]
              }
              inputElements[key].addEventListener('input', update)
          })

          // Initialisation
          update()
        }
    }

    window.mathadata.update_points = (id, params) => {
        const {points} = params;
        const chart = mathadata.charts[`${id}-chart`];
        chart.updatePoints(points)
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

input_values = {}


def set_input_values(values):
    global input_values
    input_values = json.loads(values)


def compute_score(a, b, c, custom=False, above=None, below=None):
    if custom:
        carac = common.challenge.deux_caracteristiques_custom
    else:
        carac = common.challenge.deux_caracteristiques

    c_train = compute_c_train(carac, common.challenge.d_train)
    error = erreur_lineaire(a, b, c, c_train, above, below)
    return error


def compute_score_json(a, b, c, custom=False):
    error = compute_score(a, b, c, custom)
    return json.dumps({'error': error})


def calculer_score_droite_geo(custom=False, validate=None, error_msg=None, banque=True, success_msg=None):
    global input_values

    if custom:
        deux_caracteristiques = common.challenge.deux_caracteristiques_custom
    else:
        deux_caracteristiques = common.challenge.deux_caracteristiques

    base_score = compute_score(input_values['a'], input_values['b'], input_values['c'], custom)

    if base_score <= 50:
        above = common.challenge.r_grande_caracteristique
        below = common.challenge.r_petite_caracteristique
    else:
        above = common.challenge.r_petite_caracteristique
        below = common.challenge.r_grande_caracteristique

    if validate is not None and base_score >= validate and base_score <= 100 - validate:
        if error_msg is None:
            print_error(
                f"Vous pourrez passer à la suite quand vous aurez un pourcentage d'erreur de moins de {validate}%.")
        else:
            print_error(error_msg)

    def algorithme(d):
        k = deux_caracteristiques(d)
        return estim_2d(input_values['a'], input_values['b'], input_values['c'], k, above, below)

    def cb(score):
        if validate is not None and score * 100 <= validate:
            if success_msg is None:
                pretty_print_success("Bravo, vous pouvez passer à la suite.")
            else:
                print(success_msg)
            pass_breakpoint()

    calculer_score(algorithme, method="2 moyennes",
                   parameters=f"a={input_values['a']}, b={input_values['b']}, c={input_values['c']}", cb=cb,
                   banque=banque)


### Validation

def check_coordinates(coords, errors):
    if not (isinstance(coords, tuple)):
        errors.append(
            "Les coordonnées doivent être écrites entre parenthèses séparés par une virgule. Exemple : (3, 5)")
        return False
    if len(coords) != 2:
        errors.append(
            "Les coordonnées doivent être composées de deux valeurs séparés par une virgule. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    if coords[0] is Ellipsis or coords[1] is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if not (isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float))):
        errors.append(
            "Les coordonnées doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    return True


def function_validation_2_points(errors, answers):
    A = answers['A']
    B = answers['B']
    if not check_coordinates(A, errors) or not check_coordinates(B, errors):
        return False

    deux_caracteristiques = common.challenge.deux_caracteristiques

    global A_true, B_true

    distA = np.sqrt((A[0] - A_true[0]) ** 2 + (A[1] - A_true[1]) ** 2)
    distB = np.sqrt((B[0] - B_true[0]) ** 2 + (B[1] - B_true[1]) ** 2)

    if distA > 3:
        distARev = np.sqrt((A[1] - A_true[0]) ** 2 + (A[0] - A_true[1]) ** 2)
        distAB = np.sqrt((A[0] - B_true[0]) ** 2 + (A[1] - B_true[1]) ** 2)
        if distAB < 3:
            errors.append(
                "Les coordonnées de A ne sont pas correctes. Tu as peut être donné les coordonnées du point B à la place ?")
        elif distARev < 3:
            errors.append(
                "Les coordonnées de A ne sont pas correctes. Attention, la première coordonnée est l'abscisse x et la deuxième l'ordonnée y.")
        else:
            errors.append("Les coordonnées de A ne sont pas correctes.")
    if distB > 3:
        distBRev = np.sqrt((B[1] - B_true[0]) ** 2 + (B[0] - B_true[1]) ** 2)
        distAB = np.sqrt((B[0] - A_true[0]) ** 2 + (B[1] - A_true[1]) ** 2)
        if distAB < 3:
            errors.append(
                "Les coordonnées de B ne sont pas correctes. Tu as peut être donné les coordonnées du point A à la place ?")
        elif distBRev < 3:
            errors.append(
                "Les coordonnées de B ne sont pas correctes. Attention, la première coordonnée est l'abscisse x et la deuxième l'ordonnée y.")
        else:
            errors.append("Les coordonnées de B ne sont pas correctes.")


def function_validation_score_droite(errors, answers):
    user_answer = answers['erreur_10']

    # Vérifications de base
    if not isinstance(user_answer, (int, float)):
        errors.append("Le pourcentage d'erreur doit être un nombre.")
        return False

    if user_answer < 0 or user_answer > 100:
        errors.append("Le pourcentage d'erreur doit être compris entre 0 et 100.")
        return False

    """Calcule dynamiquement le score attendu pour la droite à partir des données du défi"""
    # Récupération des données du défi
    dataset_10_points = common.challenge.dataset_10_points
    labels_10_points = common.challenge.labels_10_points
    droite_10_points = common.challenge.droite_10_points

    c_train = [common.challenge.deux_caracteristiques(d) for d in dataset_10_points]

    # Récupération des paramètres de la droite
    if 'm' in droite_10_points and 'p' in droite_10_points:
        m = droite_10_points['m']
        p = droite_10_points['p']
        # Conversion en forme ax + by + c = 0
        a = m
        b = -1
        c = p
    elif 'a' in droite_10_points and 'b' in droite_10_points and 'c' in droite_10_points:
        a = droite_10_points['a']
        b = droite_10_points['b']
        c = droite_10_points['c']
    else:
        raise ValueError("Paramètres de droite non supportés")

    # Calcul des prédictions
    nb_erreurs = 0
    nb_erreurs_par_classe = [0, 0]

    for i, (k, r_true) in enumerate(zip(c_train, labels_10_points)):
        # Classification 2D
        r_pred = estim_2d(a, b, c, k)

        if r_pred != r_true:
            nb_erreurs += 1
            if r_true == common.challenge.classes[0]:
                nb_erreurs_par_classe[0] += 1
            else:
                nb_erreurs_par_classe[1] += 1

    pourcentage_erreur = (nb_erreurs / len(dataset_10_points)) * 100

    # Vérification si l'utilisateur a donné le nombre d'erreurs au lieu du pourcentage
    if user_answer == nb_erreurs:
        errors.append(
            f"Ce n'est pas la bonne valeur. Vous avez donné le nombre d'erreurs ({nb_erreurs}) et non le pourcentage d'erreur.")
        return False

    # Vérification de la réponse correcte
    if user_answer == pourcentage_erreur:
        if nb_erreurs == 0:
            pretty_print_success(
                "Bravo, c'est la bonne réponse. Il n'y a aucune erreur de classification sur ce schéma.")
        else:
            # Détails sur les erreurs pour le message
            pretty_print_success(
                f"Bravo, c'est la bonne réponse. Il y a {nb_erreurs_par_classe[0] == 1 and 'un' or nb_erreurs_par_classe[0]} {common.challenge.classes[0]} {'au dessus' if common.challenge.classes[0] == common.challenge.r_petite_caracteristique else 'en dessous'} de la droite et {nb_erreurs_par_classe[1] == 1 and 'un' or nb_erreurs_par_classe[1]} {common.challenge.classes[1]} {'au dessus' if common.challenge.classes[1] == common.challenge.r_petite_caracteristique else 'en dessous'}, donc {nb_erreurs} erreurs soit {pourcentage_erreur}%.")
        return True
    else:
        errors.append(
            f"Ce n'est pas la bonne réponse. Comptez le nombre d'erreurs c'est à dire le nombre de points du mauvais côté de la droite puis calculez le pourcentage d'erreur.")
        return False


def function_validation_score_droite_20(errors, answers):
    user_answer = answers['erreur_20']

    # Vérifications de base
    if not isinstance(user_answer, (int, float)):
        errors.append("Le pourcentage d'erreur doit être un nombre.")
        return False

    if user_answer < 0 or user_answer > 100:
        errors.append("Le pourcentage d'erreur doit être compris entre 0 et 100.")
        return False

    """Calcule dynamiquement le score attendu pour la droite à partir des données du défi"""
    # Récupération des données du défi
    dataset_20_points = common.challenge.dataset_20_points
    labels_20_points = common.challenge.labels_20_points
    droite_20_points = common.challenge.droite_20_points

    c_train = [common.challenge.deux_caracteristiques(d) for d in dataset_20_points]

    # Récupération des paramètres de la droite
    if 'm' in droite_20_points and 'p' in droite_20_points:
        m = droite_20_points['m']
        p = droite_20_points['p']
        # Conversion en forme ax + by + c = 0
        a = m
        b = -1
        c = p
    elif 'a' in droite_20_points and 'b' in droite_20_points and 'c' in droite_20_points:
        a = droite_20_points['a']
        b = droite_20_points['b']
        c = droite_20_points['c']
    else:
        raise ValueError("Paramètres de droite non supportés")

    # Calcul des prédictions
    nb_erreurs = 0
    nb_erreurs_par_classe = [0, 0]

    for i, (k, r_true) in enumerate(zip(c_train, labels_20_points)):
        # Classification 2D
        r_pred = estim_2d(a, b, c, k)

        if r_pred != r_true:
            nb_erreurs += 1
            if r_true == common.challenge.classes[0]:
                nb_erreurs_par_classe[0] += 1
            else:
                nb_erreurs_par_classe[1] += 1

    pourcentage_erreur = (nb_erreurs / len(dataset_20_points)) * 100

    # Vérification si l'utilisateur a donné le nombre d'erreurs au lieu du pourcentage
    if user_answer == nb_erreurs:
        errors.append(
            f"Ce n'est pas la bonne valeur. Vous avez donné le nombre d'erreurs ({nb_erreurs}) et non le pourcentage d'erreur.")
        return False

    # Vérification de la réponse correcte
    if user_answer == pourcentage_erreur:
        if nb_erreurs == 0:
            pretty_print_success(
                "Bravo, c'est la bonne réponse. Il n'y a aucune erreur de classification sur ce schéma.")
        else:
            # Détails sur les erreurs pour le message
            pretty_print_success(
                f"Bravo, c'est la bonne réponse. Il y a {nb_erreurs_par_classe[0] == 1 and 'un' or nb_erreurs_par_classe[0]} {common.challenge.classes[0]} {'au dessus' if common.challenge.classes[0] == common.challenge.r_petite_caracteristique else 'en dessous'} de la droite et {nb_erreurs_par_classe[1] == 1 and 'un' or nb_erreurs_par_classe[1]} {common.challenge.classes[1]} {'au dessus' if common.challenge.classes[1] == common.challenge.r_petite_caracteristique else 'en dessous'}, donc {nb_erreurs} erreurs soit {pourcentage_erreur}%.")
        return True
    else:
        errors.append(
            f"Ce n'est pas la bonne réponse. Comptez le nombre d'erreurs c'est à dire le nombre de points du mauvais côté de la droite puis calculez le pourcentage d'erreur.")
        return False


validation_execution_2_points = MathadataValidate(success="")
validation_question_2_points = MathadataValidateVariables({
    'A': None,
    'B': None,
}, function_validation=function_validation_2_points)
validation_execution_200_points = MathadataValidate(success="")
validation_question_couleur = MathadataValidateVariables({
    'classe_points_bleus': {
        'value': common.challenge.classes[0],
        'errors': [
            {
                'value': {
                    'in': common.challenge.classes,
                },
                'else': f"classe_points_bleus n'a pas la bonne valeur. Vous devez répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
            }
        ]
    },
    'classe_points_oranges': {
        'value': common.challenge.classes[1],
        'errors': [
            {
                'value': {
                    'in': common.challenge.classes,
                },
                'else': f"classe_points_oranges n'a pas la bonne valeur. Vous devez répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
            }
        ]
    }
})

validation_execution_10_points = MathadataValidate(success="")
validation_execution_20_points = MathadataValidate(success="")
validation_question_score_droite = MathadataValidateVariables({
    'erreur_10': None
},
    function_validation=function_validation_score_droite,
    success="")
validation_question_score_droite_20 = MathadataValidateVariables({
    'erreur_20': None
},
    function_validation=function_validation_score_droite_20,
    success="")
validation_execution_tracer_points_droite = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_execution_point_droite = MathadataValidate(success="")
validation_score_droite_custom = MathadataValidate(
    success="Bravo, vous pouvez continuer à essayer d'améliorer votre score. Il est possible de faire seulement 3% d'erreur.")
validation_execution_scatter_caracteristiques_ripou = MathadataValidate(success="")
validation_execution_afficher_customisation = MathadataValidate(success="")
