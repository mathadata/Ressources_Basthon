import base64
import mimetypes
import os
import sys
import warnings

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

    run_js(
        f"mathadata.add_observer('{id}', () => window.mathadata.tracer_2_points('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'<canvas id="{id}"></canvas>'))

    df = pd.DataFrame()
    labels = ['Point A :', 'Point B :']
    df.index = labels
    # df.index.name = 'Point'
    df['$r$'] = [f'${r[0]}$', f'${r[1]}$']
    df['$x$'] = ['$?$', '$?$']
    df['$y$'] = ['$?$', '$?$']
    display(df)
    return


### PLACER 2 POINTS ###

def placer_caracteristiques(html_title="Calcul des caractéristiques", left_width=None, images=None,
                            expected_points_a=None, expected_points_b=None, preplace_points_a=False):
    """
    Render the JSXGraph iframe in a Jupyter notebook cell and show a 2x2 image grid
    to the left. Accepts expected_points_A and expected_points_B as dicts:
      e.g. expected_points_A = {'A':[120,90], 'B':[135,105]}
    When a placed point matches (within match_tolerance) one of those, the point
    is colored with the group's matched color and the label is shown.
    """

    # --- defaults (now dictionaries) ---
    if expected_points_a is None:
        expected_points_a = {"A": [2, 3], "B": [10, 12]}
    if expected_points_b is None:
        expected_points_b = {"C": [5, 5], "D": [7, 8]}

    matched_color_a = "#4C6EF5"
    matched_color_b = "#F6C85F"
    default_color = "#8c1fb4"

    width = 830
    panel_h = 370

    left_ratio = 0.35

    match_tolerance = 0.35
    click_delete_tolerance = 0.45

    image_mode = "contain"
    # normalize images list to length 4
    if images is None:
        images = [None, None, None, None]
    else:
        images = list(images)[:4] + [None] * max(0, 4 - len(images))

    instance_id = uuid.uuid4().hex

    def to_data_uri_if_local(src):
        if src is None:
            return None
        if isinstance(src, str) and src.startswith("data:"):
            return src
        if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
            return src
        if isinstance(src, str) and os.path.exists(src) and os.path.isfile(src):
            mime, _ = mimetypes.guess_type(src)
            mime = mime or "application/octet-stream"
            with open(src, "rb") as f:
                b = f.read()
            return f"data:{mime};base64," + base64.b64encode(b).decode("ascii")
        return src

    images_prepared = [to_data_uri_if_local(i) for i in images]
    images_json = json.dumps(images_prepared)

    # pass the dicts through to JS as objects
    expected_json_a = json.dumps(expected_points_a, sort_keys=True)
    expected_json_b = json.dumps(expected_points_b, sort_keys=True)

    checkpoint_payload = {
        "type": "placer_caracteristiques",
        "title": html_title,
        "expected_points_a": expected_points_a,
        "expected_points_b": expected_points_b,
    }
    checkpoint_payload_json = json.dumps(checkpoint_payload, sort_keys=True, ensure_ascii=False)

    if left_width is None:
        left_w_px = int(round(left_ratio * width))
    else:
        left_w_px = int(left_width)

    iframe_height = panel_h + 64

    # Choose CSS snippet for the selected image_mode
    if image_mode == "fit-height":
        img_css_rules = "width:auto; height:100%; object-fit:unset; object-position:center; display:block; margin:0 auto;"
    else:
        img_css_rules = "width:100%; height:100%; object-fit:contain; object-position:center; display:block;"

    page = r"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>IFRAME_TITLE_PLACEHOLDER</title>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <link rel="stylesheet" type="text/css" href="https://jsxgraph.org/distrib/jsxgraph.css" />
    <style>
      html, body { margin:0; padding:0; overflow:hidden; font-family: sans-serif; }
      .title { width:100%; text-align:center; font-size:16px; font-weight:700; padding:6px 0; }

      .container { display:flex; gap:8px; padding:8px; height: PANEL_H_PX; box-sizing:border-box; }
      .left { width: LEFT_W_PX; min-width:120px; display:flex; align-items:stretch; justify-content:center; }
      .right { flex:1; display:flex; align-items:stretch; justify-content:center; }

      .imggrid {
        display:grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: 1fr 1fr;
        gap:8px;
        width:100%;
        height:100%;
        box-sizing:border-box;
      }

      .img-cell {
        overflow:hidden;
        border:0px solid #e6e6e6;
        border-radius:6px;
        background:#fff;
        display:flex;
        align-items:center;
        justify-content:center;
      }

      /* injected rule for chosen mode */
      .img-cell img {
        INJECTED_IMG_CSS
      }

      .img-placeholder-text {
        padding:8px;
        color:#444;
        font-size:13px;
        text-align:center;
      }

      #box, .jxgbox { width:100%; height:100%; min-height:0; border:1px solid #e6e6e6; border-radius:6px; box-sizing:border-box; }

      @media (max-width: 820px) {
        .container { flex-direction:column; height:auto; }
        .left, .right { width:100%; min-height:220px; }
      }
    </style>
    </head>
    <body>
      <div class="title">HUMAN_TITLE_PLACEHOLDER</div>

      <div class="container">
        <div class="left" aria-label="images column">
          <div class="imggrid" role="group" aria-label="2 by 2 image placeholders">
            <div class="img-cell" id="img1"></div>
            <div class="img-cell" id="img2"></div>
            <div class="img-cell" id="img3"></div>
            <div class="img-cell" id="img4"></div>
          </div>
        </div>

        <div class="right" aria-label="interactive column">
          <div id="box"></div>
        </div>
      </div>

      <script src="https://cdnjs.cloudflare.com/ajax/libs/jsxgraph/1.4.0/jsxgraphcore.js"></script>
      <script>
        (function(){
          // expectedPoints are JS objects: { label: [x,y], ... }
          var expectedPointsA = EXPECTED_A_PLACEHOLDER;
          var expectedPointsB = EXPECTED_B_PLACEHOLDER;
          var preplaceGroupA = PREPLACE_A_PLACEHOLDER;
          var matchTol = MATCH_TOL_PLACEHOLDER;
          var clickDeleteTol = CLICK_DELETE_TOL_PLACEHOLDER;
          var defaultColor = "DEFAULT_COLOR_PLACEHOLDER";
          var matchedColorA = "MATCHED_COLOR_A_PLACEHOLDER";
          var matchedColorB = "MATCHED_COLOR_B_PLACEHOLDER";

          var images = IMAGES_PLACEHOLDER || [null, null, null, null];
          // Filtrer les images non-null pour obtenir les images réelles
          var actualImages = images.filter(function (img) {
              return img !== null && img !== undefined;
          });
          var imageCount = actualImages.length;
    
          function fillImageCell(id, src, idx) {
              var el = document.getElementById(id);
              if (!el) return; // Si l'élément n'existe pas, on ignore
    
              el.innerHTML = "";
              if (!src) {
                  // Ne plus afficher de placeholder, masquer la cellule
                  el.style.display = 'none';
                  return;
              }
              var img = document.createElement('img');
              img.src = src;
              img.alt = 'image ' + (idx + 1);
              el.appendChild(img);
              el.style.display = 'block';
          }
    
          // Gérer l'affichage selon le nombre d'images
          if (imageCount === 2) {
              // Cas de 2 images : centrer l'affichage
              // Masquer les cellules 1 et 4, utiliser les cellules 2 et 3 pour centrer
              fillImageCell('img1', null, 0); // Masquer
              fillImageCell('img2', actualImages[0], 1); // Première image
              fillImageCell('img3', actualImages[1], 2); // Deuxième image  
              fillImageCell('img4', null, 3); // Masquer
          } else if (imageCount === 4) {
              // Cas de 4 images : affichage normal
              for (var i = 0; i < 4; i++) {
                  fillImageCell('img' + (i + 1), images[i], i);
              }
          }

          // -------- matching helpers that use labeled dicts ----------
          // check coords against expectedPointsA/B; returns {group:'A'|'B', label: 'label'} or null
          function whichMatchCoords(x, y) {
            try {
              // A has precedence
              for (var label in expectedPointsA) {
                if (!Object.prototype.hasOwnProperty.call(expectedPointsA, label)) continue;
                var ex = expectedPointsA[label][0], ey = expectedPointsA[label][1];
                if (Math.abs(x - ex) <= matchTol && Math.abs(y - ey) <= matchTol) {
                  return {group: 'A', label: label};
                }
              }
              for (var label2 in expectedPointsB) {
                if (!Object.prototype.hasOwnProperty.call(expectedPointsB, label2)) continue;
                var bx = expectedPointsB[label2][0], by = expectedPointsB[label2][1];
                if (Math.abs(x - bx) <= matchTol && Math.abs(y - by) <= matchTol) {
                  return {group: 'B', label: label2};
                }
              }
            } catch (e) {
              console.error('whichMatchCoords error', e);
            }
            return null;
          }

          // Check if all expected points are visible
          function checkAllExpectedPointsVisible() {
            try {
              var expectedLabelsA = Object.keys(expectedPointsA);
              var expectedLabelsB = Object.keys(expectedPointsB);
              var allExpectedLabels = expectedLabelsA.concat(expectedLabelsB);

              var matchedLabels = {};
              userPoints.forEach(function(pt) {
                if (board.objectsList.indexOf(pt) === -1) return;
                var x = pt.X(), y = pt.Y();
                var res = whichMatchCoords(x, y);
                if (res) {
                  matchedLabels[res.label] = true;
                }
              });

              var allVisible = allExpectedLabels.every(function(label) {
                return matchedLabels[label] === true;
              });

              console.log('Expected labels:', allExpectedLabels);
              console.log('Matched labels:', Object.keys(matchedLabels));
              console.log('All visible:', allVisible);

              if (allVisible) {
                console.log('All expected points are correctly placed!');
                // Send message to parent window to update Python
                window.parent.postMessage({type:'all_points_matched', status: true}, '*');
              } else {
                // Send message to reset Python variable when not all points are matched
                window.parent.postMessage({type:'all_points_matched', status: false}, '*');
              }
            } catch (e) {
              console.error('checkAllExpectedPointsVisible error', e);
            }
          }

          // update a point's color and label according to which expected point it matches
          function updatePointColorAndLabel(pt) {
            var x = pt.X(), y = pt.Y();
            var res = whichMatchCoords(x, y);
            if (res && res.group === 'A') {
              pt.setAttribute({
                fillColor: matchedColorA,
                strokeColor: matchedColorA,
                name: res.label,
                withLabel: true,
                fixed: true,
                frozen: true,
                highlight: false,
                showInfobox: false
              });
              // Marquer comme point validé (non-interactif)
              pt.isMatched = true;
            } else if (res && res.group === 'B') {
              pt.setAttribute({
                fillColor: matchedColorB,
                strokeColor: matchedColorB,
                name: res.label,
                withLabel: true,
                fixed: true,
                frozen: true,
                highlight: false,
                showInfobox: false
              });
              // Marquer comme point validé (non-interactif)
              pt.isMatched = true;
            } else {
              pt.setAttribute({
                fillColor: defaultColor,
                strokeColor: defaultColor,
                name: 'Mauvais point',
                withLabel: true,
                fixed: false,
                frozen: false
              });
              pt.isMatched = false;
              // Schedule auto-deletion after 2 seconds for bad points
              setTimeout(function() {
                try {
                  var idx = userPoints.indexOf(pt);
                  if (idx !== -1) {
                    if (board.objectsList.indexOf(pt) !== -1) {
                      board.removeObject(pt);
                    }
                    userPoints.splice(idx, 1);
                    sendPointsToParent();
                    checkAllExpectedPointsVisible();
                  }
                } catch (e) { console.error('auto-delete error', e); }
              }, 2000);
            }
            checkAllExpectedPointsVisible();
          }

          // initialize board with requested bounding box
          var xmin = 50, xmax = 165, ymin = 50, ymax = 160;
          var xaxisdisplacement = 10;
          
          var board = JXG.JSXGraph.initBoard('box', {
            boundingbox: [xmin, ymax, xmax, ymin],
            axis: true,
            showNavigation: false,
            keepaspectratio: true,
            showCopyright: false,
          });

          var MAJOR = 10;
          var MINOR_COUNT = 1;
          var GRID_STEP = MAJOR / (MINOR_COUNT + 1);

          var xAxis = board.create('axis', [[xmin+xaxisdisplacement, ymin+5], [xmax, ymin+5]], {
            name: '', withLabel: false,
            ticks: {
              insertTicks: false,
              ticksDistance: MAJOR,
              minorTicks: MINOR_COUNT,
              minorHeight: -1,
              majorHeight: -1,
              drawZero: false,
              drawLabels: true,
              label: { offset: [-9, -8], anchorX: 'top' }
            }
          });
          if (xAxis && xAxis.ticks && xAxis.ticks.labels && xAxis.ticks.labels.length > 0) {
            xAxis.ticks.labels[0].setText('');
          }

          var yAxis = board.create('axis', [[xmin+xaxisdisplacement, ymin+5], [xmin+xaxisdisplacement, ymax]], {
            name: '', withLabel: false,
            ticks: {
              insertTicks: false,
              ticksDistance: MAJOR,
              minorTicks: MINOR_COUNT,
              minorHeight: -1,
              majorHeight: -1,
              drawZero: false,
              drawLabels: true,
              label: { offset: [-2, 0], anchorX: 'right' }
            }
          });

	          // store user-created points (and optionally preplaced expected points)
	          var userPoints = [];

	          // Optionnel : préplacer les points du groupe A (ex: A et B en bleu)
	          if (preplaceGroupA) {
	            try {
	              for (var label in expectedPointsA) {
	                if (!Object.prototype.hasOwnProperty.call(expectedPointsA, label)) continue;
	                var coords = expectedPointsA[label];
	                var p0 = board.create('point', coords, {
	                  withLabel: false, size: 6, name: '',
	                  snapToGrid: true, snapSizeX: GRID_STEP, snapSizeY: GRID_STEP
	                });
	                if (typeof p0.snapToGrid === 'function') p0.snapToGrid(true);
	                userPoints.push(p0);
	                updatePointColorAndLabel(p0);
	              }
	              sendPointsToParent();
	            } catch (e) {
	              console.error('Error preplacing group A points', e);
	            }
	          }

          function sendPointsToParent() {
            try {
              var pts = [];
              userPoints.forEach(function(pt) {
                if (board.objectsList.indexOf(pt) !== -1) {
                  pts.push([ +pt.X().toFixed(6), +pt.Y().toFixed(6) ]);
                }
              });
              window.parent.postMessage({type:'jxg_points', points: pts}, '*');
            } catch (err) {
              console.error('sendPointsToParent error', err);
            }
          }

          // distance + neighbor finder
          function dist(a, b) {
            var dx = a[0] - b[0];
            var dy = a[1] - b[1];
            return Math.sqrt(dx*dx + dy*dy);
          }

          function findNearbyUserPointIndex(coords) {
            for (var i = 0; i < userPoints.length; i++) {
              var pt = userPoints[i];
              if (board.objectsList.indexOf(pt) === -1) continue;
              var pcoords = [pt.X(), pt.Y()];
              if (dist(coords, pcoords) <= clickDeleteTol) return i;
            }
            return -1;
          }

          // On click: delete nearby or create a new snapped point
          board.on('down', function(evt) {
            try {
              var raw = board.getUsrCoordsOfMouse(evt);
              var nearbyIdx = findNearbyUserPointIndex(raw);
              if (nearbyIdx !== -1) {
                var p = userPoints[nearbyIdx];
                // Si le point est validé (bon point), ne rien faire (pas de suppression, pas de copie)
                if (p && p.isMatched) {
                  return;
                }
                // Sinon, suppression classique du point cliqué
                if (board.objectsList.indexOf(p) !== -1) board.removeObject(p);
                userPoints.splice(nearbyIdx, 1);
                sendPointsToParent();
                return;
              }

              var snapped = snapCoords = (function(coords) {
                var x = Math.round(coords[0]);
                var y = Math.round(coords[1]);
                var bb = board.getBoundingBox();
                var xmin = bb[0], ymax = bb[1], xmax = bb[2], ymin = bb[3];
                if (x < xmin) x = xmin;
                if (x > xmax) x = xmax;
                if (y < ymin) y = ymin;
                if (y > ymax) y = ymax;
                return [x, y];
              })(raw);

              var p = board.create('point', snapped, {
                withLabel: false, size: 6, name: '',
                snapToGrid: true, snapSizeX: GRID_STEP, snapSizeY: GRID_STEP
              });
              if (typeof p.snapToGrid === 'function') p.snapToGrid(true);

              userPoints.push(p);
              updatePointColorAndLabel(p);

              // drag: enforce snapping & colour/label update
              p.on('drag', function() {
                try {
                  // Empêcher tout déplacement si le point est validé
                  if (this.isMatched) { return; }
                  var s = snapCoords([this.X(), this.Y()]);
                  this.moveTo(JXG.COORDS_BY_USER, s);
                  updatePointColorAndLabel(this);
                  sendPointsToParent();
                } catch (e) { console.error('drag error', e); }
              });

              // up: final snap & update
              p.on('up', function() {
                try {
                  if (this.isMatched) { return; }
                  var s = snapCoords([this.X(), this.Y()]);
                  this.moveTo(JXG.COORDS_BY_USER, s);
                  updatePointColorAndLabel(this);
                  sendPointsToParent();
                } catch (e) { console.error('up error', e); }
              });

              sendPointsToParent();
            } catch (err) { console.error('Error handling down event', err); }
          });

          // during drag send updates (not too chatty)
          board.on('move', function(evt) {
            if (board.hasMouseDown) sendPointsToParent();
          });

          // clear API
          window.__jsx_clear_points = function() {
            try {
              for (var i = 0; i < userPoints.length; i++) {
                var el = userPoints[i];
                if (board.objectsList.indexOf(el) !== -1) board.removeObject(el);
              }
              userPoints = [];
              sendPointsToParent();
            } catch (err) { console.error('clear error', err); }
          };

          document.getElementById('box').tabIndex = 0;
        })();
      </script>
    </body>
    </html>
    """

    # replacements
    page = page.replace("PANEL_H_PX", f"{panel_h}px")
    page = page.replace("LEFT_W_PX", f"{left_w_px}px")
    page = page.replace("HUMAN_TITLE_PLACEHOLDER", html_title)
    page = page.replace("IFRAME_TITLE_PLACEHOLDER", html_title)
    page = page.replace("EXPECTED_A_PLACEHOLDER", expected_json_a)
    page = page.replace("EXPECTED_B_PLACEHOLDER", expected_json_b)
    page = page.replace("PREPLACE_A_PLACEHOLDER", "true" if preplace_points_a else "false")
    page = page.replace("MATCH_TOL_PLACEHOLDER", str(match_tolerance))
    page = page.replace("CLICK_DELETE_TOL_PLACEHOLDER", str(click_delete_tolerance))
    page = page.replace("DEFAULT_COLOR_PLACEHOLDER", default_color)
    page = page.replace("MATCHED_COLOR_A_PLACEHOLDER", matched_color_a)
    page = page.replace("MATCHED_COLOR_B_PLACEHOLDER", matched_color_b)
    page = page.replace("IMAGES_PLACEHOLDER", images_json)
    page = page.replace("INJECTED_IMG_CSS", img_css_rules)

    page_bytes = page.encode('utf-8')
    page_b64 = base64.b64encode(page_bytes).decode('ascii')
    data_uri = f"data:text/html;base64,{page_b64}"
    iframe_html = f"""
    <div id="{instance_id}-wrapper" style="display:flex; flex-direction:column; align-items:center; gap:8px;">
      <div id="{instance_id}-status" style="text-align:center; font-weight:bold; min-height:1.5rem;"></div>
      <iframe id="{instance_id}-jsxframe" src="{data_uri}" style="width:{width}px; height:{iframe_height}px; border:none;" sandbox="allow-scripts allow-same-origin"></iframe>
    </div>
    """

    listener_script = f"""
    (function(){{
      const instanceId = {json.dumps(instance_id)};
      const statusEl = document.getElementById(instanceId + '-status');

      const checkpointPayload = {checkpoint_payload_json};
      const checkpointId = (window.mathadata && window.mathadata.checkpoints)
        ? ('placer_points_' + window.mathadata.checkpoints.hash(checkpointPayload))
        : null;

      function setStatus(text, color, italic) {{
        if (!statusEl) return;
        statusEl.textContent = text || '';
        statusEl.style.color = color || '';
        statusEl.style.fontStyle = italic ? 'italic' : 'normal';
      }}

      function tryRunPython(code) {{
        try {{
          if (window.mathadata && typeof window.mathadata.run_python === 'function') {{
            window.mathadata.run_python(code);
            return;
          }}
          if (window.Jupyter && Jupyter.notebook && Jupyter.notebook.kernel) {{
            Jupyter.notebook.kernel.execute(code);
          }}
        }} catch (e) {{
          console.error('tryRunPython error', e);
        }}
      }}

      function tryPassBreakpoint() {{
        try {{
          if (window.mathadata && typeof window.mathadata.pass_breakpoint === 'function') {{
            window.mathadata.pass_breakpoint();
          }}
        }} catch (e) {{
          console.error('tryPassBreakpoint error', e);
        }}
      }}

      let alreadyPassed = false;

      // Si l'exercice a déjà été validé, afficher un message et passer automatiquement.
      if (checkpointId && window.mathadata && window.mathadata.checkpoints && window.mathadata.checkpoints.check(checkpointId)) {{
        setStatus('✓ Tu as déjà réussi cet exercice précédemment. Tu peux continuer.', 'green', true);
        alreadyPassed = true;
        tryRunPython("set_exercice_droite_carac_ok()");
        tryPassBreakpoint();
      }}

      var hidden = document.getElementById(instanceId + '_jsx_points_hidden');
      if (!hidden) {{
        hidden = document.createElement('div');
        hidden.id = instanceId + '_jsx_points_hidden';
        hidden.style.display = 'none';
        document.body.appendChild(hidden);
      }}

      window.addEventListener('message', function(e) {{
        try {{
          var data = e.data;
          if (data && data.type === 'jxg_points') {{
            hidden.textContent = JSON.stringify(data.points || []);
            if (window.Jupyter && Jupyter.notebook && Jupyter.notebook.kernel) {{
              var code = "jsx_points = " + JSON.stringify(data.points || []) + "\\n";
              Jupyter.notebook.kernel.execute(code);
            }} else {{
              console.log('jsx_points received (kernel injection not available) — stored in hidden DOM element.');
            }}
          }} else if (data && data.type === 'all_points_matched') {{
            if (data.status === true) {{
              if (!alreadyPassed) {{
                setStatus('Bravo ! Tu as bien placé les points. Tu peux continuer.', 'green', false);
                if (checkpointId && window.mathadata && window.mathadata.checkpoints) {{
                  window.mathadata.checkpoints.save(checkpointId);
                }}
                alreadyPassed = true;
                tryPassBreakpoint();
              }}
              tryRunPython("set_exercice_droite_carac_ok()");
            }} else if (data.status === false) {{
              tryRunPython("reset_exercice_droite_carac()");
            }}
          }}
        }} catch (err) {{ console.error('Listener error', err); }}
      }}, false);
    }})();
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consider using IPython.display.IFrame instead")
        display(HTML(iframe_html))
        run_js(listener_script)


moyenne_carac = False


def reset_exercice_droite_carac():
    global moyenne_carac
    moyenne_carac = False


def mauvaises_caracteristiques():
    reset_exercice_droite_carac()
    similar_image_caracteristics = [
        files_url + "image_2_5x3_180_similaire.png",
        files_url + "image_2_5x3_210-180_similaire.png",
        files_url + "image_7_5x3_180_similaire.png",
        files_url + "image_7_5x3_210-150_similaire.png"
    ]

    placer_caracteristiques(
        html_title="Des mauvaises caractéristiques",
        expected_points_a={'A': [120, 90], 'B': [135, 105]},  # mettre les valeurs pour 2
        expected_points_b={'C': [120, 105], 'D': [130, 90]},  # mettre les valeurs pour 7
        images=similar_image_caracteristics,
        preplace_points_a=True,
    )


def meilleures_caracteristiques(custom=True):
    reset_exercice_droite_carac()
    if custom:
        title = "Des meilleures caractéristiques"
        different_image_caracteristics = [
            files_url + "image_2_5x3_180_differentiante.png",
            files_url + "image_2_5x3_210-180_differentiante.png",
            files_url + "image_7_5x3_180_differentiante.png",
            files_url + "image_7_5x3_210-150_differentiante.png"
        ]
        exp_a = {'E': [140, 120], 'F': [160, 135]}
        exp_b = {'G': [100, 70], 'H': [110, 60]}

    else:
        title = "De l'image au plan"
        different_image_caracteristics = [
            files_url + "image_7_6x3_caracteristique.png",
            files_url + "image_2_6x3_caracteristique.png",
        ]
        exp_a = {'B': [140, 140]}
        exp_b = {'A': [100, 60]}
    placer_caracteristiques(
        html_title=title,
        expected_points_a=exp_a,  # mettre les valeurs pour 2
        expected_points_b=exp_b,  # mettre les valeurs pour 7
        images=different_image_caracteristics,
        preplace_points_a=False,
    )


def placer_2_points():
    meilleures_caracteristiques(False)


### --------------------------------- ###


def tracer_200_points(nb=200):
    id = uuid.uuid4().hex

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques,
                                                      d_train=common.challenge.d_train[0:nb],
                                                      r_train=common.challenge.r_train[0:nb])
    params = {
        'points': c_train_par_population,
        'hideClasses': True,
    }

    run_js(
        f"mathadata.add_observer('{id}-chart', () => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'''
        <canvas id="{id}-chart"></canvas>
    '''))


def tracer_points_droite_vecteur(id_content=None, carac=None, initial_hidden=False, save=True, normal=None,
                                 directeur=False, directeur_a=False, reglage_normal=False, initial_values=None,
                                 sliders=False, interception_point=True, equation_hide=True):
    if id_content is None:
        id_content = uuid.uuid4().hex

    if normal is None:
        normal = False

    if carac is None:
        carac = common.challenge.deux_caracteristiques

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

    params = {
        'points': c_train_par_population,
        'custom': carac == common.challenge.deux_caracteristiques_custom,
        'hover': True,
        'displayValue': False,
        'save': save,
        'equation_hide': equation_hide,
        'vecteurs': {
            'directeur': directeur,
            'directeur_a': directeur_a,
            'normal': normal,
        },
        'droite': {
            'mode': 'cartesienne'
        },
        'inputs': {
            'xa': True,
            'ya': True,
        },
        'compute_score': True,
        'initial_values': initial_values,
        'force_origin': True,
        'interception_point': interception_point,
    }

    if reglage_normal:
        params['inputs']['nx'] = True
        params['inputs']['ny'] = True
    else:
        params['inputs']['ux'] = True
        params['inputs']['uy'] = True

    # default values
    ux = 5
    uy = 10
    xa = 50
    ya = 50
    nx = 10
    ny = -5

    if initial_values:
        ux = initial_values.get('ux', ux)
        uy = initial_values.get('uy', uy)
        xa = initial_values.get('xa', xa)
        ya = initial_values.get('ya', ya)
        nx = initial_values.get('nx', nx)
        ny = initial_values.get('ny', ny)

    # Ensure JS receives the complete set of values
    if params.get('initial_values') is None:
        params['initial_values'] = {}
    params['initial_values'].update({
        'ux': ux, 'uy': uy, 'xa': xa, 'ya': ya, 'nx': nx, 'ny': ny
    })

    run_js(
        f"mathadata.add_observer('{id_content}-container', () => window.mathadata.tracer_points('{id_content}', '{json.dumps(params, cls=NpEncoder)}'))")

    # Mise en place du conteneur pour le graphique
    if sliders:
        display(HTML(f'''
            <div id="{id_content}-container" style="{'visibility:hidden;' if initial_hidden else ''}">
                <div id="{id_content}-score-container"
                    style="text-align:center; font-weight:bold; font-size:2rem;">
                    Pourcentage d'erreur : <span id="{id_content}-score">...</span>
                </div>

                <canvas id="{id_content}-chart"></canvas>

                <div id="{id_content}-inputs"
                     style="display:flex; flex-direction:column; gap:2rem; align-items:center;">
                
                    <!-- Ligne 1 -->
                    <div style="display:flex; flex-direction:row; gap:2rem; justify-content:center;">
                        <div style="
                            display:{'flex' if (directeur and not reglage_normal) else 'none'};
                            flex-direction:row; gap:1.5rem;">
                            <div>
                                <label style="color: green;">x<sub>u</sub> =
                                    <span id="{id_content}-ux-val">{ux}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-ux"
                                       value="{ux}" min="-20" max="20" step="0.1"
                                       oninput="document.getElementById('{id_content}-ux-val').textContent=this.value;">
                            </div>
                            <div>
                                <label style="color: firebrick;">y<sub>u</sub> =
                                    <span id="{id_content}-uy-val">{uy}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-uy"
                                       value="{uy}" min="-20" max="20" step="0.1"
                                       oninput="document.getElementById('{id_content}-uy-val').textContent=this.value;">
                            </div>
                        </div>
                
                        <div style="
                            display:{'flex' if reglage_normal else 'none'};
                            flex-direction:row; gap:1.5rem;">
                            <div>
                                <label>⃗n<sub>x</sub> =
                                    <span id="{id_content}-nx-val">{nx}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-nx"
                                       value="{nx}" min="-100" max="100" step="1"
                                       oninput="document.getElementById('{id_content}-nx-val').textContent=this.value;">
                            </div>
                            <div>
                                <label>⃗n<sub>y</sub> =
                                    <span id="{id_content}-ny-val">{ny}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-ny"
                                       value="{ny}" min="0" max="88" step="1"
                                       oninput="document.getElementById('{id_content}-ny-val').textContent=this.value;">
                            </div>
                        </div>
                    </div>
                
                    <!-- Ligne 2 -->
                    <div style="display:flex; flex-direction:row; gap:2rem; justify-content:center;">
                        <div>
                            <label>x<sub>A</sub> =
                                <span id="{id_content}-xa-val">{xa}</span>
                            </label>
                            <input type="range"
                                   id="{id_content}-input-xa"
                                   value="{xa}" min="0" max="88" step="1"
                                   oninput="document.getElementById('{id_content}-xa-val').textContent=this.value;">
                        </div>
                        <div>
                            <label>y<sub>A</sub> =
                                <span id="{id_content}-ya-val">{ya}</span>
                            </label>
                            <input type="range"
                                   id="{id_content}-input-ya"
                                   value="{ya}" min="0" max="88" step="1"
                                   oninput="document.getElementById('{id_content}-ya-val').textContent=this.value;">
                        </div>
                    </div>
                
                </div>


                </div>
            </div>
            '''))
    else:
        display(HTML(f'''
        <!-- Conteneur pour afficher le taux d'erreur -->
        <div id="{id_content}-container" style="{'visibility:hidden;' if initial_hidden else ''}">
            <div id="{id_content}-score-container"
                style="
                text-align: center;
                font-weight: bold;
                font-size: 2rem;
                ">
                Pourcentage d'erreur : <span id="{id_content}-score">...</span>
            </div>

            <!-- Zone canvas pour tracer le graphique -->
            <canvas id="{id_content}-chart"></canvas>

            <!-- Conteneur pour les champs d'entrée -->
            <div id="{id_content}-inputs"
                style="
                display: flex;
                gap: 2rem;
                justify-content: center;
                flex-direction: row;
                ">
                <!-- Cas « directeur » et pas en mode « reglage_normal » -->
                <div style="
                    display: {'flex' if (directeur and not reglage_normal) else 'none'};
                    flex-direction: row;
                    gap: 1.5rem;
                    ">
                    <!-- Paramètre ux -->
                    <div>
                        <label for="{id_content}-input-ux" id="{id_content}-label-ux">x<sub>u</sub> = </label>
                        <input type="number"
                            id="{id_content}-input-ux"
                            value="{ux}"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                    <!-- Paramètre uy -->
                    <div>
                        <label for="{id_content}-input-uy" id="{id_content}-label-uy">y<sub>u</sub> = </label>
                        <input type="number"
                            id="{id_content}-input-uy"
                            value="{uy}"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                </div>

                <!-- Cas du mode « reglage_normal » -->
                <div style="
                    display: {'flex' if reglage_normal else 'none'};
                    flex-direction: row;
                    gap: 1.5rem;
                    ">
                    <!-- Paramètre a -->
                    <div>
                        <label for="{id_content}-input-nx" id="{id_content}-label-nx">\u20D7n<sub>x</sub> = </label>
                        <input type="number"
                            id="{id_content}-input-nx"
                            value="{nx}"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                    <!-- Paramètre b -->
                    <div>
                        <label for="{id_content}-input-ny" id="{id_content}-label-ny">\u20D7n<sub>y</sub> = </label>
                        <input type="number"
                            id="{id_content}-input-ny"
                            value="{ny}"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                </div>

                <!-- Paramètre x_A -->
                <div style="display: flex; flex-direction: row; gap: 1.5rem;">
                    <div>
                        <label for="{id_content}-input-xa" id="{id_content}-label-xa">x<sub>A</sub> = </label>
                        <input type="number"
                            id="{id_content}-input-xa"
                            value="{xa}"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                    <!-- Paramètre y_A -->
                    <div>
                        <label for="{id_content}-input-ya" id="{id_content}-label-ya">y<sub>A</sub> = </label>
                        <input type="number"
                            id="{id_content}-input-ya"
                            value="{ya}"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                </div>
            </div>
        </div>
    '''))


def tracer_points_droite_vecteur_directeur():
    # Valeurs par défaut spécifiques pour cet exercice
    defaults = {'ux': 5, 'uy': 10, 'xa': 20, 'ya': 10}

    tracer_points_droite_vecteur(directeur=True, directeur_a=True, initial_values=defaults, sliders=True,
                                 interception_point=False)


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


pointA = (20, 40)
pointB = (30, 10)

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
      
    window.mathadata.findIntersectionPoints = function (
      a,
      b,
      c,
      x_min,
      x_max
    ) {
      const INF = (x_max - x_min) * 10; // Hack pour faire tendre les points de la droite vers l'infini (donc en dehors du canvas)
      const points = [];
    
      // Droite horizontale
      if (b === 0) {
        const x = -c / a;
        points.push({ x, y: x_min - INF }, { x, y: x_max + INF });
        return points;
      }
    
      // Droite verticale
      if (a === 0) {
        const y = -c / b;
        points.push({ x: x_min - INF, y }, { x: x_max + INF, y });
        return points;
      }
    
      // Points d'interesection -/+ facteur 'infini'
      const x1 = x_min - INF;
      const x2 = x_max + INF;
    
      // Renvoie deux points pour tracer la droite
      points.push({ x: x1, y: (-a * x1 - c) / b }, { x: x2, y: (-a * x2 - c) / b });
    
      return points;
    };

    window.mathadata.getLineEquationStr = function(a, b, c) {
        const round2 = v => Math.round((v + Number.EPSILON) * 100) / 100;
    
        a = round2(a);
        b = round2(b);
        c = round2(c);
    
        return `${a}x ${b < 0 ? '-' : '+'} ${Math.abs(b)}y ${c < 0 ? '-' : '+'} ${Math.abs(c)} = 0`;
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

      const {points, droite, vecteurs, centroides, additionalPoints, hideClasses, hover, inputs, initial_values, displayValue, save, custom, compute_score, drag, force_origin, equation_hide, param_colors, equation_fixed_position, side_box = true , interception_point = true} = params;
        // points: tableau des données en entrée sous forme de coordonnées (deux éléments, les points des 2 et les points des 7) [[[x,y],[x,y],...] , [[x,y],[x,y],...]]
        // droite: la droite à afficher (objet)
        // vecteurs: vecteurs à afficher pour le bouger
        //   - normal: bool: affiche le vecteur normal
        //   - directeur: bool: affiche le vecteur directeur (origin)
        //   - directeur_a: bool: affiche le vecteur directeur (attaché au point A)
        // centroides: bool: afficher les centroides
        // additionalPoints: tableau de points additionnels
        // hideClasses: bool: (default: false) pour afficher la légende au dessus du graphe
        // hover: objet par défault, peux être appelé comme booleen pour un affichage hover selon son type
        // drag: bool: autorise à bouger les points
        // force_origin: bool: force le cadre à l'origine
        // custom: bool: pour le calcul du score, est-ce qu'on utilise la caractèristique custom ?
        // compute_score: bool: affichage du score
        // displayValue: bool: relatif à l'affichage des valeurs des inputs externes
        // equation_hide: bool: masque l'equation de la droite
        // param_colors: affiche les couleurs de m et p
        // equation_fixed_position: bool: fixe l'équation dans le coin inférieur droit au lieu de la dessiner le long de la droite
        // inputs: objet avec les inputs à afficher et gérer (ux, uy, xa, ya, nx, ny, a, b, c)
        // initial_values: valeurs initiales des inputs (certaines valeurs peuvent etre fixées auquel cas il y a une valeur initiale mais pas d'input)
        // interception_point: affiche le point d'interception avec l'axe des ordonnées
        
      const getColoredEquation = (m, p, colors) => {
        if (!colors || (!colors.m && !colors.p)) {
            return `y = ${m}x ${p < 0 ? '-' : '+'} ${Math.abs(p)}`;
        }

        const mColor = colors.m || 'black';
        const pColor = colors.p || 'black';
        const sign = p < 0 ? '-' : '+';
        const absP = Math.abs(p);

        return {
            html: `y = <span style="color: ${mColor}">${m}</span>x ${sign} <span style="color: ${pColor}">${absP}</span>`,
            text: `y = ${m}x ${sign} ${absP}`
        };
      };

      const getYInterceptPoint = (vals) => {
        if (!vals || vals.b === undefined || vals.c === undefined) {
            return null;
        }
        if (vals.b === 0) {
            return null;
        }
        const y0 = -vals.c / vals.b;
        if (!isFinite(y0)) {
            return null;
        }
        return { x: 0, y: y0 };
      };
      
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
      if (initial_values) {
          Object.assign(values, initial_values)
      }

      // Slope box (optionnel)
      const slopeBox = side_box ? document.getElementById(`${id}-slope-box`) : null;
      if (slopeBox && !slopeBox.dataset.init) {
        slopeBox.style.border = 'none'; // Remove border if any
        slopeBox.innerHTML = `
          <svg id="${id}-slope-svg" width="100%" height="100%" viewBox="0 0 250 180" style="overflow: visible;">
              <line id="${id}-slope-base" stroke="purple" stroke-width="4" stroke-linecap="round" />
              <line id="${id}-slope-height" stroke="orange" stroke-width="4" stroke-linecap="round" />
              <line id="${id}-slope-hypo" stroke="black" stroke-width="5" stroke-linecap="round" />
              <text id="${id}-text-base" text-anchor="middle" font-weight="bold" font-size="16" fill="purple">10</text>
              <text id="${id}-text-calc" text-anchor="start" font-weight="bold" font-size="16" fill="orange"></text>
              <text id="${id}-text-formula" text-anchor="middle" font-weight="bold" font-size="16" fill="black" x="125" y="160"></text>
          </svg>
        `;
        slopeBox.dataset.init = '1';
      }

      const getSlope = () => {
        if (values.m !== undefined) return values.m;
        if (values.a !== undefined && values.b !== undefined && values.b !== 0) {
          return -(values.a / values.b);
        }
        return null;
      };

      const updateSlopeBox = () => {
        if (!slopeBox) return;
        const svg = document.getElementById(`${id}-slope-svg`);
        if (!svg) return;
        const baseLine = document.getElementById(`${id}-slope-base`);
        const heightLine = document.getElementById(`${id}-slope-height`);
        const hypoLine = document.getElementById(`${id}-slope-hypo`);
        const textBase = document.getElementById(`${id}-text-base`);
        const textCalc = document.getElementById(`${id}-text-calc`);
        const textFormula = document.getElementById(`${id}-text-formula`);
        if (!baseLine || !heightLine || !hypoLine || !textBase || !textCalc) return;

        const mVal = getSlope() ?? 0;
        
        // Dynamic width calculation
        let vizBase = 80; 
        const chart = mathadata.charts[`${id}-chart`];
        let boxHeight = 180;

        if (chart) {
            const chartArea = chart.chartArea;
            const scales = chart.scales;
            if (chartArea && scales && scales.x) {
                const graphWidthPixels = chartArea.width;
                const xRange = scales.x.max - scales.x.min;
                if (xRange > 0) {
                    vizBase = (10 / xRange) * graphWidthPixels;
                }
            }
        }

        const centerY = boxHeight / 2;
        const hViz = vizBase * mVal;
        
        const Ax = 125 - vizBase / 2;
        const Ay = centerY + hViz / 2;
        const Bx = 125 + vizBase / 2;
        const By = Ay;
        const Cx = Bx;
        const Cy = Ay - hViz;

        baseLine.setAttribute('x1', Ax);
        baseLine.setAttribute('y1', Ay);
        baseLine.setAttribute('x2', Bx);
        baseLine.setAttribute('y2', By);

        heightLine.setAttribute('x1', Bx);
        heightLine.setAttribute('y1', By);
        heightLine.setAttribute('x2', Cx);
        heightLine.setAttribute('y2', Cy);

        hypoLine.setAttribute('x1', Ax);
        hypoLine.setAttribute('y1', Ay);
        hypoLine.setAttribute('x2', Cx);
        hypoLine.setAttribute('y2', Cy);

        textBase.setAttribute('x', (Ax + Bx) / 2);
        textBase.setAttribute('y', Ay + 20);

        textCalc.setAttribute('x', Bx + 10);
        textCalc.setAttribute('y', (By + Cy) / 2 + 5);

        const mDisp = parseFloat(mVal.toFixed(2));
        const resDisp = parseFloat((10 * mVal).toFixed(2));
        textCalc.textContent = resDisp;

        if (textFormula) {
            // Position dynamique de la formule pour éviter le chevauchement
            // Ay est la coordonnée Y de la base du triangle (le bas du triangle si m > 0)
            // On place le texte un peu en dessous de Ay
            // Si m < 0, Ay est le haut du triangle, mais le triangle descend vers Cy.
            // Dans ce cas, le bas du triangle est Cy.
            // Donc on prend le max(Ay, Cy) pour trouver le point le plus bas du triangle.
            
            const triangleBottomY = Math.max(Ay, Cy);
            const formulaY = Math.max(160, triangleBottomY + 60); // Au moins 160, ou plus bas si nécessaire
            textFormula.setAttribute('y', formulaY);

            textFormula.textContent = "";
            const mColor = param_colors?.m || '#239E28';
            var parts = [
                {text: "m", color: mColor},
                {text: " = ", color: "black"},
                {text: resDisp, color: "orange"},
                {text: " / ", color: "black"},
                {text: "10", color: "purple"},
                {text: " = ", color: "black"},
                {text: mDisp, color: mColor}
            ];
            parts.forEach(function(p) {
                var ts = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
                ts.textContent = p.text;
                ts.setAttribute("fill", p.color);
                textFormula.appendChild(ts);
            });
        }
      };
      
      const plugins = [];

      // Colors for the populations
      const colors = droite ? mathadata.classColorCodes.map(c => `rgb(${c})`) : mathadata.classColors;
      
      // Prepare the data for Chart.js
      
      // Points (dataset 0 and 1)
      const datasets = points.map((set, index) => {
          return {
              label: hideClasses ? '?' : mathadata.classe(index, {alt: true, plural: true, uppercase: true}),
              data: [],
              backgroundColor: colors[index],
              borderColor: colors[index],
              pointStyle: 'cross',
              pointRadius: 5,
              order: 1,
          }
      });
      
      let max, min;
      let start_ux, start_uy;
      let droiteDatasetIndex;
      let yInterceptDatasetIndex;
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
            
            // Update start_ux and start_uy based on new min/max
            start_ux = Math.round((max + min) / 2 / 10) * 10
            start_uy = Math.round((min + 10) / 10) * 10
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

        if (yInterceptDatasetIndex !== undefined) {
            const interceptData = getYInterceptPoint(values);
            datasets[yInterceptDatasetIndex].data = interceptData ? [interceptData] : [];
        }

        if (uDatasetIndex) {
            datasets[uDatasetIndex].data = [{ x: start_ux, y: start_uy }, { x: start_ux + values.ux, y: start_uy + values.uy }]
        }
        if (nDatasetIndex) {
            datasets[nDatasetIndex].data = [{ x: values.xa, y: values.ya }, { x: values.xa + values.nx, y: values.ya + values.ny }]
        }
        if (uaDatasetIndex) {
            datasets[uaDatasetIndex].data = [{ x: values.xa, y: values.ya }, { x: values.xa + values.ux, y: values.ya + values.uy }]
        }
        if (aDatasetIndex) {
            datasets[aDatasetIndex].data = [{ x: values.xa, y: values.ya }]
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

        updateSlopeBox();
      }

        // rend la fonction updatePoints accessible via l'objet chart
      plugins.push({
        beforeInit(chart, args, options) {
            chart.updatePoints = updatePoints;
        }
      })

      let uDatasetIndex, nDatasetIndex, uaDatasetIndex, aDatasetIndex;
      if (vecteurs) {
        const {normal, directeur, directeur_a} = vecteurs;
        const vectorParams = []

        if (directeur) {
          // add vector u
          datasets.push({
              type: 'line',
              data: [],
              borderColor: 'purple',
              borderWidth: 2,
              pointRadius: 0,
              pointHitRadius: 0,
              label: '\u20D7u',
          }); 
          vectorParams.push({
            datasetIndex: datasets.length - 1,
            color: 'purple',
            label: '\u20D7u',
            id: 'directeur',
          })
          uDatasetIndex = datasets.length - 1;
        }

        if (directeur_a) {
          // add vector u attached to A
          datasets.push({
              type: 'line',
              data: [],
              borderColor: 'purple',
              borderWidth: 2,
              pointRadius: 0,
              pointHitRadius: 0,
              label: '',
          }); 
          vectorParams.push({
            datasetIndex: datasets.length - 1,
            color: 'purple',
            label: '',
            id: 'directeur_a',
          })
          uaDatasetIndex = datasets.length - 1;
        }

        if (normal) {
          // add vector n
          datasets.push({
              type: 'line',
              data: [],
              borderColor: 'brown',
              borderWidth: 2,
              pointRadius: 0,
              pointHitRadius: 0,
              label: '\u20D7n',
          });
          vectorParams.push({
            datasetIndex: datasets.length - 1,
            color: 'brown',
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
                if (label) {
                    let label_x, label_y;
                    if (id === 'directeur' || id === 'directeur_a') {
                        label_x = values.ux;
                        label_y = values.uy;
                    } else {
                        label_x = values.nx;
                        label_y = values.ny;
                    }
                    if (directeur_a) {
                        ctx.fillText(`${label}(`, x2 + 10, y2);
                        let currentX = x2 + 10 + ctx.measureText(`${label}(`).width;
                        
                        ctx.fillStyle = 'green';
                        ctx.fillText(`${label_x}`, currentX, y2);
                        currentX += ctx.measureText(`${label_x}`).width;
                        
                        ctx.fillStyle = color;
                        ctx.fillText(`, `, currentX, y2);
                        currentX += ctx.measureText(`, `).width;
                        
                        ctx.fillStyle = 'firebrick';
                        ctx.fillText(`${label_y}`, currentX, y2);
                        currentX += ctx.measureText(`${label_y}`).width;
                        
                        ctx.fillStyle = color;
                        ctx.fillText(`)`, currentX, y2);
                    }
                    else {
                        ctx.fillText(`${label}(${label_x}, ${label_y})`, x2 + 10, y2);
                    }
                }
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
              label: `Point moyen de la classe ${mathadata.challenge.classes[index]}`,
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
                        ctx.fillText(`point moyen ${index === 0 ? mathadata.challenge.classes[0] : mathadata.challenge.classes[1]}`, element.x + 18, element.y - 3); // Adjusted X and Y offset
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
            if (param_colors) {
                const coloredEq = getColoredEquation(m, p, param_colors);
                label = coloredEq.text; // Pour le label du dataset, on utilise le texte simple
            } else {
                label = `y = ${m}x ${p < 0 ? '-' : '+'} ${Math.abs(p)}`;
            }
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

          if (m !== undefined && p !== undefined) {
            values.m = m
            values.p = p
          }

          if (a !== undefined && b !== undefined && c !== undefined) {
            values.a = a
            values.b = b
            values.c = c
          }

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

                    let textX, textY, rotationAngle;

                    if (equation_fixed_position) {
                        // Position fixe dans le coin inférieur droit
                        textX = chart.chartArea.right - 10;
                        textY = chart.chartArea.bottom - 10;
                        rotationAngle = 0; // Pas de rotation pour la position fixe
                    } else {
                        // Position le long de la droite (comportement par défaut)
                        textX = Math.min(x2 + 20, chart.chartArea.right - 5);
                        textY = Math.min(y2 + 20, chart.chartArea.bottom - 5);
                        rotationAngle = angle;
                    }

                    ctx.save();
                    ctx.translate(textX, textY);
                    ctx.rotate(rotationAngle);
                    ctx.font = '18px Arial';
                    ctx.fillStyle = 'black';
                    ctx.textAlign = 'right';
                    
                    if ((values.m !== undefined && values.p !== undefined) && mode !== 'cartesienne' && param_colors) {
                            // Affichage avec couleurs pour m et p
                            const mColor = param_colors.m || 'black';
                            const pColor = param_colors.p || 'black';
                            const sign = values.p < 0 ? '-' : '+';
                            const absP = Math.abs(values.p);

                            // Comme le texte est aligné à droite, on dessine de droite à gauche
                            let currentX = 0;

                            // Dessiner la valeur de p avec sa couleur (le plus à droite)
                            ctx.fillStyle = pColor;
                            const pText = `${absP}`;
                            ctx.fillText(pText, currentX, 0);
                            currentX -= ctx.measureText(pText).width;

                            // Dessiner " " + signe + " "
                            ctx.fillStyle = 'black';
                            const signText = ` ${sign} `;
                            ctx.fillText(signText, currentX, 0);
                            currentX -= ctx.measureText(signText).width;

                            // Dessiner "x"
                            const xText = 'x';
                            ctx.fillText(xText, currentX, 0);
                            currentX -= ctx.measureText(xText).width;

                            // Dessiner la valeur de m avec sa couleur
                            ctx.fillStyle = mColor;
                            const mText = `${values.m}`;
                            ctx.fillText(mText, currentX, 0);
                            currentX -= ctx.measureText(mText).width;

                            // Dessiner "y = " (le plus à gauche)
                            ctx.fillStyle = 'black';
                            const yEqualText = 'y = ';
                            ctx.fillText(yEqualText, currentX, 0);
                        } else {
                            ctx.fillStyle = 'black';
                            let equation;
                            if ((values.m !== undefined && values.p !== undefined) && mode !== 'cartesienne') {
                                equation = `y = ${values.m}x ${values.p < 0 ? '-' : '+'} ${Math.abs(values.p)}`;
                            } else {
                                equation = mathadata.getLineEquationStr(values.a, values.b, values.c);
                            }
                            ctx.fillText(equation, 0, 0);
                        }
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
                backgroundColor: `rgba(${mathadata.classColorCodes[mathadata.challenge.classes.indexOf(mathadata.challenge.strings.r_grande_caracteristique)]}, 0.1)`,
            });

            datasets.push({
                label: `Zone inférieure (${mathadata.challenge.strings.r_petite_caracteristique})`,
                type: 'line',
                data: lineDataVertical,
                pointsRadius: 0,
                pointHitRadius: 0,
                borderColor: 'transparent',
                fill: 'origin',
                backgroundColor: `rgba(${mathadata.classColorCodes[mathadata.challenge.classes.indexOf(mathadata.challenge.strings.r_petite_caracteristique)]}, 0.1)`,
            });
          }

          if (compute_score) {
            computeScore()
          }

          // Ordonnée à l'origine (point rouge sur l'axe des ordonnées)
          if (interception_point) {
              const initialIntercept = getYInterceptPoint(values);
              datasets.push({
                  label: "Ordonnée à l'origine",
                  data: initialIntercept ? [initialIntercept] : [],
                  backgroundColor: 'red',
                  borderColor: 'red',
                  pointStyle: 'circle',
                  pointRadius: 6,
                  pointHoverRadius: 6,
                  order: 2,
              });
              yInterceptDatasetIndex = datasets.length - 1;
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
      
      // initialisation (après création des datasets, incluant l'ordonnée à l'origine)
      updatePoints(points, {animate: false})
      
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
        if (side_box) {
          updateSlopeBox();
        }

        if (inputs) {
          // On suppose que les inputs ont un élément html correspondant avec l'id {id}-input-{key}
          const inputElements = {}
          
          const update = () => {
            const chart = mathadata.charts[`${id}-chart`];

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
              newValues.nx = -newValues.uy
              newValues.ny = newValues.ux
            }

            if (newValues.nx !== undefined && newValues.ny !== undefined) {
              newValues.ux = newValues.ny
              newValues.uy = -newValues.nx

              if (newValues.xa !== undefined && newValues.ya !== undefined) {
                newValues.a = -newValues.nx
                newValues.b = -newValues.ny
                newValues.c = newValues.nx * newValues.xa + newValues.ny * newValues.ya
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
            if (uaDatasetIndex) {
              datasets[uaDatasetIndex].data = [{ x: values.xa, y: values.ya }, { x: values.xa + values.ux, y: values.ya + values.uy }]
            }
            if (nDatasetIndex) {
              datasets[nDatasetIndex].data = [{ x: values.xa, y: values.ya }, { x: values.xa + values.nx, y: values.ya + values.ny }]
            }
            if (aDatasetIndex) {
              datasets[aDatasetIndex].data = [{ x: values.xa, y: values.ya }]
            }

            let minWithIntercept = min;
            let maxWithIntercept = max;
            if (yInterceptDatasetIndex !== undefined) {
              const interceptData = getYInterceptPoint(values);
              datasets[yInterceptDatasetIndex].data = interceptData ? [interceptData] : [];
              if (interceptData) {
                minWithIntercept = Math.min(min, interceptData.y);
                maxWithIntercept = Math.max(max, interceptData.y);
                chart.options.scales.x.min = minWithIntercept
                chart.options.scales.x.max = maxWithIntercept
                chart.options.scales.y.min = minWithIntercept
                chart.options.scales.y.max = maxWithIntercept
              }
            }
            if (droiteDatasetIndex) {
              datasets[droiteDatasetIndex].data = mathadata.findIntersectionPoints(values.a, values.b, values.c, minWithIntercept, maxWithIntercept);
            }


            mathadata.charts[`${id}-chart`].update()

            // Update du score
            if (compute_score) {
              computeScore()
            }

            updateSlopeBox();
          }

          // Initialisation point A
          if (inputs.xa !== undefined && inputs.ya !== undefined && values.xa === undefined && values.ya === undefined) {
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

              

          console.log('initial values', values)
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


def calculer_score_droite_geo(custom=False, validate=None, error_msg=None, banque=True, success_msg=None,
                              animation=True):
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

    if validate is not None and validate <= base_score <= 100 - validate:
        if error_msg is None:
            print_error(
                f"Tu pourras passer à la suite quand tu auras un pourcentage d'erreur de moins de {validate}%.")
        else:
            print_error(error_msg)

    def algorithme(d):
        k = deux_caracteristiques(d)
        return estim_2d(input_values['a'], input_values['b'], input_values['c'], k, above, below)

    def cb(score):
        if (validate is not None and score * 100 <= validate) or (
                has_variable('superuser') and get_variable('superuser') == True):
            if success_msg is None:
                pretty_print_success("Bravo, tu peux passer à la suite.")
            else:
                print(success_msg)
            pass_breakpoint()

    calculer_score(algorithme, cb=cb,
                   banque=banque, animation=animation)


def qcm_choix_caracteristiques():
    create_qcm({
        'question': 'Quelles zones choisir pour mieux distinguer les 2 de 7 ?',
        'choices': ['1er choix', '2ème choix'],
        'answer': '2ème choix',
    })


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
            f"Ce n'est pas la bonne valeur. Tu as donné le nombre d'erreurs ({nb_erreurs}) et non le pourcentage d'erreur.")
        return False

    # Vérification de la réponse correcte
    if user_answer == pourcentage_erreur:
        if nb_erreurs == 0:
            pretty_print_success(
                "Bravo, c'est la bonne réponse. Il n'y a aucune erreur de classification sur ce schéma.")
        else:
            # Détails sur les erreurs pour le message
            pretty_print_success(
                f"Bravo, c'est la bonne réponse. Il y a {nb_erreurs_par_classe[0]} {classe(0, alt=True, plural=(nb_erreurs_par_classe[0] > 1))} {'au dessus' if common.challenge.classes[0] == common.challenge.r_petite_caracteristique else 'en dessous'} de la droite et {nb_erreurs_par_classe[1]} {classe(1, alt=True, plural=(nb_erreurs_par_classe[1] > 1))} {'au dessus' if common.challenge.classes[1] == common.challenge.r_petite_caracteristique else 'en dessous'}, donc {nb_erreurs} erreurs soit {pourcentage_erreur}%.")
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
                'else': f"classe_points_bleus n'a pas la bonne valeur. Tu dois répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
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
                'else': f"classe_points_oranges n'a pas la bonne valeur. Tu dois répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
            }
        ]
    }
}, tips=[
    {
        'seconds': 15,
        'trials': 1,
        'operator': 'OR',
        'tip': f"Il faut répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}"
    }
])

validation_execution_placer_2_points = MathadataValidate(success="")
validation_execution_10_points = MathadataValidate(success="")
validation_execution_20_points = MathadataValidate(success="")
validation_question_score_droite = MathadataValidateVariables({
    'erreur_10': None
},
    function_validation=function_validation_score_droite,
    success="")
validation_execution_tracer_points_droite = MathadataValidate(success="")
validation_execution_tracer_points_droite_c = MathadataValidate(success="")
validation_execution_tracer_points_droite_a_b = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Bien joué, tu peux passer à la partie suivante.")
validation_execution_mauvaises_caracteristiques = MathadataValidate(success="")
validation_execution_meilleures_caracteristiques = MathadataValidate(success="")
validation_score_droite_custom = MathadataValidate(
    success="Bravo, tu peux continuer à essayer d'améliorer ton score. Il est possible de faire seulement 3% d'erreur.")
validation_execution_scatter_caracteristiques_ripou = MathadataValidate(success="")
validation_execution_afficher_customisation = MathadataValidate(success="")


def function_validation_carac(errors, answers):
    moyenne_haut_2 = answers['x_2']
    moyenne_bas_2 = answers['y_2']

    error_str = "Réessaie, il y a des erreurs dans les coordonnées des caractéristiques :\n"
    len_start = len(error_str)
    if moyenne_haut_2 != 140:
        error_str += "L'abscisse de 2 est incorrecte.\n"
    if moyenne_bas_2 != 140:
        error_str += "L'ordonnée de 2 est incorrecte."
    if len(error_str) > len_start:
        errors.append(error_str)
        return False
    return True


validation_carac = MathadataValidateVariables({
    'x_2': None,
    'y_2': None
}, function_validation=function_validation_carac)


def set_exercice_droite_carac_ok():
    global moyenne_carac
    moyenne_carac = True


def validate_moyenne_carac(errors, answers):
    if moyenne_carac:
        return True
    else:
        errors.append("Réponds d'abord à la question ci-dessus en plaçant les points sur le graphe.")
        return False


chat = 'chat'
Chat = 'Chat'


def function_validation_cartesienne_determinante(errors, answers):
    a = answers['a']
    if not isinstance(a, (int, float, str)):
        errors.append("La valeur de a doit être un nombre. Si tu veux passer à la suite, répond 'chat'")
        return False
    if a == chat or a == 'chat' or a == 'Chat':
        pretty_print_success("Pas de galère, on continue")
        return True
    if a != 5:
        errors.append("La valeur de a est incorrecte. Réessaie. Si tu veux passer à la suite, répond 'chat'")
        return False
    pretty_print_success("Bravo, tu peux passer à la suite.")
    return True


validation_placer_2_points = MathadataValidate(function_validation=validate_moyenne_carac,
                                               success="Bravo tu as bien placé les points !")

validation_moyenne_carac_mauvaise = MathadataValidate(function_validation=validate_moyenne_carac)
validation_moyenne_carac_meilleure = MathadataValidate(function_validation=validate_moyenne_carac)

validation_question_cartesienne_determinant = MathadataValidateVariables({
    'a': None,
}, function_validation=function_validation_cartesienne_determinante
    , success="")
