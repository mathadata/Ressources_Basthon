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
                            expected_points_a=None, expected_points_b=None, preplace_points_a=False,
                            known_points_a=None, known_points_b=None,
                            line_params=None, show_zones=False,
                            zone_colors=None, panel_w=None, panel_h=None, left_ratio=None,
                            hide_expected_labels=False, expected_point_color=None,
                            keep_aspect_ratio=True,
                            image_caption_html=None, show_legend=False, force_origin=False, 
                            preplace_known_points=False, preplace_mystere=False,
                            interactive=True, known_points_animate=True,
                            checkpoint_enabled=True, show_status=True, exercise_validation=True,
                            auto_pass_on_known_points_done=False,
                            match_tolerance=0.35, click_delete_tolerance=0.45,
                            hide_left_panel=False, instance_id=None,
                            known_points_order=None):
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

    width = 830 if panel_w is None else int(panel_w)
    panel_h = 370 if panel_h is None else int(panel_h)

    if left_ratio is None:
        left_ratio = 0.35

    match_tolerance = float(match_tolerance)
    click_delete_tolerance = float(click_delete_tolerance)

    image_mode = "contain"
    # normalize images list to length 4
    if images is None:
        images = [None, None, None, None]
    else:
        images = list(images)[:4] + [None] * max(0, 4 - len(images))

    if instance_id is None:
        instance_id = uuid.uuid4().hex
    else:
        instance_id = str(instance_id)

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

    if known_points_a is None:
        known_points_a = {}
    if known_points_b is None:
        known_points_b = {}
    if zone_colors is None:
        zone_colors = {"above": "rgba(76,110,245,0.15)", "below": "rgba(246,200,95,0.15)"}
    known_json_a = json.dumps(known_points_a, sort_keys=True)
    known_json_b = json.dumps(known_points_b, sort_keys=True)
    # Optionnelle : séquence d'apparition des points connus pour l'animation
    known_queue = None
    if known_points_order:
        known_queue = known_points_order
    known_queue_json = json.dumps(known_queue) if known_queue is not None else "null"
    expected_color_json = json.dumps(expected_point_color) if expected_point_color is not None else "null"
    line_json = json.dumps(line_params) if line_params is not None else "null"
    zones_json = json.dumps(zone_colors)

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

    if image_caption_html is None:
        image_caption_html = ""

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
      .left { width: LEFT_W_PX; min-width:120px; display:flex; flex-direction:column; align-items:stretch; justify-content:center; gap:6px; }
      .right { flex:1; display:flex; flex-direction:column; align-items:stretch; justify-content:center; gap:6px; }
      HIDE_LEFT_PANEL_CSS

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

      .img-wrap {
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        width:100%;
        height:100%;
        gap:6px;
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

      .img-caption { width:100%; font-size:12px; color:#333; text-align:center; margin-top:6px; }

      .legend { display:flex; gap:14px; align-items:center; justify-content:center; font-size:13px; color:#333; }
      .legend-item { display:flex; align-items:center; gap:6px; }
      .legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }

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
          <div class="img-caption" id="img-caption" style="display:none;"></div>
        </div>

        <div class="right" aria-label="interactive column">
          <div class="legend" id="legend" style="display:none;">
            <div class="legend-item"><span class="legend-dot" id="legend-dot-a"></span>Exemples d'images de 2</div>
            <div class="legend-item"><span class="legend-dot" id="legend-dot-b"></span>Exemples d'images de 7</div>
          </div>
          <div id="box"></div>
        </div>
      </div>

	      <script src="https://cdnjs.cloudflare.com/ajax/libs/jsxgraph/1.4.0/jsxgraphcore.js"></script>
	      <script>
	        (function(){
	          var parentInstanceId = INSTANCE_ID_PLACEHOLDER;
	          // expectedPoints are JS objects: { label: [x,y], ... }
	          var expectedPointsA = EXPECTED_A_PLACEHOLDER;
	          var expectedPointsB = EXPECTED_B_PLACEHOLDER;
	          var knownPointsA = KNOWN_A_PLACEHOLDER;
	          var knownPointsB = KNOWN_B_PLACEHOLDER;
          var lineParams = LINE_PARAMS_PLACEHOLDER;
          var showZones = SHOW_ZONES_PLACEHOLDER;
          var showLegend = SHOW_LEGEND_PLACEHOLDER;
          var forceOrigin = FORCE_ORIGIN_PLACEHOLDER;
          var zoneColors = ZONE_COLORS_PLACEHOLDER;
          var hideExpectedLabels = HIDE_EXPECTED_LABELS_PLACEHOLDER;
          var expectedPointColor = EXPECTED_POINT_COLOR_PLACEHOLDER;
          var preplaceKnownPoints = PREPLACE_KNOWN_POINTS;
          var preplaceMystere = PREPLACE_MYSTERE_PLACEHOLDER;
          var interactive = INTERACTIVE_PLACEHOLDER;
          var knownPointsAnimate = KNOWN_POINTS_ANIMATE_PLACEHOLDER;
          var exerciseValidation = EXERCISE_VALIDATION_PLACEHOLDER;

          function normalizePoints(obj) {
              if (!obj) return [];
              if (Array.isArray(obj)) return obj;
              if (typeof obj === 'string') {
                  try {
                      var parsed = JSON.parse(obj);
                      if (Array.isArray(parsed)) return parsed;
                      if (parsed && typeof parsed === 'object') return Object.values(parsed);
                  } catch (e) {
                      return [];
                  }
              }
              if (typeof obj === 'object') return Object.values(obj);
              return [];
          }
          var preplaceGroupA = PREPLACE_A_PLACEHOLDER;
          var keepAspectRatio = KEEP_ASPECT_RATIO_PLACEHOLDER;
          var matchTol = MATCH_TOL_PLACEHOLDER;
          var clickDeleteTol = CLICK_DELETE_TOL_PLACEHOLDER;
          var defaultColor = "DEFAULT_COLOR_PLACEHOLDER";
          var matchedColorA = "MATCHED_COLOR_A_PLACEHOLDER";
          var matchedColorB = "MATCHED_COLOR_B_PLACEHOLDER";
          var knownQueue = KNOWN_QUEUE_PLACEHOLDER;

          var images = IMAGES_PLACEHOLDER || [null, null, null, null];
          var imageCaptionHtml = IMAGE_CAPTION_PLACEHOLDER;
          // Filtrer les images non-null pour obtenir les images réelles
          var actualImages = images.filter(function (img) {
              return img !== null && img !== undefined;
          });
          var imageCount = actualImages.length;

          // legend (optional)
          if (showLegend) {
              var legend = document.getElementById('legend');
              if (legend) {
                  legend.style.display = 'flex';
                  var dotA = document.getElementById('legend-dot-a');
                  var dotB = document.getElementById('legend-dot-b');
                  if (dotA) dotA.style.background = matchedColorA;
                  if (dotB) dotB.style.background = matchedColorB;
              }
          }
    
          function fillImageCell(id, src, idx) {
              var el = document.getElementById(id);
              if (!el) return; // Si l'élément n'existe pas, on ignore
    
              el.innerHTML = "";
              if (!src) {
                  // Ne plus afficher de placeholder, masquer la cellule
                  el.style.display = 'none';
                  return;
              }
              var wrap = document.createElement('div');
              wrap.className = 'img-wrap';
              var img = document.createElement('img');
              img.src = src;
              img.alt = 'image ' + (idx + 1);
              wrap.appendChild(img);
              if (imageCaptionHtml && imageCount === 1 && id === 'img2') {
                  var cap = document.createElement('div');
                  cap.className = 'img-caption';
                  cap.innerHTML = imageCaptionHtml;
                  wrap.appendChild(cap);
              }
              el.appendChild(wrap);
              el.style.display = 'block';
          }
    
          // Gérer l'affichage selon le nombre d'images
          if (imageCount === 1) {
              // Cas d'1 image : centrer l'affichage (cellule 2)
              fillImageCell('img1', null, 0); // Masquer
              fillImageCell('img2', actualImages[0], 1); // Image
              fillImageCell('img3', null, 2); // Masquer
              fillImageCell('img4', null, 3); // Masquer
          } else if (imageCount === 2) {
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

	          var knownPlaced = false;
          function drawLineAndZones() {
            if (!lineParams) return;
            try {
              var bb = board.getBoundingBox();
              var xminB = bb[0], ymaxB = bb[1], xmaxB = bb[2], yminB = bb[3];
              var m = lineParams.m;
              var p = lineParams.p;
              var x1 = xminB;
              var x2 = xmaxB;
              var y1 = m * x1 + p;
              var y2 = m * x2 + p;
              board.create('line', [[x1, y1], [x2, y2]], {
                strokeWidth: 2,
                strokeColor: '#000',
                fixed: true
              });

              if (showZones) {
                // compute line intersections with bounding box
                var pts = [];
                function addIfValid(px, py) {
                  if (px >= xminB - 1e-6 && px <= xmaxB + 1e-6 && py >= yminB - 1e-6 && py <= ymaxB + 1e-6) {
                    pts.push([px, py]);
                  }
                }
                // vertical edges
                addIfValid(xminB, m * xminB + p);
                addIfValid(xmaxB, m * xmaxB + p);
                // horizontal edges
                if (m !== 0) {
                  addIfValid((yminB - p) / m, yminB);
                  addIfValid((ymaxB - p) / m, ymaxB);
                }
                // keep two unique points
                var uniquePts = [];
                pts.forEach(function(pt) {
                  if (!uniquePts.some(function(u){ return Math.abs(u[0]-pt[0])<1e-6 && Math.abs(u[1]-pt[1])<1e-6; })) {
                    uniquePts.push(pt);
                  }
                });
                if (uniquePts.length < 2) {
                  uniquePts = [[xminB, m * xminB + p], [xmaxB, m * xmaxB + p]];
                }
                var A = uniquePts[0];
                var B = uniquePts[1];
                if (A[0] > B[0]) { var tmp = A; A = B; B = tmp; }

                var polyAbove = board.create('polygon', [
                  [xminB, ymaxB], [xmaxB, ymaxB], [B[0], B[1]], [A[0], A[1]]
                ], {
                  fillColor: zoneColors.above || "rgba(76,110,245,0.25)",
                  fillOpacity: 0.28,
                  hasInnerPoints: true,
                  borders: {visible: false},
                  fixed: true,
                  withLabel: false,
                  name: '',
                  highlight: false,
                  highlightFillColor: zoneColors.above || "rgba(76,110,245,0.25)",
                  highlightFillOpacity: 0.28,
                  highlightStrokeColor: 'transparent'
                });
                var polyBelow = board.create('polygon', [
                  [xminB, yminB], [xmaxB, yminB], [B[0], B[1]], [A[0], A[1]]
                ], {
                  fillColor: zoneColors.below || "rgba(246,200,95,0.25)",
                  fillOpacity: 0.28,
                  hasInnerPoints: true,
                  borders: {visible: false},
                  fixed: true,
                  withLabel: false,
                  name: '',
                  highlight: false,
                  highlightFillColor: zoneColors.below || "rgba(246,200,95,0.25)",
                  highlightFillOpacity: 0.28,
                  highlightStrokeColor: 'transparent'
                });
                if (polyAbove && polyAbove.vertices) {
                  polyAbove.vertices.forEach(function(v){ v.setAttribute({visible:false}); });
                }
                if (polyBelow && polyBelow.vertices) {
                  polyBelow.vertices.forEach(function(v){ v.setAttribute({visible:false}); });
                }
                if (polyAbove && polyAbove.label) {
                  polyAbove.label.setAttribute({visible:false});
                }
                if (polyBelow && polyBelow.label) {
                  polyBelow.label.setAttribute({visible:false});
                }
              }
            } catch (e) { console.error('drawLineAndZones error', e); }
          }
          
	          function placeKnownPoints(onComplete) {
	            if (knownPlaced) return;
	            try {
	              function notifyKnownPointsDone() {
	                try {
	                  window.parent.postMessage({type:'known_points_done', instance_id: parentInstanceId}, '*');
	                } catch (e) {}
	              }
              var addPoint = function(coords, color, label) {
                var p = board.create('point', coords, {
                  withLabel: true,
                  size: 4,
                  name: label || '',
                  color: color,
                  fillColor: color,
                  strokeColor: '#000',
                  fixed: true,
                  frozen: true,
                  highlight: true,
                  showInfobox: false
                });
                if (p && p.label) {
                  p.label.setAttribute({visible: false});
                  p.on('over', function () { p.label.setAttribute({visible: true}); });
                  p.on('out', function () { p.label.setAttribute({visible: false}); });
                }
              };
              var queue = [];
              if (knownQueue && Array.isArray(knownQueue) && knownQueue.length > 0) {
                // Utiliser la séquence d'apparition fournie par Python
                knownQueue.forEach(function(item) {
                  if (!item || !item.group || !item.key) return;
                  var src = (item.group === 'A') ? knownPointsA : knownPointsB;
                  if (!src || !Object.prototype.hasOwnProperty.call(src, item.key)) return;
                  var coords = src[item.key];
                  if (!coords) return;
                  var color = (item.group === 'A') ? matchedColorA : matchedColorB;
                  var labelText = (item.group === 'A') ? '2' : '7';
                  queue.push({coords: coords, color: color, label: labelText});
                });
              } else {
                // Comportement par défaut : tous les A puis tous les B
                for (var label in knownPointsA) {
                  if (!Object.prototype.hasOwnProperty.call(knownPointsA, label)) continue;
                  queue.push({coords: knownPointsA[label], color: matchedColorA, label: '2'});
                }
                for (var label2 in knownPointsB) {
                  if (!Object.prototype.hasOwnProperty.call(knownPointsB, label2)) continue;
                  queue.push({coords: knownPointsB[label2], color: matchedColorB, label: '7'});
                }
              }

              function preplaceMysterePoint() {
                if (!preplaceMystere) return;
                if (!expectedPointsA || Object.keys(expectedPointsA).length === 0) return;
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
                  console.error('Error preplacing mystere point', e);
                }
              }

	              if (!knownPointsAnimate) {
	                queue.forEach(function(item) {
	                  addPoint(item.coords, item.color, item.label);
	                });
	                drawLineAndZones();
	                knownPlaced = true;
	                preplaceMysterePoint();
	                notifyKnownPointsDone();
	                if (onComplete) onComplete();
	                return;
	              }

              var delay = 300;

              queue.forEach(function(item, idx) {
                setTimeout(function() {
                  addPoint(item.coords, item.color, item.label);
                }, idx * delay);
              });

	              setTimeout(function() {
	                drawLineAndZones();
	                knownPlaced = true;

	                if (preplaceMystere) {
	                  setTimeout(function() { preplaceMysterePoint(); }, 200);
	                  setTimeout(function() { notifyKnownPointsDone(); }, 260);
	                } else {
	                  notifyKnownPointsDone();
	                }

	                if (onComplete) onComplete();
	              }, queue.length * delay + 50);
            } catch (e) { console.error('placeKnownPoints error', e); }
          }

          // Check if all expected points are visible
          function checkAllExpectedPointsVisible() {
            try {
              if (!exerciseValidation) return;
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
                placeKnownPoints();
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
                fillColor: expectedPointColor || matchedColorA,
                strokeColor: expectedPointColor || matchedColorA,
                name: hideExpectedLabels ? '' : res.label,
                withLabel: hideExpectedLabels ? false : true,
                fixed: true,
                frozen: true,
                highlight: false,
                showInfobox: false
              });
              if (hideExpectedLabels && pt.label) {
                pt.label.setAttribute({visible: false});
                pt.on('over', function () {
                  if (pt.label) pt.label.setAttribute({visible: false});
                });
                pt.on('out', function () {
                  if (pt.label) pt.label.setAttribute({visible: false});
                });
              }
              // Marquer comme point validé (non-interactif)
              pt.isMatched = true;
            } else if (res && res.group === 'B') {
              pt.setAttribute({
                fillColor: expectedPointColor || matchedColorB,
                strokeColor: expectedPointColor || matchedColorB,
                name: hideExpectedLabels ? '' : res.label,
                withLabel: hideExpectedLabels ? false : true,
                fixed: true,
                frozen: true,
                highlight: false,
                showInfobox: false
              });
              if (hideExpectedLabels && pt.label) {
                pt.label.setAttribute({visible: false});
                pt.on('over', function () {
                  if (pt.label) pt.label.setAttribute({visible: false});
                });
                pt.on('out', function () {
                  if (pt.label) pt.label.setAttribute({visible: false});
                });
              }
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
                    if (exerciseValidation) checkAllExpectedPointsVisible();
                  }
                } catch (e) { console.error('auto-delete error', e); }
              }, 2000);
            }
            if (exerciseValidation) checkAllExpectedPointsVisible();
          }

          // initialize board with requested bounding box
          // boundingbox = [left, top, right, bottom] so we need top > bottom.
          // Let expected points or known points guide the bounding box
          var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
          // Try to adapt bounds to expected points (or known points if no expected points)
          try {
              var allPts = [];
              normalizePoints(expectedPointsA).forEach(function(p){ allPts.push(p); });
              normalizePoints(expectedPointsB).forEach(function(p){ allPts.push(p); });
              // If there are no expected points (ex: display-only with preplaced known points),
              // use known points for computing bounds, but robustly to avoid a single outlier
              // stretching the axes too much.
              if (allPts.length === 0) {
                  normalizePoints(knownPointsA).forEach(function(p){ allPts.push(p); });
                  normalizePoints(knownPointsB).forEach(function(p){ allPts.push(p); });
              }

              function percentile(arr, q) {
                  if (!arr || arr.length === 0) return null;
                  var a = arr.slice().sort(function(x, y){ return x - y; });
                  var idx = (a.length - 1) * q;
                  var lo = Math.floor(idx);
                  var hi = Math.ceil(idx);
                  if (lo === hi) return a[lo];
                  var h = idx - lo;
                  return a[lo] * (1 - h) + a[hi] * h;
              }

              if (allPts.length > 0) {
                  var xs = allPts.map(function(p){ return p[0]; }).filter(function(v){ return typeof v === 'number' && isFinite(v); });
                  var ys = allPts.map(function(p){ return p[1]; }).filter(function(v){ return typeof v === 'number' && isFinite(v); });
                  var maxX = percentile(xs, 0.98);
                  var maxY = percentile(ys, 0.98);
                  var margin = 20;
                  xmin = 0;
                  ymin = 0;
                  if (typeof maxX === 'number' && isFinite(maxX)) xmax = Math.max(xmax, Math.ceil(maxX + margin));
                  if (typeof maxY === 'number' && isFinite(maxY)) ymax = Math.max(ymax, Math.ceil(maxY + margin));
              }
          } catch (e) { /* fallback to defaults */ }
          var xaxisdisplacement = 10;
          
          var board = JXG.JSXGraph.initBoard('box', {
            boundingbox: [xmin - xaxisdisplacement, ymax, xmax, ymin - xaxisdisplacement],
            axis: false,
            showNavigation: false,
            keepaspectratio: keepAspectRatio,
            showCopyright: false,
          });

          var MAJOR = 10;
          var MINOR_COUNT = 1;
          var GRID_STEP = MAJOR / (MINOR_COUNT + 1);

          var xAxis = board.create('axis', [[0,0],[1,0]], {
            name: '', withLabel: false,
            ticks: {
              insertTicks: false,
              ticksDistance: MAJOR,
              minorTicks: MINOR_COUNT,
              minorHeight: -1,
              majorHeight: -1,
              drawZero: true,
              drawLabels: true,
              label: { offset: [-9, -8], anchorX: 'top' }
            }
          });
          if (xAxis && xAxis.ticks && xAxis.ticks.labels && xAxis.ticks.labels.length > 0) {
            xAxis.ticks.labels[0].setText('');
          }

          var yAxis = board.create('axis', [[0,0],[0,1]], {
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
	          
	          // Placer les points connus au démarrage si demandé
	          if (preplaceKnownPoints) {
	            setTimeout(function() {
	              placeKnownPoints();
	            }, 100);
	          }

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
          if (interactive) board.on('down', function(evt) {
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
          if (interactive) board.on('move', function(evt) {
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
    page = page.replace("INSTANCE_ID_PLACEHOLDER", json.dumps(instance_id))
    page = page.replace("PANEL_H_PX", f"{panel_h}px")
    page = page.replace("LEFT_W_PX", f"{left_w_px}px")
    page = page.replace("HUMAN_TITLE_PLACEHOLDER", html_title)
    page = page.replace("IFRAME_TITLE_PLACEHOLDER", html_title)
    page = page.replace("EXPECTED_A_PLACEHOLDER", expected_json_a)
    page = page.replace("EXPECTED_B_PLACEHOLDER", expected_json_b)
    page = page.replace("KNOWN_A_PLACEHOLDER", known_json_a)
    page = page.replace("KNOWN_B_PLACEHOLDER", known_json_b)
    page = page.replace("LINE_PARAMS_PLACEHOLDER", line_json)
    page = page.replace("SHOW_ZONES_PLACEHOLDER", "true" if show_zones else "false")
    page = page.replace("ZONE_COLORS_PLACEHOLDER", zones_json)
    page = page.replace("HIDE_EXPECTED_LABELS_PLACEHOLDER", "true" if hide_expected_labels else "false")
    page = page.replace("EXPECTED_POINT_COLOR_PLACEHOLDER", expected_color_json)
    page = page.replace("KEEP_ASPECT_RATIO_PLACEHOLDER", "true" if keep_aspect_ratio else "false")
    page = page.replace("PREPLACE_A_PLACEHOLDER", "true" if preplace_points_a else "false")
    page = page.replace("MATCH_TOL_PLACEHOLDER", str(match_tolerance))
    page = page.replace("CLICK_DELETE_TOL_PLACEHOLDER", str(click_delete_tolerance))
    page = page.replace("DEFAULT_COLOR_PLACEHOLDER", default_color)
    page = page.replace("MATCHED_COLOR_A_PLACEHOLDER", matched_color_a)
    page = page.replace("MATCHED_COLOR_B_PLACEHOLDER", matched_color_b)
    page = page.replace("KNOWN_QUEUE_PLACEHOLDER", known_queue_json)
    page = page.replace("IMAGES_PLACEHOLDER", images_json)
    page = page.replace("IMAGE_CAPTION_PLACEHOLDER", json.dumps(image_caption_html or ""))
    page = page.replace("INJECTED_IMG_CSS", img_css_rules)
    page = page.replace("SHOW_LEGEND_PLACEHOLDER", "true" if show_legend else "false")
    page = page.replace("FORCE_ORIGIN_PLACEHOLDER", "true" if force_origin else "false")
    page = page.replace("PREPLACE_KNOWN_POINTS", "true" if preplace_known_points else "false")
    page = page.replace("PREPLACE_MYSTERE_PLACEHOLDER", "true" if preplace_mystere else "false")
    page = page.replace("INTERACTIVE_PLACEHOLDER", "true" if interactive else "false")
    page = page.replace("KNOWN_POINTS_ANIMATE_PLACEHOLDER", "true" if known_points_animate else "false")
    page = page.replace("EXERCISE_VALIDATION_PLACEHOLDER", "true" if exercise_validation else "false")
    page = page.replace(
        "HIDE_LEFT_PANEL_CSS",
        ".left { display:none !important; }\n      .container { gap:0;justify-content:center; }  .right { width:513px !important; height:333px !important; flex:none; }\n" if hide_left_panel else ""
    )

    page_bytes = page.encode('utf-8')
    page_b64 = base64.b64encode(page_bytes).decode('ascii')
    data_uri = f"data:text/html;base64,{page_b64}"
    iframe_html = f"""
    <div id="{instance_id}-wrapper" style="display:flex; flex-direction:column; align-items:center; gap:8px;">
	      <div id="{instance_id}-status" style="text-align:center; font-weight:bold; min-height:1.5rem;{'' if show_status else 'display:none; min-height:0;'}"></div>
      <iframe id="{instance_id}-jsxframe" src="{data_uri}" style="width:{width}px; height:{iframe_height}px; border:none;" sandbox="allow-scripts allow-same-origin"></iframe>
    </div>
    """

    listener_script = f"""
    (function(){{
	      const instanceId = {json.dumps(instance_id)};
      const statusEl = document.getElementById(instanceId + '-status');
	      const showStatus = {json.dumps(bool(show_status))};
	      const checkpointEnabled = {json.dumps(bool(checkpoint_enabled))};
	      const exerciseValidation = {json.dumps(bool(exercise_validation))};
	      const autoPassOnKnownPointsDone = {json.dumps(bool(auto_pass_on_known_points_done))};

      const checkpointPayload = {checkpoint_payload_json};
      const checkpointId = (checkpointEnabled && window.mathadata && window.mathadata.checkpoints)
        ? ('placer_points_' + window.mathadata.checkpoints.hash(checkpointPayload))
        : null;

      function setStatus(text, color, italic) {{
        if (!statusEl || !showStatus) return;
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
	      try {{
	        if (checkpointId && window.mathadata && window.mathadata.checkpoints && window.mathadata.checkpoints.check(checkpointId)) {{
	          setStatus('✓ Tu as déjà réussi cet exercice précédemment. Tu peux continuer.', 'green', true);
	          alreadyPassed = true;
	          tryRunPython("set_exercice_droite_carac_ok()");
	          tryPassBreakpoint();
	        }}
	      }} catch (e) {{
	        console.warn('Checkpoint check failed, continuing without checkpoint:', e);
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
	          }} else if (exerciseValidation && data && data.type === 'all_points_matched') {{
	            if (data.status === true) {{
	              if (!alreadyPassed) {{
	                setStatus('Bravo ! Tu as bien placé les points. Tu peux continuer.', 'green', false);
	                try {{
	                  if (checkpointId && window.mathadata && window.mathadata.checkpoints) {{
	                    window.mathadata.checkpoints.save(checkpointId);
	                  }}
	                }} catch (e) {{
	                  console.warn('Checkpoint save failed:', e);
	                }}
	                alreadyPassed = true;
	              }}
	              // Toujours tenter de passer le breakpoint (Basthon/Capytale séquencé)
	              tryPassBreakpoint();
	              tryRunPython("set_exercice_droite_carac_ok()");
	            }} else if (data.status === false) {{
	              tryRunPython("reset_exercice_droite_carac()");
	            }}
	          }}
	          else if (autoPassOnKnownPointsDone && data && data.type === 'known_points_done' && data.instance_id === instanceId) {{
	            tryPassBreakpoint();
	          }}
	        }} catch (err) {{ console.error('Listener error', err); }}
	      }}, false);
	    }})();
	    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consider using IPython.display.IFrame instead")
        display(HTML(iframe_html))
        run_js(listener_script)


def placer_mystere(html_title="Place le point C", images=None, expected_point=None,
                   known_points_a=None, known_points_b=None,
                   line_params=None, show_zones=False,
                   image_caption_html=None, show_legend=False, force_origin=False, preplace_known_points=True,
                    preplace_mystere=False, interactive=True, known_points_animate=True, hide_left_panel=False,
                    checkpoint_enabled=True, show_status=True, exercise_validation=True,
                    auto_pass_on_known_points_done=False):
    if expected_point is None:
        expected_point = [0, 0]
    if known_points_a is None:
        known_points_a = {}
    if known_points_b is None:
        known_points_b = {}

    placer_caracteristiques(
        html_title=html_title,
        images=images,
        expected_points_a={'C': [expected_point[0], expected_point[1]]},
        expected_points_b={},
        known_points_a=known_points_a,
        known_points_b=known_points_b,
        line_params=line_params,
        show_zones=show_zones,
        zone_colors={"above": "rgba(76,110,245,0.15)", "below": "rgba(246,200,95,0.15)"},
        hide_expected_labels=True,
        expected_point_color="red",
        keep_aspect_ratio=False,
        preplace_points_a=False,
        image_caption_html=image_caption_html,
        show_legend=show_legend,
        force_origin=force_origin,
        preplace_known_points=preplace_known_points,
        preplace_mystere=preplace_mystere,
        interactive=interactive,
        known_points_animate=known_points_animate,
        hide_left_panel=hide_left_panel,
        checkpoint_enabled=checkpoint_enabled,
        show_status=show_status,
        exercise_validation=exercise_validation,
        auto_pass_on_known_points_done=auto_pass_on_known_points_done,
    )


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
    reset_exercice_droite_carac()

    # Utiliser les zones exo au lieu de zones hardcodées 
    zone_1 = getattr(common.challenge, "zone_1_exo", None)
    zone_2 = getattr(common.challenge, "zone_2_exo", None)
    
    if zone_1 is None or zone_2 is None:
        print_error("Les zones zone_1_exo et zone_2_exo doivent être définies dans le challenge.")
        return
    
    zones = [zone_1, zone_2]

    # Choix des images
    ids_images_ref = getattr(common.challenge, "ids_images_ref", None)
    if ids_images_ref is not None:
        try:
            ids_images_ref = tuple(ids_images_ref)
        except Exception:
            ids_images_ref = None

    if ids_images_ref and len(ids_images_ref) >= 2:
        id1 = int(ids_images_ref[0])
        id2 = int(ids_images_ref[1])
    else:
        # Fallback : une de chaque classe
        try:
            r = common.challenge.r_train
            classes = common.challenge.classes
            id1 = int(np.where(r == classes[0])[0][0])
            id2 = int(np.where(r == classes[1])[0][0])
        except Exception:
            id1 = 0
            id2 = 1

    # Calcul des points caractéristiques (sur image seuillée 0/200/250) 
    def _seuillage_0_200_250(img):
        a = np.asarray(img)
        a = np.clip(a, 0, 255)
        out = np.empty_like(a, dtype=np.uint8)
        out[a < 180] = 0
        out[(a >= 180) & (a < 220)] = 200
        # Exception (dev) : conserver 240 tel quel (ne pas le ramener à 250)
        out[(a >= 220) & (a <= 239)] = 250
        out[a == 240] = 240
        out[a > 240] = 250
        return out

    def _moyenne_zone(img, zone):
        if zone is None:
            return 0.0
        A, B = zone
        r0, c0 = int(A[0]), int(A[1])
        r1, c1 = int(B[0]), int(B[1])
        rmin, rmax = min(r0, r1), max(r0, r1)
        cmin, cmax = min(c0, c1), max(c0, c1)
        return float(np.mean(img[rmin:rmax + 1, cmin:cmax + 1]))

    img_a = common.challenge.d_train[id1]
    img_b = common.challenge.d_train[id2]
    img_a_t = _seuillage_0_200_250(img_a)
    img_b_t = _seuillage_0_200_250(img_b)

    x_a = _moyenne_zone(img_a_t, zones[0])
    y_a = _moyenne_zone(img_a_t, zones[1])
    x_b = _moyenne_zone(img_b_t, zones[0])
    y_b = _moyenne_zone(img_b_t, zones[1])

    expected_points_a = {"A": [x_a, y_a]}
    expected_points_b = {"B": [x_b, y_b]}

    # --- Layout : widget MNIST (gauche) + plan JSXGraph (droite) ---
    layout_id = uuid.uuid4().hex
    widget_id = f"{layout_id}-mnist"
    board_id = f"{layout_id}-board"
    status_id = f"{layout_id}-status"

    display(HTML(f"""
    <div id="{layout_id}" class="mathadata-placer2points-layout">
      <div id="{layout_id}-left" class="mathadata-placer2points-left">
        <div id="{widget_id}" class="mathadata-mnist-exemples-zones"></div>
      </div>
      <div id="{layout_id}-right" class="mathadata-placer2points-right">
        <div class="title" style="width:100%; text-align:center; font-size:16px; font-weight:700; padding:6px 0;">Placer les 2 points</div>
        <div id="{status_id}" style="text-align:center; font-weight:bold; min-height:1.5rem; margin-bottom:8px;"></div>
        <div id="{board_id}" class="jxgbox" style="width:600px; height:434px; border:1px solid #e6e6e6; border-radius:6px; box-sizing:border-box;"></div>
      </div>
    </div>
    """))

    run_js(f"""
    (function(){{
      const styleId = "mathadata-style-placer2points";
      const existing = document.getElementById(styleId);
      if (existing) existing.remove();

      const s = document.createElement("style");
      s.id = styleId;
      s.textContent = `
        .mathadata-placer2points-layout {{
          display: flex;
          gap: 12px;
          align-items: flex-start;
          justify-content: center;
          flex-wrap: wrap;
          width: 100%;
          margin: 0.75rem 0;
        }}
        .mathadata-placer2points-left {{
          flex: 0 0 240px;
          width: 240px;
          max-width: 240px;
        }}
        .mathadata-placer2points-right {{
          flex: 1 1 auto;
          min-width: 600px;
          display: flex;
          flex-direction: column;
          align-items: stretch;
        }}

        /* Version compacte du widget MNIST dans ce layout */
        .mathadata-placer2points-left .mathadata-mnist-exemples-zones-grid {{
          gap: 0.75rem;
        }}
        .mathadata-placer2points-left .mathadata-mnist-exemples-zones-item {{
          max-width: 120px;
        }}
        .mathadata-placer2points-left .mathadata-mnist-exemples-zones-point {{
          font-size: 1.4rem;
          line-height: 1.1;
          overflow-wrap: anywhere;
          word-break: break-word;
        }}

        .mathadata-placer2points-right .title {{
          width: 100%;
          text-align: center;
          font-size: 16px;
          font-weight: 700;
          padding: 6px 0;
        }}
      `;
      document.head.appendChild(s);
    }})(); 
    """)

    # Rendu du widget avec les zones exo
    params_widget = {
        "ids": [id1, id2],
        "zones": zones,  # Utilise zone_1_exo et zone_2_exo
        "show_points": True,
        "show_zones": True,
        "fontsize": 4,
        "grid_gap_px": 0,
        "max_width_px": 120,
        "show_values": True,
        "zone_lw": 5,
        "point_names": ["A", "B"],
    }
    params_widget_json = json.dumps(params_widget, cls=NpEncoder)
    run_js(f"""
    mathadata.add_observer('{widget_id}', () => {{
      const el = document.getElementById('{widget_id}');
      if (!window.mathadata || typeof window.mathadata.afficher_deux_exemples_zones !== "function") {{
        if (el) {{
          el.innerHTML = "<div style=\\"font-family:sans-serif;color:#b00020;\\">Erreur : afficher_deux_exemples_zones n'est pas disponible. As-tu bien importé le challenge MNIST ?</div>";
        }}
        return;
      }}
      window.mathadata.afficher_deux_exemples_zones('{widget_id}', '{params_widget_json}');
    }});
    """)

    # Initialisation JSXGraph directement dans le DOM (sans iframe)
    expected_json_a = json.dumps(expected_points_a, sort_keys=True)
    expected_json_b = json.dumps(expected_points_b, sort_keys=True)
    
    checkpoint_payload = {
        "type": "placer_caracteristiques",
        "title": "De l'image au plan",
        "expected_points_a": expected_points_a,
        "expected_points_b": expected_points_b,
    }
    checkpoint_payload_json = json.dumps(checkpoint_payload, sort_keys=True, ensure_ascii=False)
    
    run_js(f"""
    (function(){{
      // Capytale/JupyterLite: le JS peut s'exécuter AVANT que l'output HTML soit inséré.
      // On utilise la même stratégie que les gros widgets: `mathadata.add_observer(...)`.
      const boardId = {json.dumps(board_id)};

      function boot() {{
        const boardContainer = document.getElementById(boardId);
        if (!boardContainer) {{
          console.error('Container JSXGraph introuvable:', boardId);
          return;
        }}
        // anti double-init (si observer rappelé)
        if (boardContainer.dataset && boardContainer.dataset.jxgInit === '1') return;
        if (boardContainer.dataset) boardContainer.dataset.jxgInit = '1';

        // Charger JSXGraph si pas déjà chargé
        if (typeof JXG === 'undefined') {{
          const link = document.createElement('link');
          link.rel = 'stylesheet';
          link.href = 'https://jsxgraph.org/distrib/jsxgraph.css';
          document.head.appendChild(link);
          
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jsxgraph/1.4.0/jsxgraphcore.js';
          script.onload = function() {{
            initJSXGraph();
          }};
          document.head.appendChild(script);
        }} else {{
          initJSXGraph();
        }}
      }}

      if (window.mathadata && typeof window.mathadata.add_observer === 'function') {{
        window.mathadata.add_observer(boardId, boot);
      }} else {{
        // Fallback si `mathadata.add_observer` indisponible: retry court
        (function retry(n) {{
          if (document.getElementById(boardId)) return boot();
          if (n <= 0) return console.error('Container JSXGraph introuvable (retry épuisé):', boardId);
          requestAnimationFrame(() => retry(n - 1));
        }})(120);
      }}
      
      function initJSXGraph() {{
        const statusId = {json.dumps(status_id)};
        const instanceId = {json.dumps(layout_id)};
        
        const statusEl = document.getElementById(statusId);
        const boardContainer = document.getElementById(boardId);
        
        if (!boardContainer) {{
          console.error('Container JSXGraph introuvable:', boardId);
          return;
        }}
        
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
        
        const checkpointPayload = {checkpoint_payload_json};
        const checkpointId = (window.mathadata && window.mathadata.checkpoints)
          ? ('placer_points_' + window.mathadata.checkpoints.hash(checkpointPayload))
          : null;
        
        let alreadyPassed = false;
        
        if (checkpointId && window.mathadata && window.mathadata.checkpoints && window.mathadata.checkpoints.check(checkpointId)) {{
          setStatus('✓ Tu as déjà réussi cet exercice précédemment. Tu peux continuer.', 'green', true);
          alreadyPassed = true;
          tryRunPython("set_exercice_droite_carac_ok()");
          tryPassBreakpoint();
        }}
        
        // Variables JSXGraph
        var expectedPointsA = {expected_json_a};
        var expectedPointsB = {expected_json_b};
        var preplaceGroupA = false;
        var matchTol = 3.0;
        var clickDeleteTol = 0.45;
        var defaultColor = "#8c1fb4";
        var matchedColorA = "#4C6EF5";
        var matchedColorB = "#F6C85F";
        var keepAspectRatio = false;
        
        // Calculer les bornes du graphique à partir des points attendus
        var xmin = 0, xmax = 165, ymin = 0, ymax = 160;
        try {{
          var allPts = [];
          Object.values(expectedPointsA).forEach(function(p){{ allPts.push(p); }});
          Object.values(expectedPointsB).forEach(function(p){{ allPts.push(p); }});
          if (allPts.length > 0) {{
            var xs = allPts.map(function(p){{ return p[0]; }});
            var ys = allPts.map(function(p){{ return p[1]; }});
            var minX = Math.min.apply(null, xs);
            var maxX = Math.max.apply(null, xs);
            var minY = Math.min.apply(null, ys);
            var maxY = Math.max.apply(null, ys);
            var margin = 20;
            xmin = Math.max(0, Math.floor(minX - margin));
            ymin = Math.max(0, Math.floor(minY - margin));
            xmax = Math.ceil(maxX + margin);
            ymax = Math.ceil(maxY + margin);
          }}
        }} catch (e) {{ console.error('Error calculating bounds:', e); }}
        
        var xaxisdisplacement = 10;
        var yaxisdisplacement = 5;
        var MAJOR = 10;
        var MINOR_COUNT = 1;
        var GRID_STEP = MAJOR / (MINOR_COUNT + 1);
        
        // Initialiser le board JSXGraph
        var board = JXG.JSXGraph.initBoard(boardId, {{
          boundingbox: [xmin - xaxisdisplacement, ymax, xmax, ymin- yaxisdisplacement],
          axis: false,
          showNavigation: false,
          keepaspectratio: keepAspectRatio,
          showCopyright: false,
        }});
        
        var xAxis = board.create('axis',
          [[0, 0], [1, 0]],
        {{
          withLabel: false,
          ticks: {{
            insertTicks: false,
            ticksDistance: MAJOR,
            minorTicks: MINOR_COUNT,
            minorHeight: -1,
            majorHeight: -1,
            drawZero: true,
            drawLabels: true,
            label: {{ offset: [-9, -8], anchorX: 'top' }}
          }}
        }});
        
        var yAxis = board.create('axis',
          [[0, 0], [0, 1]],
        {{
          withLabel: false,
          ticks: {{
            insertTicks: false,
            ticksDistance: MAJOR,
            minorTicks: MINOR_COUNT,
            drawZero: false,
            minorHeight: -1,
            majorHeight: -1,
            drawLabels: true,
            label: {{ offset: [-2, 0], anchorX: 'right' }}
          }}
        }});
        
        var userPoints = [];
        
        // Fonction pour vérifier si les coordonnées correspondent à un point attendu
        function whichMatchCoords(x, y) {{
          try {{
            for (var label in expectedPointsA) {{
              if (!Object.prototype.hasOwnProperty.call(expectedPointsA, label)) continue;
              var ex = expectedPointsA[label][0], ey = expectedPointsA[label][1];
              if (Math.abs(x - ex) <= matchTol && Math.abs(y - ey) <= matchTol) {{
                return {{group: 'A', label: label}};
              }}
            }}
            for (var label2 in expectedPointsB) {{
              if (!Object.prototype.hasOwnProperty.call(expectedPointsB, label2)) continue;
              var bx = expectedPointsB[label2][0], by = expectedPointsB[label2][1];
              if (Math.abs(x - bx) <= matchTol && Math.abs(y - by) <= matchTol) {{
                return {{group: 'B', label: label2}};
              }}
            }}
          }} catch (e) {{
            console.error('whichMatchCoords error', e);
          }}
          return null;
        }}
        
        // Fonction pour envoyer les points à Python
        function sendPointsToPython() {{
          try {{
            var pts = [];
            userPoints.forEach(function(pt) {{
              if (board.objectsList.indexOf(pt) !== -1) {{
                pts.push([ +pt.X().toFixed(6), +pt.Y().toFixed(6) ]);
              }}
            }});
            var code = "jsx_points = " + JSON.stringify(pts) + "\\n";
            // même logique que le reste: privilégier `mathadata.run_python` (Capytale/JupyterLite)
            tryRunPython(code);
          }} catch (err) {{
            console.error('sendPointsToPython error', err);
          }}
        }}
        
        // Fonction pour vérifier si tous les points attendus sont placés
        function checkAllExpectedPointsVisible() {{
          try {{
            var expectedLabelsA = Object.keys(expectedPointsA);
            var expectedLabelsB = Object.keys(expectedPointsB);
            var allExpectedLabels = expectedLabelsA.concat(expectedLabelsB);
            
            var matchedLabels = {{}};
            userPoints.forEach(function(pt) {{
              if (board.objectsList.indexOf(pt) === -1) return;
              var x = pt.X(), y = pt.Y();
              var res = whichMatchCoords(x, y);
              if (res) {{
                matchedLabels[res.label] = true;
              }}
            }});
            
            var allVisible = allExpectedLabels.every(function(label) {{
              return matchedLabels[label] === true;
            }});
            
            console.log('Expected labels:', allExpectedLabels);
            console.log('Matched labels:', Object.keys(matchedLabels));
            console.log('All visible:', allVisible);
            
            if (allVisible) {{
              console.log('All expected points are correctly placed!');
              if (!alreadyPassed) {{
                setStatus('Bravo ! Tu as bien placé les points. Tu peux continuer.', 'green', false);
                if (checkpointId && window.mathadata && window.mathadata.checkpoints) {{
                  window.mathadata.checkpoints.save(checkpointId);
                }}
                alreadyPassed = true;
                tryPassBreakpoint();
              }}
              tryRunPython("set_exercice_droite_carac_ok()");
            }} else {{
              tryRunPython("reset_exercice_droite_carac()");
            }}
          }} catch (e) {{
            console.error('checkAllExpectedPointsVisible error', e);
          }}
        }}
        
        // Fonction pour mettre à jour la couleur et le label d'un point
        function updatePointColorAndLabel(pt) {{
          var x = pt.X(), y = pt.Y();
          var res = whichMatchCoords(x, y);
          if (res && res.group === 'A') {{
            pt.setAttribute({{
              fillColor: matchedColorA,
              strokeColor: matchedColorA,
              name: res.label,
              withLabel: true,
              fixed: true,
              frozen: true,
              highlight: false,
              showInfobox: false
            }});
            pt.isMatched = true;
          }} else if (res && res.group === 'B') {{
            pt.setAttribute({{
              fillColor: matchedColorB,
              strokeColor: matchedColorB,
              name: res.label,
              withLabel: true,
              fixed: true,
              frozen: true,
              highlight: false,
              showInfobox: false
            }});
            pt.isMatched = true;
          }} else {{
            pt.setAttribute({{
              fillColor: defaultColor,
              strokeColor: defaultColor,
              name: 'Mauvais point',
              withLabel: true,
              fixed: false,
              frozen: false
            }});
            pt.isMatched = false;
            setTimeout(function() {{
              try {{
                var idx = userPoints.indexOf(pt);
                if (idx !== -1) {{
                  if (board.objectsList.indexOf(pt) !== -1) {{
                    board.removeObject(pt);
                  }}
                  userPoints.splice(idx, 1);
                  sendPointsToPython();
                  checkAllExpectedPointsVisible();
                }}
              }} catch (e) {{ console.error('auto-delete error', e); }}
            }}, 2000);
          }}
          checkAllExpectedPointsVisible();
        }}
        
        // Fonction pour trouver un point proche
        function dist(a, b) {{
          var dx = a[0] - b[0];
          var dy = a[1] - b[1];
          return Math.sqrt(dx*dx + dy*dy);
        }}
        
        function findNearbyUserPointIndex(coords) {{
          for (var i = 0; i < userPoints.length; i++) {{
            var pt = userPoints[i];
            if (board.objectsList.indexOf(pt) === -1) continue;
            var pcoords = [pt.X(), pt.Y()];
            if (dist(coords, pcoords) <= clickDeleteTol) return i;
          }}
          return -1;
        }}
        
        // Gestion des clics sur le board
        board.on('down', function(evt) {{
          try {{
            var raw = board.getUsrCoordsOfMouse(evt);
            var nearbyIdx = findNearbyUserPointIndex(raw);
            if (nearbyIdx !== -1) {{
              var p = userPoints[nearbyIdx];
              if (p && p.isMatched) {{
                return;
              }}
              if (board.objectsList.indexOf(p) !== -1) board.removeObject(p);
              userPoints.splice(nearbyIdx, 1);
              sendPointsToPython();
              return;
            }}
            
            var snapped = (function(coords) {{
              var x = Math.round(coords[0]);
              var y = Math.round(coords[1]);
              var bb = board.getBoundingBox();
              var xmin = bb[0], ymax = bb[1], xmax = bb[2], ymin = bb[3];
              if (x < xmin) x = xmin;
              if (x > xmax) x = xmax;
              if (y < ymin) y = ymin;
              if (y > ymax) y = ymax;
              return [x, y];
            }})(raw);
            
            var p = board.create('point', snapped, {{
              withLabel: false, size: 6, name: '',
              snapToGrid: true, snapSizeX: GRID_STEP, snapSizeY: GRID_STEP
            }});
            if (typeof p.snapToGrid === 'function') p.snapToGrid(true);
            
            userPoints.push(p);
            updatePointColorAndLabel(p);
            
            p.on('drag', function() {{
              try {{
                if (this.isMatched) {{ return; }}
                var s = (function(coords) {{
                  var x = Math.round(coords[0]);
                  var y = Math.round(coords[1]);
                  var bb = board.getBoundingBox();
                  var xmin = bb[0], ymax = bb[1], xmax = bb[2], ymin = bb[3];
                  if (x < xmin) x = xmin;
                  if (x > xmax) x = xmax;
                  if (y < ymin) y = ymin;
                  if (y > ymax) y = ymax;
                  return [x, y];
                }})([this.X(), this.Y()]);
                this.moveTo(JXG.COORDS_BY_USER, s);
                updatePointColorAndLabel(this);
                sendPointsToPython();
              }} catch (e) {{ console.error('drag error', e); }}
            }});
            
            p.on('up', function() {{
              try {{
                if (this.isMatched) {{ return; }}
                var s = (function(coords) {{
                  var x = Math.round(coords[0]);
                  var y = Math.round(coords[1]);
                  var bb = board.getBoundingBox();
                  var xmin = bb[0], ymax = bb[1], xmax = bb[2], ymin = bb[3];
                  if (x < xmin) x = xmin;
                  if (x > xmax) x = xmax;
                  if (y < ymin) y = ymin;
                  if (y > ymax) y = ymax;
                  return [x, y];
                }})([this.X(), this.Y()]);
                this.moveTo(JXG.COORDS_BY_USER, s);
                updatePointColorAndLabel(this);
                sendPointsToPython();
              }} catch (e) {{ console.error('up error', e); }}
            }});
            
            sendPointsToPython();
          }} catch (err) {{ console.error('Error handling down event', err); }}
        }});
        
        board.on('move', function(evt) {{
          if (board.hasMouseDown) sendPointsToPython();
        }});
        
        boardContainer.tabIndex = 0;
        console.log('JSXGraph initialisé avec succès dans le conteneur direct');
      }}
    }})();
    """)


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
                                 sliders=False, interception_point=True, equation_hide=True, orthonormal=False,
                                 center_canvas=False, hide_inputs=False, vector_inset=False):
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
        'vector_inset': vector_inset,
        'vecteurs': {
            'directeur': directeur,
            'directeur_a': directeur_a,
            'normal': normal,
        },
        'center_canvas': center_canvas,
        'orthonormal': orthonormal,
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
    xa = 30
    ya = 70
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

                <!-- Conteneur flex horizontal pour canvas + sliders -->
                <div style="display:flex; flex-direction:row; gap:2rem; align-items:flex-start;">
                    
                    <!-- Canvas à gauche -->
                    <div style="flex: 1;">
                        <canvas id="{id_content}-chart"></canvas>
                    </div>

                    <!-- Sliders à droite -->
                    <div id="{id_content}-inputs"
                         style="display:{'none' if hide_inputs else 'flex'}; flex-direction:column; gap:2rem; min-width:250px;">
                    
                        <!-- Sliders pour vecteur directeur -->
                        <div style="
                            display:{'flex' if (directeur and not reglage_normal) else 'none'};
                            flex-direction:column; gap:1.5rem;">
                            <div>
                                <label style="color: green;">x<sub>u</sub> =
                                    <span id="{id_content}-ux-val">{ux}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-ux"
                                       value="{ux}" min="-20" max="20" step="0.1"
                                       oninput="document.getElementById('{id_content}-ux-val').textContent=this.value;"
                                       style="width:100%;">
                            </div>
                            <div>
                                <label style="color: firebrick;">y<sub>u</sub> =
                                    <span id="{id_content}-uy-val">{uy}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-uy"
                                       value="{uy}" min="-20" max="20" step="0.1"
                                       oninput="document.getElementById('{id_content}-uy-val').textContent=this.value;"
                                       style="width:100%;">
                            </div>
                        </div>
                    
                        <!-- Sliders pour vecteur normal -->
                        <div style="
                            display:{'flex' if reglage_normal else 'none'};
                            flex-direction:column; gap:1.5rem;">
                            <div>
                                <label>⃗n<sub>x</sub> =
                                    <span id="{id_content}-nx-val">{nx}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-nx"
                                       value="{nx}" min="-10" max="10" step="1"
                                       oninput="document.getElementById('{id_content}-nx-val').textContent=this.value;"
                                       style="width:100%;">
                            </div>
                            <div>
                                <label>⃗n<sub>y</sub> =
                                    <span id="{id_content}-ny-val">{ny}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-ny"
                                       value="{ny}" min="-10" max="10" step="1"
                                       oninput="document.getElementById('{id_content}-ny-val').textContent=this.value;"
                                       style="width:100%;">
                            </div>
                        </div>
                    
                        <!-- Sliders pour point A -->
                        <div style="display:flex; flex-direction:column; gap:1.5rem;">
                            <div>
                                <label>x<sub>A</sub> =
                                    <span id="{id_content}-xa-val">{xa}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-xa"
                                       value="{xa}" min="0" max="88" step="1"
                                       oninput="document.getElementById('{id_content}-xa-val').textContent=this.value;"
                                       style="width:100%;">
                            </div>
                            <div>
                                <label>y<sub>A</sub> =
                                    <span id="{id_content}-ya-val">{ya}</span>
                                </label>
                                <input type="range"
                                       id="{id_content}-input-ya"
                                       value="{ya}" min="0" max="88" step="1"
                                       oninput="document.getElementById('{id_content}-ya-val').textContent=this.value;"
                                       style="width:100%;">
                            </div>
                        </div>
                    
                    </div>

                </div>
            </div>
            '''))
    else:
        display(HTML(f'''
        <!-- Conteneur pour afficher le pourcentage d'erreur -->
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
                display: {'none' if hide_inputs else 'flex'};
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
    
# fonction particulière pour nb prod scal mnist
def tracer_points_droite_vecteur_directeur_nb_ps():
    # Valeurs par défaut spécifiques pour cet exercice
    defaults = {'ux': 5, 'uy': 10, 'xa': 20, 'ya': 10}

    tracer_points_droite_vecteur(directeur=True, directeur_a=True, initial_values=defaults, sliders=True,
                 interception_point=False, equation_hide=False, orthonormal=True, center_canvas=True)
    # fonction particulière pour nb prod scal mnist
def tracer_points_droite_vecteur_directeur_nb_ps_rappel():

    tracer_points_droite_vecteur(directeur=True, directeur_a=True, sliders=True,
                 interception_point=False, equation_hide=False, orthonormal=True, center_canvas=True, hide_inputs=True)

# fonction particulière pour nb prod scal mnist
def tracer_points_droite_vecteur_normal_nb_ps():
    # Valeurs par défaut spécifiques pour cet exercice
    defaults = {'nx': 5, 'ny': 10, 'xa': 20, 'ya': 10}
    tracer_points_droite_vecteur(directeur_a=True, normal=True, initial_values=defaults, sliders=True,reglage_normal=True,
                 interception_point=False, equation_hide=False, orthonormal=True, save=False, vector_inset=True)
   

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


def erreur_lineaire(a, b, c, c_train, above=None, below=None, r_train=None):
    r_est_train = _classification(a, b, c, c_train, above, below)
    erreurs = (r_est_train != r_train).astype(int)
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

      const {points, droite, vecteurs, vector_inset = false, centroides, additionalPoints, hideClasses, hover, inputs, initial_values, displayValue, save, custom, compute_score, drag, force_origin, equation_hide, param_colors, equation_fixed_position, orthonormal, center_canvas, side_box = true , interception_point = true, disable_python_updates = false} = params;
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
              <foreignObject id="${id}-text-formula" x="0" y="125" width="250" height="55"></foreignObject>
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
            const formulaY = Math.max(125, triangleBottomY + 60); // Au moins 125, ou plus bas si nécessaire
            textFormula.setAttribute('y', formulaY);

            const mColor = param_colors?.m || '#239E28';
            // Render the formula as a fraction (keep colors).
            textFormula.innerHTML = `
              <div xmlns="http://www.w3.org/1999/xhtml"
                   style="width:250px; height:55px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:16px; line-height:1;">
                <span style="color:${mColor};">m</span>
                <span style="color:black;">&nbsp;=&nbsp;</span>
                <span style="display:inline-flex; flex-direction:column; align-items:center; margin:0 4px;">
                  <span style="color:orange; padding:0 2px;">${resDisp}</span>
                  <span style="height:2px; width:100%; background:black; margin:2px 0;"></span>
                  <span style="color:purple; padding:0 2px;">10</span>
                </span>
                <span style="color:black;">&nbsp;=&nbsp;</span>
                <span style="color:${mColor};">${mDisp}</span>
              </div>
            `;
        }
      };
      
      const plugins = [];
      // Axes lines (x=0, y=0) drawn before points
      plugins.push({
        beforeDatasetsDraw(chart) {
          const scales = chart.scales;
          if (!scales || !scales.x || !scales.y) return;
          const x0 = scales.x.getPixelForValue(0);
          const y0 = scales.y.getPixelForValue(0);
          const area = chart.chartArea;
          if (!area) return;
          const ctx = chart.ctx;
          ctx.save();
          ctx.strokeStyle = '#555';
          ctx.lineWidth = 2;
          if (y0 >= area.top && y0 <= area.bottom) {
            ctx.beginPath();
            ctx.moveTo(area.left, y0);
            ctx.lineTo(area.right, y0);
            ctx.stroke();
          }
          if (x0 >= area.left && x0 <= area.right) {
            ctx.beginPath();
            ctx.moveTo(x0, area.top);
            ctx.lineTo(x0, area.bottom);
            ctx.stroke();
          }
          ctx.restore();
        }
      });

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
      let minX, maxX, minY, maxY;
      let start_ux, start_uy;
      let droiteDatasetIndex;
      let yInterceptDatasetIndex;
      let centroid1DatasetIndex, centroid2DatasetIndex;

        // when caracteristique changes
      const updatePoints = (points, params) => {
        if (points) {
            const allData = points.flat(2);
            const xs = points.flat(1).map(p => p[0]);
            const ys = points.flat(1).map(p => p[1]);
            maxX = Math.ceil(Math.max(...xs) + 1);
            minX = Math.floor(Math.min(...xs) - 1);
            maxY = Math.ceil(Math.max(...ys) + 1);
            minY = Math.floor(Math.min(...ys) - 1);
            if (force_origin) {
              minX = Math.min(minX, 0);
              minY = Math.min(minY, 0);
              maxX = Math.max(maxX, 0);
              maxY = Math.max(maxY, 0);
            }
            max = Math.max(maxX, maxY);
            min = Math.min(minX, minY);
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
            const lineMin = minX ?? min;
            const lineMax = maxX ?? max;
            const lineData = mathadata.findIntersectionPoints(values.a, values.b, values.c, lineMin, lineMax);
            datasets[droiteDatasetIndex].data = lineData; 
            if (droite?.avec_zones) {
                const lineDataVertical = mathadata.findIntersectionPoints(values.a, values.b, values.c, lineMin, lineMax, true);  
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
          // include y-intercept in y range only
          let yMin = minY ?? min
          let yMax = maxY ?? max
          if (yInterceptDatasetIndex !== undefined) {
            const interceptData = getYInterceptPoint(values);
            if (interceptData) {
              yMin = Math.min(yMin, interceptData.y);
              yMax = Math.max(yMax, interceptData.y);
            }
          }
          chart.options.scales.x.min = minX ?? min
          chart.options.scales.x.max = maxX ?? max
          chart.options.scales.y.min = yMin
          chart.options.scales.y.max = yMax
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
                if (vector_inset) {
                    chart.$equationBBox = null;
                }
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
                        // Position le long de la droite (comportement par défaut) -- Sauf que ici, on reste dans le chartarea en dur avec ce nouvel update.
                        textX = Math.min(
                            chart.chartArea.right - 5,
                            Math.max(chart.chartArea.left + 5, x2 + 20)
                        );
                        textY = Math.min(
                            chart.chartArea.bottom - 5,
                            Math.max(chart.chartArea.top + 5, y2 + 20)
                        );
                        rotationAngle = angle;
                    }

                    let equationText;
                    if ((values.m !== undefined && values.p !== undefined) && mode !== 'cartesienne') {
                        const signEq = values.p < 0 ? '-' : '+';
                        equationText = `y = ${values.m}x ${signEq} ${Math.abs(values.p)}`;
                    } else {
                        equationText = mathadata.getLineEquationStr(values.a, values.b, values.c);
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
                            ctx.fillText(equationText, 0, 0);
                        }

                        // Stocker une bounding box de l'équation (en pixels canvas) pour permettre d'éviter les chevauchements avec d'autres éléments (ex: encart de vecteurs en bas à droite).
                        if (vector_inset) {
                            const metrics = ctx.measureText(equationText || '');
                            const width = metrics?.width ?? 0;
                            const ascent = metrics?.actualBoundingBoxAscent ?? 14;
                            const descent = metrics?.actualBoundingBoxDescent ?? 4;

                            // textAlign = 'right' et fillText(..., 0, 0) => le texte s'étend sur [-width, 0]
                            const xMin = -width;
                            const xMax = 0;
                            const yMin = -ascent;
                            const yMax = descent;

                            const cos = Math.cos(rotationAngle);
                            const sin = Math.sin(rotationAngle);
                            const pts = [
                                {x: xMin, y: yMin},
                                {x: xMax, y: yMin},
                                {x: xMax, y: yMax},
                                {x: xMin, y: yMax},
                            ].map(({x, y}) => ({
                                x: textX + x * cos - y * sin,
                                y: textY + x * sin + y * cos,
                            }));

                            const xs = pts.map(p => p.x);
                            const ys = pts.map(p => p.y);
                            const pad = 6;
                            chart.$equationBBox = {
                                left: Math.min(...xs) - pad,
                                top: Math.min(...ys) - pad,
                                right: Math.max(...xs) + pad,
                                bottom: Math.max(...ys) + pad,
                            };
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

      // Encart (inset) optionnel : répliques des vecteurs u et n en bas à droite,
      // en évitant de chevaucher l'équation de la droite.
      if (vector_inset) {
        plugins.push({
          afterDatasetsDraw: function(chart) {
            const ctx = chart.ctx;
            const area = chart.chartArea;
            if (!area) return;

            const showU = !!(vecteurs && (vecteurs.directeur || vecteurs.directeur_a));
            const showN = !!(vecteurs && vecteurs.normal);
            if (!showU && !showN) return;

            const ux = (values.ux ?? 0);
            const uy = (values.uy ?? 0);
            const nx = (values.nx ?? 0);
            const ny = (values.ny ?? 0);

            const insetW = 140;
            const insetH = 140;
            const margin = 10;

            const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
            const makeRect = (x, y) => ({left: x, top: y, right: x + insetW, bottom: y + insetH});
            const overlaps = (a, b) => !(
              a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom
            );

            const eqBBox = (!equation_hide && chart.$equationBBox) ? chart.$equationBBox : null;

            const bounds = {
              left: area.left + margin,
              top: area.top + margin,
              right: area.right - margin,
              bottom: area.bottom - margin,
            };

            const candidates = [
              // bottom-right
              {x: area.right - margin - insetW, y: area.bottom - margin - insetH},
            ];

            if (eqBBox) {
              // juste au-dessus de l'équation (aligné à droite)
              candidates.push({x: area.right - margin - insetW, y: eqBBox.top - margin - insetH});
              // juste à gauche de l'équation (aligné en bas)
              candidates.push({x: eqBBox.left - margin - insetW, y: area.bottom - margin - insetH});
            }

            // fallbacks
            candidates.push(
              {x: area.right - margin - insetW, y: area.top + margin}, // top-right
              {x: area.left + margin, y: area.bottom - margin - insetH}, // bottom-left
            );

            let rect = null;
            for (const c of candidates) {
              const x = clamp(c.x, bounds.left, bounds.right - insetW);
              const y = clamp(c.y, bounds.top, bounds.bottom - insetH);
              const r = makeRect(x, y);
              if (!eqBBox || !overlaps(r, eqBBox)) {
                rect = r;
                break;
              }
            }
            if (!rect) {
              const x = clamp(area.right - margin - insetW, bounds.left, bounds.right - insetW);
              const y = clamp(area.bottom - margin - insetH, bounds.top, bounds.bottom - insetH);
              rect = makeRect(x, y);
            }

            const roundedRect = (x, y, w, h, r) => {
              const radius = Math.max(0, Math.min(r, w / 2, h / 2));
              ctx.beginPath();
              ctx.moveTo(x + radius, y);
              ctx.lineTo(x + w - radius, y);
              ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
              ctx.lineTo(x + w, y + h - radius);
              ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
              ctx.lineTo(x + radius, y + h);
              ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
              ctx.lineTo(x, y + radius);
              ctx.quadraticCurveTo(x, y, x + radius, y);
              ctx.closePath();
            };

            const drawArrow = (x1, y1, x2, y2, color) => {
              const dx = x2 - x1;
              const dy = y2 - y1;
              const len = Math.hypot(dx, dy);
              if (!isFinite(len) || len < 1e-6) {
                // vecteur nul
                ctx.save();
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x1, y1, 3, 0, 2 * Math.PI);
                ctx.fill();
                ctx.restore();
                return;
              }

              ctx.save();
              ctx.strokeStyle = color;
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x2, y2);
              ctx.stroke();

              const headlen = 10;
              const angle = Math.atan2(dy, dx);
              ctx.beginPath();
              ctx.moveTo(x2, y2);
              ctx.lineTo(x2 - headlen * Math.cos(angle - Math.PI / 6), y2 - headlen * Math.sin(angle - Math.PI / 6));
              ctx.lineTo(x2 - headlen * Math.cos(angle + Math.PI / 6), y2 - headlen * Math.sin(angle + Math.PI / 6));
              ctx.closePath();
              ctx.fillStyle = color;
              ctx.fill();
              ctx.restore();
            };

            const vectors = [];
            if (showU) vectors.push({x: ux, y: uy, color: 'purple', label: '\u20D7u'});
            if (showN) vectors.push({x: nx, y: ny, color: 'brown', label: '\u20D7n'});

            let maxLen = 0;
            vectors.forEach(v => {
              const l = Math.hypot(v.x, v.y);
              if (isFinite(l)) maxLen = Math.max(maxLen, l);
            });
            const radiusMax = Math.min(insetW, insetH) / 2 - 22;
            const scale = maxLen > 0 ? (radiusMax / maxLen) : 0;

            const originX = rect.left + insetW / 2;
            const originY = rect.top + insetH / 2;

            ctx.save();
            // fond + bordure
            roundedRect(rect.left, rect.top, insetW, insetH, 10);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.22)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // axes (repère léger)
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(rect.left + 12, originY);
            ctx.lineTo(rect.right - 12, originY);
            ctx.moveTo(originX, rect.top + 12);
            ctx.lineTo(originX, rect.bottom - 12);
            ctx.stroke();

            // vecteurs
            vectors.forEach(v => {
              const endX = originX + v.x * scale;
              const endY = originY - v.y * scale; // y vers le haut (repère math)
              drawArrow(originX, originY, endX, endY, v.color);

              // label proche de la pointe (décalage perpendiculaire)
              const dx = endX - originX;
              const dy = endY - originY;
              const len = Math.hypot(dx, dy) || 1;
              const nxp = -dy / len;
              const nyp = dx / len;
              const labelX = endX + nxp * 10 + (dx / len) * 4;
              const labelY = endY + nyp * 10 + (dy / len) * 4;

              ctx.save();
              ctx.font = 'bold 14px Arial';
              ctx.fillStyle = v.color;
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              ctx.fillText(v.label, labelX, labelY);
              ctx.restore();
            });

            // origine
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.beginPath();
            ctx.arc(originX, originY, 2, 0, 2 * Math.PI);
            ctx.fill();

            ctx.restore();
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
                        ticks: {
                            stepSize: 10
                        },
                        grid: {
                            drawBorder: false,
                            color: 'rgba(0,0,0,0.12)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Caractéristique y'
                        },
                        min,
                        max,
                        ticks: {
                            stepSize: 10
                        },
                        grid: {
                            drawBorder: false,
                            color: 'rgba(0,0,0,0.12)'
                        }
                    }
                },
                aspectRatio: 1,
                maintainAspectRatio: true,
                orthonormal: orthonormal === true,
                centerCanvas: center_canvas === true,
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
            if (!disable_python_updates) {
                mathadata.run_python(`set_input_values('${values_json}')`)
            }
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
            if (compute_score && !disable_python_updates) {
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


def compute_score(a, b, c, custom=False, above=None, below=None, test_dataset=False):
    if custom:
        carac = common.challenge.deux_caracteristiques_custom
    else:
        carac = common.challenge.deux_caracteristiques

    d_train = common.challenge.d_train_test if test_dataset else common.challenge.d_train
    r_train = common.challenge.r_train_test if test_dataset else common.challenge.r_train
    c_train = compute_c_train(carac, d_train)
    error = erreur_lineaire(a, b, c, c_train, above, below, r_train)
    return error


def compute_score_json(a, b, c, custom=False):
    error = compute_score(a, b, c, custom)
    return json.dumps({'error': error})


def calculer_score_droite_geo(custom=False, validate=None, error_msg=None, banque=True, success_msg=None,
                              animation=True, ensure_draw=False, test_r = None, test_d = None):
    global input_values

    if custom:
        deux_caracteristiques = common.challenge.deux_caracteristiques_custom
    else:
        deux_caracteristiques = common.challenge.deux_caracteristiques

    test_dataset = True if test_r is not None and test_d is not None else False

    base_score = compute_score(input_values['a'], input_values['b'], input_values['c'], custom, test_dataset=test_dataset)

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
                pretty_print_success(success_msg)
            pass_breakpoint()

    calculer_score(algorithme, cb=cb,
                   banque=banque, animation=animation, ensure_draw=ensure_draw, test_d=test_d, test_r=test_r)


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

def _mnist_seuillage_0_200_250_np(img):
    a = np.asarray(img)
    a = np.clip(a, 0, 255)
    out = np.empty_like(a, dtype=np.uint8)
    out[a < 180] = 0
    out[(a >= 180) & (a < 220)] = 200
    # Exception (dev) : conserver 240 tel quel (ne pas le ramener à 250)
    out[(a >= 220) & (a <= 239)] = 250
    out[a == 240] = 240
    out[a > 240] = 250
    return out


def _mnist_mean_zone_inclusive(arr, zone):
    (r0, c0), (r1, c1) = zone
    rmin, rmax = min(int(r0), int(r1)), max(int(r0), int(r1))
    cmin, cmax = min(int(c0), int(c1)), max(int(c0), int(c1))
    return float(np.mean(arr[rmin:rmax + 1, cmin:cmax + 1]))

def _mnist_zone_dims(zone):
    (r0, c0), (r1, c1) = zone
    rmin, rmax = min(int(r0), int(r1)), max(int(r0), int(r1))
    cmin, cmax = min(int(c0), int(c1)), max(int(c0), int(c1))
    n_rows = (rmax - rmin + 1)
    n_cols = (cmax - cmin + 1)
    return n_rows, n_cols, n_rows * n_cols


def _mnist_ref_id_for_class(target):
    ids_images_ref = getattr(common.challenge, "ids_images_ref", None)
    if ids_images_ref is not None:
        try:
            ids_images_ref = tuple(ids_images_ref)
        except Exception:
            ids_images_ref = None

    r = getattr(common.challenge, "r_train", None)
    if ids_images_ref and r is not None:
        for idx in ids_images_ref:
            try:
                i = int(idx)
                if int(r[i]) == int(target):
                    return i
            except Exception:
                continue

    if r is None:
        return None
    try:
        return int(np.where(np.asarray(r) == int(target))[0][0])
    except Exception:
        return None


def _mnist_expected_carac_exo_for_ref_image():
    zone_x = getattr(common.challenge, "zone_1_exo", None)
    zone_y = getattr(common.challenge, "zone_2_exo", None)
    if zone_x is None or zone_y is None:
        # Fallback (anciens notebooks) : zones de référence génériques
        zone_x = getattr(common.challenge, "zone_1_ref", None)
        zone_y = getattr(common.challenge, "zone_2_ref", None)
    if zone_x is None or zone_y is None:
        return None

    classes = getattr(common.challenge, "classes", (2, 7))
    target = int(classes[1]) if classes and len(classes) >= 2 else 7
    ref_id = _mnist_ref_id_for_class(target)
    if ref_id is None:
        return None

    d = getattr(common.challenge, "d_train", None)
    if d is None:
        return None

    img = d[ref_id]
    img_t = _mnist_seuillage_0_200_250_np(img)
    expected_x = _mnist_mean_zone_inclusive(img_t, zone_x)
    expected_y = _mnist_mean_zone_inclusive(img_t, zone_y)
    return expected_x, expected_y, ref_id


def function_validation_carac_x(errors, answers):
    x_7 = answers["x_7"]

    # Sécurité (normalement déjà géré par MathadataValidateVariables)
    if x_7 is Ellipsis:
        errors.append("Remplace les ... par ta réponse pour x_7.")
        return False

    # Type attendu : un nombre
    if isinstance(x_7, str):
        errors.append('x_7 doit être un nombre, pas un texte (enlève les guillemets).')
        return False
    if isinstance(x_7, tuple) and len(x_7) == 2:
        errors.append("Pour écrire un nombre décimal, utilise un point : par exemple `183.3` (pas `183,3`).")
        return False
    if isinstance(x_7, (list, dict, np.ndarray, tuple)):
        errors.append("x_7 doit être un nombre (pas une liste/tuple).")
        return False
    # bool est un sous-type de int -> on l'exclut explicitement
    if isinstance(x_7, (bool, np.bool_)):
        errors.append("x_7 doit être un nombre, pas True/False.")
        return False
    if not isinstance(x_7, (int, float, np.integer, np.floating)):
        errors.append("x_7 doit être un nombre.")
        return False

    # Valeur attendue
    try:
        x = float(x_7)
    except Exception:
        errors.append("x_7 doit être un nombre.")
        return False

    expected = _mnist_expected_carac_exo_for_ref_image()
    if expected is None:
        errors.append("Erreur interne : impossible de calculer la valeur attendue (images/zones de référence manquantes).")
        return False
    expected_x, _, _ref_id = expected

    # Tolérance : si la moyenne attendue est décimale, on accepte un arrondi au dixième / à l'unité.
    expected_is_integerish = abs(expected_x - round(expected_x)) <= 0.05
    tol = 0.05 if expected_is_integerish else 0.55
    if abs(x - expected_x) <= tol:
        return True

    if x < 0 or x > 255:
        # Si l'élève donne une valeur >255, c'est presque toujours la somme des pixels (oubli de la division).
        try:
            zone_x = getattr(common.challenge, "zone_1_exo", None) or getattr(common.challenge, "zone_1_ref", None)
            if zone_x is not None:
                n_rows, n_cols, n = _mnist_zone_dims(zone_x)
                errors.append(
                    f"Ta réponse est trop grande pour une moyenne de pixels (elle doit être entre 0 et 255). "
                    f"As-tu oublié de diviser par le nombre total de pixels du rectangle ({n_rows}×{n_cols}={n}) ?"
                )
            else:
                errors.append("Ta réponse est trop grande pour une moyenne de pixels : elle doit être comprise entre 0 et 255.")
        except Exception:
            errors.append("x_7 doit être compris entre 0 et 255 (moyenne de pixels).")
        return False

    try:
        zone_x = getattr(common.challenge, "zone_1_exo", None) or getattr(common.challenge, "zone_1_ref", None)
        if zone_x is not None:
            n_rows, n_cols, n = _mnist_zone_dims(zone_x)
            errors.append(
                "Ce n'est pas la bonne valeur. Calcule la moyenne des pixels du rectangle rouge, pour l'image de 7."
            )
        else:
            errors.append("Ce n'est pas la bonne valeur. Recompte bien les pixels du rectangle et calcule la moyenne.")
    except Exception:
        errors.append("Ce n'est pas la bonne valeur. Réessaie encore.")
    return False


validation_carac_x = MathadataValidateVariables(
    {"x_7": None},
    function_validation=function_validation_carac_x,
    success="Bravo, tu peux passer à la suite.",
    get_tips=lambda: (lambda _z: [
        {
            'seconds': 15,
            'trials': 1,
            'operator': 'OR',
            'tip': (
                f"Commence par compter le nombre total de pixels du rectangle : {_z[0]} lignes × {_z[1]} colonnes = ? "
            )
        },
        {
            'seconds': 30,
            'trials': 2,
            'operator': 'OR',
            'tip': (
                "Tu peux compter combien de pixels valent 200 et combien valent 250, puis faire la somme et diviser par le total."
            )
        },
    ])(_mnist_zone_dims(getattr(common.challenge, "zone_1_exo", None) or getattr(common.challenge, "zone_1_ref", None) or [(0, 0), (0, 0)])),
)

def function_validation_carac_y(errors, answers):
    y_7 = answers["y_7"]

    # Sécurité (normalement déjà géré par MathadataValidateVariables)
    if y_7 is Ellipsis:
        errors.append("Remplace les ... par ta réponse pour y_7.")
        return False

    # Type attendu : un nombre
    if isinstance(y_7, str):
        errors.append('y_7 doit être un nombre, pas un texte (enlève les guillemets).')
        return False
    if isinstance(y_7, tuple) and len(y_7) == 2:
        errors.append("Pour écrire un nombre décimal, utilise un point : par exemple `12.5` (pas `12,5`).")
        return False
    if isinstance(y_7, (list, dict, np.ndarray, tuple)):
        errors.append("y_7 doit être un nombre (pas une liste/tuple).")
        return False
    # bool est un sous-type de int -> on l'exclut explicitement
    if isinstance(y_7, (bool, np.bool_)):
        errors.append("y_7 doit être un nombre, pas True/False.")
        return False
    if not isinstance(y_7, (int, float, np.integer, np.floating)):
        errors.append("y_7 doit être un nombre.")
        return False

    # Valeur attendue
    try:
        y = float(y_7)
    except Exception:
        errors.append("y_7 doit être un nombre.")
        return False

    expected = _mnist_expected_carac_exo_for_ref_image()
    if expected is None:
        errors.append("Erreur interne : impossible de calculer la valeur attendue (images/zones de référence manquantes).")
        return False
    _, expected_y, _ref_id = expected

    expected_is_integerish = abs(expected_y - round(expected_y)) <= 0.05
    tol = 0.05 if expected_is_integerish else 0.55
    if abs(y - expected_y) <= tol:
        return True

    if y < 0 or y > 255:
        try:
            zone_y = getattr(common.challenge, "zone_2_exo", None) or getattr(common.challenge, "zone_2_ref", None)
            if zone_y is not None:
                n_rows, n_cols, n = _mnist_zone_dims(zone_y)
                errors.append(
                    f"Ta réponse est trop grande pour une moyenne de pixels (elle doit être entre 0 et 255). "
                    f"As-tu oublié de diviser par le nombre total de pixels du rectangle ({n_rows}×{n_cols}={n}) ?"
                )
            else:
                errors.append("Ta réponse est trop grande pour une moyenne de pixels : elle doit être comprise entre 0 et 255.")
        except Exception:
            errors.append("y_7 doit être compris entre 0 et 255 (moyenne de pixels).")
        return False

    try:
        zone_y = getattr(common.challenge, "zone_2_exo", None) or getattr(common.challenge, "zone_2_ref", None)
        if zone_y is not None:
            n_rows, n_cols, n = _mnist_zone_dims(zone_y)
            errors.append(
                "Ce n'est pas la bonne valeur. Calcule la moyenne des pixels du rectangle bleu, pour l'image de 7."
            )
        else:
            errors.append("Ce n'est pas la bonne valeur. Recompte bien les pixels du rectangle et calcule la moyenne.")
    except Exception:
        errors.append("Ce n'est pas la bonne valeur. Réessaie encore.")
    return False


validation_carac_y = MathadataValidateVariables(
    {"y_7": None},
    function_validation=function_validation_carac_y,
    success="Bravo, tu peux passer à la suite.",
    get_tips=lambda: (lambda _z: [
        {
            'seconds': 15,
            'trials': 1,
            'operator': 'OR',
            'tip': (
                f"Commence par compter le nombre total de pixels du rectangle : {_z[0]} lignes × {_z[1]} colonnes = ? "
            )
        },
        {
            'seconds': 30,
            'trials': 2,
            'operator': 'OR',
            'tip': (
                "Vérifie bien que tu prends les pixels du rectangle bleu (caractéristique y). "
            )
        },
    ])(_mnist_zone_dims(getattr(common.challenge, "zone_2_exo", None) or getattr(common.challenge, "zone_2_ref", None) or [(0, 0), (0, 0)])),
)

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
