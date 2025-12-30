/**
 * BLOCK BLAST SOLVER - DOM & Initialization
 */

const fileInput = document.getElementById('fileInput');
const sourceCanvas = document.getElementById('sourceCanvas');
const ctx = sourceCanvas.getContext('2d');
const previewPanel = document.getElementById('previewPanel');
const resultsPanel = document.getElementById('resultsPanel');
const resultGrid = document.getElementById('resultGrid');
const uploadText = document.getElementById('uploadText');

let currentImage = null;
let isProcessing = false;

// Event Listeners
fileInput.addEventListener('click', () => fileInput.value = '');
fileInput.addEventListener('change', handleFileSelect);

/**
 * FILE HANDLING
 */

function handleFileSelect(e) {
    if (isProcessing) return;

    const file = e.target.files[0];
    if (!file) return;

    resetAppUI();
    uploadText.textContent = "Loading...";

    const reader = new FileReader();
    reader.onload = function (event) {
        const img = new Image();
        img.onload = function () {
            setupCanvas(img);
            startProcessing();
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(file);
}

function setupCanvas(img) {
    currentImage = img;
    const DisplayMaxW = window.innerWidth > 600 ? 600 : window.innerWidth;
    const scale = Math.min(1, DisplayMaxW / img.width);

    sourceCanvas.width = img.width * scale;
    sourceCanvas.height = img.height * scale;

    ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    ctx.drawImage(img, 0, 0, sourceCanvas.width, sourceCanvas.height);
    previewPanel.classList.remove('hidden');
}

function startProcessing() {
    uploadText.textContent = "Processing...";
    isProcessing = true;
    setTimeout(processWithOpenCV, 100);
}

function resetAppUI() {
    currentImage = null;
    previewPanel.classList.add('hidden');
    resultsPanel.classList.add('hidden');
    resultGrid.innerHTML = '';
    const overlay = document.getElementById('gridOverlay');
    if (overlay) overlay.innerHTML = '';


    const elementsToHide = ['figuresSection', 'figuresImagesSection', 'suggestionsPanel'];
    elementsToHide.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add('hidden');
    });

    const elementsToClear = ['figuresContainer', 'figuresImagesContainer', 'suggestionsList'];
    elementsToClear.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '';
    });

    ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    uploadText.textContent = "Scan Image";
}

/**
 * OPENCV IMAGE PROCESSING PIPELINE
 */

function processWithOpenCV() {
    if (!window.opencvLoaded) {
        alert("OpenCV is still loading, please wait...");
        uploadText.textContent = "Scan Image";
        isProcessing = false;
        return;
    }

    let src, dst, gray, contours, hierarchy;

    try {
        src = cv.imread(sourceCanvas);
        dst = new cv.Mat();
        gray = new cv.Mat();

        // 1. Preprocessing
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
        cv.adaptiveThreshold(gray, dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);

        // 2. Find Board
        contours = new cv.MatVector();
        hierarchy = new cv.Mat();
        cv.findContours(dst, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        const boardContour = findBoardContour(contours);

        if (boardContour) {
            // 3. Extract Board
            const warped = warpPerspective(src, boardContour);
            cv.imshow(sourceCanvas, warped);

            const boardPattern = analyzeGrid(warped);

            // 4. Detect Pieces
            const boardRect = cv.boundingRect(boardContour);
            const pieces = detectFigures(src, boardRect, gray);

            // 5. Run Solve Engine
            if (boardPattern && pieces && pieces.length > 0) {
                suggestMoves(boardPattern, pieces);
            }

            warped.delete();
            boardContour.delete();
        } else {
            alert("Could not detect a clear 8x8 grid. Trying fallback...");
            analyzeGrid(src);
        }

        finalizeUI();

    } catch (err) {
        console.error(err);
        alert("Error processing image: " + err);
        uploadText.textContent = "Scan Image";
    } finally {
        isProcessing = false;
        // Cleanup all Mats
        if (src) src.delete();
        if (dst) dst.delete();
        if (gray) gray.delete();
        if (contours) contours.delete();
        if (hierarchy) hierarchy.delete();
    }
}

function findBoardContour(contours) {
    let maxArea = 0;
    let bestContour = null;
    let approx = new cv.Mat();

    for (let i = 0; i < contours.size(); ++i) {
        let contour = contours.get(i);
        let area = cv.contourArea(contour);

        if (area < 1000) continue;

        let peri = cv.arcLength(contour, true);
        cv.approxPolyDP(contour, approx, 0.02 * peri, true);

        if (approx.rows === 4 && area > maxArea) {
            maxArea = area;
            if (bestContour) bestContour.delete();
            bestContour = approx.clone();
        }
    }
    approx.delete();
    return bestContour;
}

function finalizeUI() {
    resultsPanel.classList.remove('hidden');
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
    uploadText.textContent = "Scan Image";
}

/**
 * BOARD EXTRACTION & ANALYSIS
 */

function warpPerspective(src, contour) {
    let pts = [];
    for (let i = 0; i < 4; i++) {
        pts.push({ x: contour.intPtr(i)[0], y: contour.intPtr(i)[1] });
    }

    pts.sort((a, b) => a.y - b.y);
    let top = pts.slice(0, 2).sort((a, b) => a.x - b.x);
    let bottom = pts.slice(2, 4).sort((a, b) => a.x - b.x);
    let orderedPts = [top[0], top[1], bottom[1], bottom[0]];

    const size = 600;
    let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
        orderedPts[0].x, orderedPts[0].y,
        orderedPts[1].x, orderedPts[1].y,
        orderedPts[2].x, orderedPts[2].y,
        orderedPts[3].x, orderedPts[3].y
    ]);
    let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, size, 0, size, size, 0, size]);

    let M = cv.getPerspectiveTransform(srcTri, dstTri);
    let dst = new cv.Mat();
    cv.warpPerspective(src, dst, M, new cv.Size(size, size));

    srcTri.delete(); dstTri.delete(); M.delete();
    return dst;
}

function analyzeGrid(mat) {
    const rows = 8, cols = 8;
    const cellW = mat.cols / cols, cellH = mat.rows / rows;
    const cellsData = [];

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            let rect = new cv.Rect(
                Math.floor(c * cellW + cellW * 0.25),
                Math.floor(r * cellH + cellH * 0.25),
                Math.floor(cellW * 0.5),
                Math.floor(cellH * 0.5)
            );
            let roi = mat.roi(rect);
            let mean = cv.mean(roi);
            roi.delete();

            const brightness = (mean[0] * 299 + mean[1] * 587 + mean[2] * 114) / 1000;
            cellsData.push({ row: r, col: c, color: { r: mean[0], g: mean[1], b: mean[2] }, brightness });
        }
    }

    const classifiedCells = kMeansCluster(cellsData);
    renderBoardGrid(classifiedCells);

    const pattern = Array(rows).fill(0).map(() => Array(cols).fill(0));
    classifiedCells.forEach(c => { if (c.groupId === 1) pattern[c.row][c.col] = 1; });
    return pattern;
}

/**
 * FIGURE DETECTION
 */

function detectFigures(src, boardRect, graySrc) {
    const yStart = boardRect.y + boardRect.height + 10;
    // Lower adaptive margin for mobile UI (15% of total height)
    let yEnd = src.rows - Math.floor(src.rows * 0.15);

    if (yStart >= yEnd) {
        if (src.rows - yStart > 40) yEnd = src.rows - 5;
        else return [];
    }

    // Use a fresh, non-blurred grayscale for better edge preservation on figures
    const cleanGray = new cv.Mat();
    cv.cvtColor(src, cleanGray, cv.COLOR_RGBA2GRAY, 0);

    const bottomRect = new cv.Rect(0, yStart, src.cols, yEnd - yStart);
    const bottomRoi = cleanGray.roi(bottomRect);
    const slotWidth = Math.floor(bottomRect.width / 3);
    const results = [];

    for (let i = 0; i < 3; i++) {
        const slotX = i * slotWidth;
        const w = (i === 2) ? (bottomRect.width - slotX) : slotWidth;
        const slotRect = new cv.Rect(slotX, 0, w, bottomRect.height);
        const slotRoi = bottomRoi.roi(slotRect);

        const res = processFigureSlot(slotRoi, slotX, yStart, w, bottomRect.height);
        results.push(res);
        slotRoi.delete();
    }

    renderDetectedFigures(src, results);
    bottomRoi.delete();
    cleanGray.delete();
    return results.map(r => r.pattern);
}

function processFigureSlot(slotRoi, slotX, yStart, w, h) {
    const slotThresh = new cv.Mat();
    cv.threshold(slotRoi, slotThresh, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

    const contours = new cv.MatVector();
    const hier = new cv.Mat();
    cv.findContours(slotThresh, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let minX = w, minY = h, maxX = 0, maxY = 0;
    let foundAny = false;

    // 1. Find the collective bounding box of all significant shapes in the slot
    for (let j = 0; j < contours.size(); j++) {
        const c = contours.get(j);
        const rect = cv.boundingRect(c);
        if (cv.contourArea(c) > 40) { // Noise filter
            foundAny = true;
            minX = Math.min(minX, rect.x);
            minY = Math.min(minY, rect.y);
            maxX = Math.max(maxX, rect.x + rect.width);
            maxY = Math.max(maxY, rect.y + rect.height);
        }
    }

    let pattern = [[1]];
    let finalRows = 1, finalCols = 1;
    let cellSize = 50;

    if (foundAny) {
        const figW = maxX - minX;
        const figH = maxY - minY;

        // 2. Estimate base unit size (cellSize)
        const dims = [];
        for (let j = 0; j < contours.size(); j++) {
            const rect = cv.boundingRect(contours.get(j));
            if (cv.contourArea(contours.get(j)) > 20) { // Lower noise gate for mobile
                dims.push(rect.width, rect.height);
            }
        }

        dims.sort((a, b) => a - b);
        let candidates = dims.filter(d => d > 12); // Capture smaller blocks on mobile
        let baseUnit = 50;

        if (candidates.length > 0) {
            let smallest = candidates[0];
            // If the smallest thing is huge, it's a solid piece. 
            // In mobile scans, smallest unit is often around 30-50px.
            if (smallest > 80) {
                baseUnit = smallest / Math.max(1, Math.round(smallest / 50));
            } else {
                baseUnit = smallest;
            }
        }

        // 3. Determine grid dimensions
        finalCols = Math.max(1, Math.min(5, Math.round(figW / baseUnit)));
        finalRows = Math.max(1, Math.min(5, Math.round(figH / baseUnit)));
        cellSize = (figW / finalCols + figH / finalRows) / 2;

        // 4. Subdivide and Sample
        pattern = Array(finalRows).fill(0).map(() => Array(finalCols).fill(0));
        for (let r = 0; r < finalRows; r++) {
            for (let c = 0; c < finalCols; c++) {
                // Sample the center 70% of the cell
                let cellW = figW / finalCols;
                let cellH = figH / finalRows;
                let sampleRect = new cv.Rect(
                    Math.floor(minX + c * cellW + cellW * 0.15),
                    Math.floor(minY + r * cellH + cellH * 0.15),
                    Math.floor(cellW * 0.7),
                    Math.floor(cellH * 0.7)
                );

                let roi = slotThresh.roi(sampleRect);
                let filled = cv.countNonZero(roi);
                // Lower fill requirement (20%) for thin pieces on mobile
                if (filled / (sampleRect.width * sampleRect.height) > 0.2) {
                    pattern[r][c] = 1;
                }
                roi.delete();
            }
        }
    } else {
        // Fallback placeholder logic
        cellSize = 50;
        minX = Math.floor((w - cellSize) / 2);
        minY = Math.floor((h - cellSize) / 2);
    }

    slotThresh.delete(); contours.delete(); hier.delete();
    return {
        pattern,
        rows: finalRows,
        cols: finalCols,
        cellSize,
        originX: slotX + minX,
        originY: yStart + minY
    };
}

/**
 * STRATEGY ENGINE (SOLVER)
 */

function suggestMoves(board, pieces) {
    const validPieces = pieces.map(p => {
        let offsets = [];
        for (let r = 0; r < p.length; r++) {
            for (let c = 0; c < p[r].length; c++) if (p[r][c] === 1) offsets.push({ dr: r, dc: c });
        }
        return offsets;
    }).filter(p => p.length > 0);

    if (validPieces.length === 0) return;

    let bestScore = -Infinity;
    let bestSequence = null;

    const permutations = getPermutations([...Array(validPieces.length).keys()]);

    const solve = (currentBoard, pieceIndices, currentPath, totalCleared) => {
        if (pieceIndices.length === 0) {
            const score = calculateStrategyScore(currentBoard, totalCleared);
            if (score > bestScore) {
                bestScore = score;
                bestSequence = [...currentPath];
            }
            return;
        }

        const pIdx = pieceIndices[0];
        const pOffsets = validPieces[pIdx];

        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                if (canFit(currentBoard, pOffsets, r, c)) {
                    const boardBefore = currentBoard.map(row => [...row]);
                    const { board: nextBoard, cleared } = applyMove(currentBoard, pOffsets, r, c);

                    currentPath.push({ pieceIdx: pIdx, row: r, col: c, cleared, boardBefore, offsets: pOffsets });
                    solve(nextBoard, pieceIndices.slice(1), currentPath, totalCleared + cleared);
                    currentPath.pop();
                }
            }
        }
    };

    permutations.forEach(pOrder => solve(board, pOrder, [], 0));
    renderSuggestions(bestSequence, validPieces.length);
}

function calculateStrategyScore(board, totalCleared) {
    let score = 5000 + (totalCleared * 200);

    // 3x3 Empty Blocks Bonus
    for (let r = 0; r <= 5; r++) {
        for (let c = 0; c <= 5; c++) {
            let empty = true;
            for (let rr = 0; rr < 3 && empty; rr++) {
                for (let cc = 0; cc < 3; cc++) if (board[r + rr][c + cc] === 1) { empty = false; break; }
            }
            if (empty) score += 50;
        }
    }

    // 5-cell Streak Bonus
    for (let i = 0; i < 8; i++) {
        let hMax = 0, hCur = 0, vMax = 0, vCur = 0;
        for (let j = 0; j < 8; j++) {
            if (board[i][j] === 0) hCur++; else { hMax = Math.max(hMax, hCur); hCur = 0; }
            if (board[j][i] === 0) vCur++; else { vMax = Math.max(vMax, vCur); vCur = 0; }
        }
        if (Math.max(hMax, hCur) >= 5) score += 30;
        if (Math.max(vMax, vCur) >= 5) score += 30;
    }

    return score + (64 - board.flat().reduce((a, b) => a + b, 0));
}

/**
 * UI RENDERING HELPERS
 */

function renderBoardGrid(cells) {
    resultGrid.innerHTML = '';
    const overlay = document.getElementById('gridOverlay');
    if (overlay) overlay.innerHTML = '';

    cells.forEach(cell => {
        const isOccupied = cell.groupId === 1;
        const className = `grid-cell ${isOccupied ? 'occupied' : 'empty'}`;

        const div = document.createElement('div');
        div.className = className;
        resultGrid.appendChild(div);

        if (overlay) {
            const overDiv = document.createElement('div');
            overDiv.className = className;
            overlay.appendChild(overDiv);
        }
    });
}

function renderDetectedFigures(src, results) {
    const figContainer = document.getElementById('figuresContainer');
    const figImgContainer = document.getElementById('figuresImagesContainer');
    const sections = [document.getElementById('figuresSection'), document.getElementById('figuresImagesSection')];

    figContainer.innerHTML = '';
    figImgContainer.innerHTML = '';

    if (results.length > 0) {
        sections.forEach(s => s?.classList.remove('hidden'));
        results.forEach(res => {
            // Render Crop
            const canvas = document.createElement('canvas');
            const sideW = Math.floor(res.cellSize * (res.cols + 0.3));
            const sideH = Math.floor(res.cellSize * (res.rows + 0.3));
            const margin = res.cellSize * 0.15;
            const rect = new cv.Rect(
                Math.max(0, Math.min(src.cols - 10, res.originX - margin)),
                Math.max(0, Math.min(src.rows - 10, res.originY - margin)),
                Math.max(1, Math.min(src.cols, sideW)),
                Math.max(1, Math.min(src.rows, sideH))
            );
            const crop = src.roi(rect);
            cv.imshow(canvas, crop);
            crop.delete();

            const card = document.createElement('div');
            card.className = 'figure-card';
            card.appendChild(canvas);
            figImgContainer.appendChild(card);

            // Render Parsed
            const wrapper = document.createElement('div');
            wrapper.className = 'figure-wrapper';
            const mini = document.createElement('div');
            mini.className = 'figure-mini-grid';
            mini.style.width = `${Math.min(70, res.cols * 25)}px`;
            mini.style.height = `${Math.min(70, res.rows * 25)}px`;
            mini.style.gridTemplateColumns = `repeat(${res.cols}, 1fr)`;

            res.pattern.flat().forEach(val => {
                const cell = document.createElement('div');
                cell.className = `fig-cell ${val ? 'occupied' : 'empty'}`;
                mini.appendChild(cell);
            });
            wrapper.appendChild(mini);
            figContainer.appendChild(wrapper);
        });
    }
}

function renderSuggestions(sequence, piecesCount) {
    const list = document.getElementById('suggestionsList');
    const panel = document.getElementById('suggestionsPanel');
    panel.classList.remove('hidden');
    list.innerHTML = '';

    if (sequence && sequence.length === piecesCount) {
        sequence.forEach((step, i) => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';

            const miniGrid = document.createElement('div');
            miniGrid.className = 'suggestion-mini-grid';
            for (let r = 0; r < 8; r++) {
                for (let c = 0; c < 8; c++) {
                    const cell = document.createElement('div');
                    cell.className = 'sug-cell';
                    const isNew = step.offsets.some(off => (step.row + off.dr === r) && (step.col + off.dc === c));
                    cell.classList.add(isNew ? 'new-piece' : (step.boardBefore[r][c] ? 'existing' : 'empty'));
                    miniGrid.appendChild(cell);
                }
            }

            item.innerHTML = `
                <div class="suggestion-step">${i + 1}</div>
                ${miniGrid.outerHTML}
                <div class="suggestion-details">
                    <span class="suggestion-piece-label">Place Piece ${step.pieceIdx + 1}</span>
                    ${step.cleared > 0 ? `<div class="cleared-badge">✨ Clears ${step.cleared} lines!</div>` : ''}
                </div>
            `;
            list.appendChild(item);
        });
    } else {
        list.innerHTML = `<div class="suggestion-item" style="justify-content:center;color:var(--text-secondary);padding:24px;"><span>No solution found.</span></div>`;
    }
}

/**
 * MATH & ALGORITHMIC UTILS
 */

function kMeansCluster(cells) {
    const points = cells.map(c => [c.color.r, c.color.g, c.color.b]);
    let sorted = [...cells].sort((a, b) => a.brightness - b.brightness);
    let cA = [sorted[0].color.r, sorted[0].color.g, sorted[0].color.b];
    let cB = [sorted[sorted.length - 1].color.r, sorted[sorted.length - 1].color.g, sorted[sorted.length - 1].color.b];
    let assign = new Array(cells.length).fill(0);

    for (let i = 0; i < 10; i++) {
        let changed = false;
        for (let j = 0; j < points.length; j++) {
            let dA = Math.hypot(points[j][0] - cA[0], points[j][1] - cA[1], points[j][2] - cA[2]);
            let dB = Math.hypot(points[j][0] - cB[0], points[j][1] - cB[1], points[j][2] - cB[2]);
            let next = dA < dB ? 0 : 1;
            if (assign[j] !== next) { assign[j] = next; changed = true; }
        }
        if (!changed) break;
        cA = avgPoint(points, assign, 0);
        cB = avgPoint(points, assign, 1);
    }
    return cells.map((c, i) => ({ ...c, groupId: assign[i] }));
}

function avgPoint(pts, assign, group) {
    let sum = [0, 0, 0], count = 0;
    pts.forEach((p, i) => { if (assign[i] === group) { sum[0] += p[0]; sum[1] += p[1]; sum[2] += p[2]; count++; } });
    return count ? sum.map(s => s / count) : [0, 0, 0];
}

function getPermutations(arr) {
    if (arr.length <= 1) return [arr];
    let perms = [];
    for (let i = 0; i < arr.length; i++) {
        let rest = getPermutations(arr.slice(0, i).concat(arr.slice(i + 1)));
        rest.forEach(r => perms.push([arr[i], ...r]));
    }
    return perms;
}

function canFit(b, offsets, r, c) {
    for (let off of offsets) {
        let rr = r + off.dr, cc = c + off.dc;
        if (rr >= 8 || cc >= 8 || b[rr][cc] === 1) return false;
    }
    return true;
}

function applyMove(b, offsets, r, c) {
    const nextB = b.map(row => [...row]);
    offsets.forEach(off => nextB[r + off.dr][c + off.dc] = 1);
    let rows = [], cols = [];
    for (let i = 0; i < 8; i++) {
        if (nextB[i].every(v => v === 1)) rows.push(i);
        let full = true;
        for (let j = 0; j < 8; j++) if (nextB[j][i] === 0) { full = false; break; }
        if (full) cols.push(i);
    }
    rows.forEach(ri => nextB[ri].fill(0));
    cols.forEach(ci => { for (let ri = 0; ri < 8; ri++) nextB[ri][ci] = 0; });
    return { board: nextB, cleared: rows.length + cols.length };
}
