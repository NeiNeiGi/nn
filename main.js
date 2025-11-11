const inputLayerLength = 28;
let hiddenLayerLength = 5;
const outputLayerSize = 10;

const inputLayerSize = inputLayerLength * inputLayerLength;
let hiddenLayerSize = hiddenLayerLength * hiddenLayerLength;

const epochs = 100;

const lossCurveLength = 50;
const lossLandscapeLength = 15;
const lossLandscapeSize = lossLandscapeLength * lossLandscapeLength;

const lossCurveRange = [-2, 10];
const lossLandscapeRange = [-5, 5];

let showT = 0;
let progress = 0;
let progressText = 'starting';

let epoch = 0;
let params;
let prediction;

let epoching = false;
let predicting = false;
let lossCurving = false;
let lossLandscaping = false;

let lossLandscape;
const lossLandscapePoints = [];

function updateLossLandscape() {
	disposeLossLandscape();

	const n = lossLandscapePoints.length;
	if (n <= 0) return;

	let min = Infinity;
	let max = -Infinity;
	for (const loss of lossLandscapePoints) {
		min = Math.min(loss, min);
		max = Math.max(loss, max);
	}

	const points = [];
	const pointData = new Float32Array(n * 3);

	for (let i = 0; i < lossLandscapePoints.length; i++) {
		const f = (lossLandscapePoints[i] - min) / (max - min);
		const p = [
			((i % lossLandscapeLength) / (lossLandscapeLength - 1) * 2 - 1) * 150, 
			-50 + f * 60, 
			(Math.floor(i / lossLandscapeLength) / (lossLandscapeLength - 2) * 2 - 1) * 150
		];
		p.f = f;
		pointData.set(p, i * 3);
		points.push(p);
	}

	const posData = [];
	const intensityData = [];
	const lineData = [];

	const h = Math.floor(n / lossLandscapeLength) - 1;
	
	for (let i = 0, l = Math.min(points.length, lossLandscapeLength) - 1; i < l; i++) {
		lineData.push(...points[i], ...points[i + 1]);
	}

	for (let y = 0; y <= h; y++) {
		const w = (y < h ? lossLandscapeLength : n % lossLandscapeLength) - 2;
		for (let x = 0; x <= w; x++) {
			const a = points[y * lossLandscapeLength + x];
			const b = points[y * lossLandscapeLength + x + 1];
			const c = points[(y + 1) * lossLandscapeLength + x];
			const d = points[(y + 1) * lossLandscapeLength + x + 1];

			y > 0 && a && b && lineData.push(...a, ...b);
			a && c && lineData.push(...a, ...c);
			x === w && b && d && lineData.push(...b, ...d);
			y >= h - 1 && c && d && lineData.push(...c, ...d);

			if (a && b && c && d) {
				posData.push(
					...b, ...a, ...c,
					...b, ...c, ...d
				);
				intensityData.push(
					b.f, a.f, c.f, 
					b.f, c.f, d.f
				);
			}
		}
	}

	lossLandscape = {
		points, 
		pointBuffer: createBuffer(pointData), 
		posBuffer: createBuffer(new Float32Array(posData)), 
		intensityBuffer: createBuffer(new Float32Array(intensityData)), 
		vertexCount: posData.length / 3, 
		lineBuffer: createBuffer(new Float32Array(lineData)), 
		lineCount: lineData.length / 3
	};
}

function disposeLossLandscape() {
	if (lossLandscape) {
		gl.deleteBuffer(lossLandscape.pointBuffer);
		gl.deleteBuffer(lossLandscape.posBuffer);
		gl.deleteBuffer(lossLandscape.intensityBuffer);
		gl.deleteBuffer(lossLandscape.lineBuffer);
		lossLandscape = null;
	}
}

function updateEpochProgress() {
	progress = Math.min(1, epoch / epochs);
	progressText = `training epoch ${epoch}/${epochs}`;
}

function reset() {
	epoch = 0;
	updateEpochProgress();
	resetGraphs();

	lossLandscapePoints.length = 0;
	disposeLossLandscape();

	predictT = 0;
	prediction = null;
	params = null;

	epoching = false;
	predicting = false;
	lossCurving = false;
	lossLandscaping = false;

	createModel();
}

const worker = new Worker('worker.js');

let loaded = false;

worker.onmessage = function (event) {
	const msg = event.data;

	switch (msg.id) {
		case 'progress':
			progress = msg.percent;
			progressText = `loading dataset ${(msg.percent * 100).toFixed(2)}%`;
			break;

		case 'loaded':
			loaded = true;
			createDataset();
			setLearningRate();
			reset();
			break;

		case 'failed':
			alert(`Failed to load dataset, can't train model. Reload the page to retry.`);
			break;

		case 'epoch':
			if (msg.modelId !== modelId) break;
			epoching = false;

			console.log(`Epoch #${epoch}:\nTrain Loss: ${msg.trainLoss}, Train Accuracy: ${(msg.trainAccuracy * 100).toFixed(2)}%\nVal Loss: ${msg.valLoss}, Val Accuracy: ${(msg.valAccuracy * 100).toFixed(2)}%\nTime Taken: ${(msg.timeTaken / 1000).toFixed(2)}s`);

			addGraph('trainLoss', msg.trainLoss);
			addGraph('trainAccuracy', msg.trainAccuracy * 100);
			addGraph('valLoss', msg.valLoss);
			addGraph('valAccuracy', msg.valAccuracy * 100);
			addGraph('epochTime', msg.timeTaken / 1000);

			epoch++;
			updateEpochProgress();

			graphs.lossCurve.points.length = 0;
			graphs.lossCurve.max = -Infinity;

			lossLandscapePoints.length = 0;
			disposeLossLandscape();
			break;

		case 'params':
			if (msg.modelId !== modelId) break;
			params = msg;
			break;

		case 'prediction':
			if (msg.modelId !== modelId) break;
			predicting = false;

			for (let i = 0; i < inputLayerSize; i++) {
				activationData[i] = msg.x[i];
			}

			let minA1 = Infinity;
			let maxA1 = -Infinity;
			for (let i = 0; i < hiddenLayerSize; i++) {
				const v = msg.a1[i];
				minA1 = Math.min(minA1, v);
				maxA1 = Math.max(maxA1, v);
			}

			for (let i = 0; i < hiddenLayerSize; i++) {
				activationData[inputLayerSize + i] = (msg.a1[i] - minA1) / (maxA1 - minA1);
			}

			let maxA2 = -Infinity;
			let maxIndex = -1;
			for (let i = 0; i < outputLayerSize; i++) {
				const v = msg.a2[i];
				if (v > maxA2) {
					maxA2 = v;
					maxIndex = i;
				}
			}

			for (let i = 0; i < outputLayerSize; i++) {
				activationData[inputLayerSize + hiddenLayerSize + i] = msg.a2[i];
			}

			gl.bindBuffer(gl.ARRAY_BUFFER, activationBuffer);
			gl.bufferSubData(gl.ARRAY_BUFFER, 0, activationData);

			msg.result = maxIndex;
			prediction = msg;

			console.log(`prediction: ${maxIndex}, prob: ${maxA2}`);
			break;

		case 'lossCurve':
			if (msg.modelId !== modelId) break;
			lossCurving = false;
			addGraph('lossCurve', msg.value);
			break;

		case 'lossLandscape': 
			if (msg.modelId !== modelId) break;
			lossLandscaping = false;
			lossLandscapePoints.push(msg.value);
			updateLossLandscape();
			break;

		default:
			console.log(`Unknown message from worker: ${msg.id}`);
	}
}

let modelId;
function createModel() {
	modelId = Math.random().toString(32).slice(2);
	worker.postMessage({
		id: 'createModel', 
		modelId, 
		inputLayerSize, 
		hiddenLayerSize, 
		outputLayerSize
	});
}

function setLearningRate() {
	worker.postMessage({
		id: 'setLearningRate', 
		value: settings.learningRate
	});
}

function createDataset() {
	loaded && worker.postMessage({
		id: 'createDataset', 
		trainSplit: settings.trainSplit, 
		dataSplit: settings.dataSplit
	});
}

function isTraining() {
	return settings.trainingEnabled && (epoch < epochs || settings.endlessTraining);
}

let userInput = null;

function setUserInput(image) {
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = inputLayerLength;
	const ctx = canvas.getContext('2d');

	ctx.drawImage(image, 0, 0, inputLayerLength, inputLayerLength);

	const imageData = ctx.getImageData(0, 0, inputLayerLength, inputLayerLength);

	userInput = new Float32Array(inputLayerSize);
	for (let i = 0; i < inputLayerSize; i++) {
		userInput[i] = imageData.data[i * 4 + 3] / 255;
	}

	predictT = 0;
}

// rendering

CanvasRenderingContext2D.prototype.scale2 = function (f) {
	this.scale(f, f);
}

const colors = {
	activation: '#ffeb3b', 
	label: '#fb382a'
};

let graphs;

function initGraphs() {
	graphs = {};

	const list = ['trainLoss', 'trainAccuracy', 'valLoss', 'valAccuracy', 'epochTime', 'lossCurve'];

	for (let i = 0; i < list.length; i++) {
		const key = list[i];
		graphs[key] = {
			name: fromCamel(key), 
			points: [], 
			max: -Infinity,
			i: 1 + i, 
			visible: true
		};
	}
}

function addGraph(name, y) {
	if (!isFinite(y)) y = 0;
	const graph = graphs[name];
	graph.points.push(y);
	graph.max = Math.max(graph.max, y);
}

function resetGraphs() {
	for (const key in graphs) {
		const graph = graphs[key];
		graph.points = [];
		graph.max = -Infinity;
	}
}

initGraphs();

const headerEl = document.querySelector('.header');
const settingsEl = document.querySelector('.settings');

const settings = {
	trainingEnabled: true, 
	endlessTraining: false, 
	showLines: true, 
	lossLandscape: false, 
	hiddenLayerSideLength: [hiddenLayerLength, 1, 10, 1], 
	learningRate: [0.5, 0.01, 1, 0.01], 
	trainSplit: [0.8, 0.01, 0.99, 0.01], 
	dataSplit: [0.12, 0.01, 1, 0.01], 
	trainAnimTime: [10, 1, 20, 1], 
	predictAnimTime: [2, 0.1, 10, 0.1], 
	orbitSpeed: [1, 0, 50, 1]
};

const settingOnChange = {
	learningRate: setLearningRate, 
	trainSplit: createDataset, 
	dataSplit: createDataset, 
	hiddenLayerSideLength(n) {
		hiddenLayerLength = n;
		hiddenLayerSize = n * n;

		initObjects();
		reset();
	}
};

for (const key in settings) {
	const value = settings[key];

	if (Array.isArray(value)) {
		const [n, min, max, step] = value;
		settings[key] = n;

		const el = fromHtml(`<div class="row">
			<div>${fromCamel(key)}:</div>
			<input type="range" class="range" min="${min}" max="${max}" step="${step}">
			<div></div>
		</div>`);

		const rangeEl = el.querySelector('.range');
		rangeEl.value = n;
		rangeEl.nextElementSibling.innerText = n;
		rangeEl.onchange = function () {
			settings[key] = parseFloat(this.value);
			this.nextElementSibling.innerText = settings[key];
			settingOnChange[key] && settingOnChange[key](settings[key]);
		}
		rangeEl.oninput = function () {
			this.nextElementSibling.innerText = this.value;
		}

		settingsEl.appendChild(el);
	} else {
		const el = fromHtml(`<label class="row">
			<input type="checkbox" class="checkbox">
			<div>${fromCamel(key)}</div>
		</label>`);

		const checkboxEl = el.querySelector('.checkbox');
		checkboxEl.checked = value;

		checkboxEl.onchange = function () {
			settings[key] = this.checked;
			settingOnChange[key] && settingOnChange[key](settings[key]);
		}

		settingsEl.appendChild(el);
	}
}

const resetBtnEl = fromHtml(`<div class="btn reset-btn" style="margin-top: 3px;">restart</div>`);
resetBtnEl.onclick = reset;
settingsEl.appendChild(resetBtnEl);

function fromCamel(text){
	text = text.replace(/([A-Z])/g,' $1');
	return text.charAt(0).toUpperCase() + text.slice(1);
}

function fromHtml(html) {
	const div = document.createElement('div');
	div.innerHTML = html;
	return div.children[0];
}

const PI2 = Math.PI * 2;

const mesh = {
	indices: [0, 2, 1, 2, 3, 1, 4, 6, 5, 6, 7, 5, 8, 10, 9, 10, 11, 9, 12, 14, 13, 14, 15, 13, 16, 18, 17, 18, 19, 17, 20, 22, 21, 22, 23, 21],
	vertices: [0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
	normals: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1]
};

const canvas = document.getElementById('canvas');

const options = {
	antilias: true, 
	alpha: true
};
const gl = canvas.getContext('webgl', options) || canvas.getContext('experimental-webgl', options);

const hudCanvas = document.getElementById('hudCanvas');
const hudCtx = hudCanvas.getContext('2d');

const uiEl = document.querySelector('.ui');

function resize() {
	canvas.width = window.innerWidth * window.devicePixelRatio;
	canvas.height = window.innerHeight * window.devicePixelRatio;

	hudCanvas.width = canvas.width;
	hudCanvas.height = canvas.height;

	const scale = Math.max(window.innerWidth / 1366, window.innerHeight / 768);

	Object.assign(uiEl.style, {
		transform: `scale(${scale})`, 
		width: window.innerWidth / scale + 'px', 
		height: window.innerHeight / scale + 'px', 
	});
}

window.onresize = function () {
	resize();
	render();
}
resize();

if (!gl) {
	alert(`webgl not supported. get better device LOL XD`);
	throw new Error('webgl not supported');
}

const ext = gl.getExtension('ANGLE_instanced_arrays');
if (!ext) {
	alert(`ext not supported RIP XD LOL no nn 4u XD`);
	throw new Error('ext not supported');
}

gl.enable(gl.DEPTH_TEST);
gl.enable(gl.CULL_FACE);
gl.cullFace(gl.BACK);

gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

const program = createProgram(`

precision mediump float;

attribute vec3 position;
attribute vec3 normal;
attribute vec3 worldPos;
attribute float active;
attribute float objectId;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform float maxActiveId;

varying vec3 vColor;

void main() {
	vec3 p = position + worldPos;
	gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);

	vec3 lightPos = vec3(-50.0, 50.0, 69.0);
	float light = max(0.0, dot(normalize(lightPos - p), normal)) * 0.4 + 0.6;
	vColor = mix(vec3(1.0), vec3(1.0, 0.24, 0.23), objectId < maxActiveId ? active : 0.0) * light;
}

`, `

precision mediump float;

varying vec3 vColor;

void main() {
	gl_FragColor = vec4(vColor, 1.0);
}

`);

const lineGlowSize = 0.1;

const lineProgram = createProgram(`

precision mediump float;

attribute vec3 position;
attribute vec2 distanceAndAlpha;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

varying vec2 vDistanceAndAlpha;

void main() {
	gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
	vDistanceAndAlpha = distanceAndAlpha;
}

`, `

precision mediump float;

varying vec2 vDistanceAndAlpha;

uniform float d;
uniform float alpha;

void main() {
	float w = ${lineGlowSize.toFixed(2.666)};
	float f = smoothstep(1.0, 0.0, abs(vDistanceAndAlpha.x - d * (1.0 + w * 2.0) + w) / w) * 5.0 + 1.0;
	gl_FragColor = vec4(f * vDistanceAndAlpha.y * alpha);
}

`);

const bgProgram = createProgram(`

precision mediump float;

attribute vec3 position;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

varying vec3 vPos;

void main() {
	gl_Position = projectionMatrix * viewMatrix * vec4(position * 5000.0, 1.0);
	gl_Position.z = 0.0;
	vPos = position;
}

`, `

precision mediump float;

uniform sampler2D map;

varying vec3 vPos;

void main() {
	vec3 p = vPos * 2.0;
	vec2 uv = abs(p.z) > 0.999 ? p.xy : (abs(p.x) > 0.999 ? p.zy : p.xz);
	gl_FragColor = texture2D(map, uv * 20.0);
}

`);

const planeProgram = createProgram(`

precision mediump float;

attribute vec3 position;
attribute float intensity;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

varying float vIntensity;

void main() {
	gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
	vIntensity = intensity;
}

`, `

precision mediump float;

varying float vIntensity;

void main() {
	gl_FragColor = vec4(mix(vec3(0.01, 0.66, 0.95), vec3(0.54, 0.81, 0.22), vIntensity), 1.0);
}

`);

const planeLineProgram = createProgram(`

precision mediump float;

attribute vec3 position;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

void main() {
	gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
	gl_Position.z -= 0.02;
}

`, `

precision mediump float;
	
void main() {
	gl_FragColor = vec4(0.5);
}

`);

const map = gl.createTexture();
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, map);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, Grid(32));
gl.generateMipmap(gl.TEXTURE_2D);

const posBuffer = createBuffer(new Float32Array(mesh.vertices));
const normalBuffer = createBuffer(new Float32Array(mesh.normals));

const indexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.indices), gl.STATIC_DRAW);

const gap = 4.666;

let objectCount,
	worldPosData, 
	activationData, 
	objectIdData;

let lineCount, 
	linePosData,
	lineDistanceAndAlphaData;

let worldPosBuffer, 
	activationBuffer, 
	objectIdBuffer;

let linePosBuffer, 
	lineDistanceAndAlphaBuffer;

let inputLayerPositions,
	hiddenLayerPositions, 
	outputLayerPositions;

function initObjects() {
	if (worldPosBuffer) {
		gl.deleteBuffer(worldPosBuffer);
		gl.deleteBuffer(activationBuffer);
		gl.deleteBuffer(objectIdBuffer);

		gl.deleteBuffer(linePosBuffer);
		gl.deleteBuffer(lineDistanceAndAlphaBuffer);
	}

	objectCount = inputLayerSize + hiddenLayerSize + outputLayerSize;
	worldPosData = new Float32Array(3 * objectCount);
	activationData = new Float32Array(objectCount);
	objectIdData = Float32Array.from({ length: objectCount }, (_, i) => i);

	lineCount = 2 * ((inputLayerSize * hiddenLayerSize) + (hiddenLayerSize * outputLayerSize));
	linePosData = new Float32Array(3 * lineCount);
	lineDistanceAndAlphaData = new Float32Array(2 * lineCount);

	for (let i = 0; i < lineCount; i += 2) {
		const x = i >= 2 * inputLayerSize * hiddenLayerSize ? 1 + lineGlowSize * 2 : 0;
		const j = 2 * i;
		lineDistanceAndAlphaData[j] = x;
		lineDistanceAndAlphaData[j + 2] = x + 1;
	}

	// alpha for weights but wasn't visible much so just set to 1 :cri:
	for (let i = 0; i < lineCount; i++) {
		lineDistanceAndAlphaData[i * 2 + 1] = 1;
	}

	inputLayerPositions = [];
	hiddenLayerPositions = [];
	outputLayerPositions = [];

	let bi = 0;

	for (let y = 0; y < inputLayerLength; y++) {
		for (let x = 0; x < inputLayerLength; x++) {
			const p = [
				getCoord(x, inputLayerLength), 
				-getCoord(y, inputLayerLength), 
				-100
			];
			inputLayerPositions.push(p);
			worldPosData.set(p, bi); bi += 3;
		}
	}

	for (let y = 0; y < hiddenLayerLength; y++) {
		for (let x = 0; x < hiddenLayerLength; x++) {
			const p = [
				getCoord(x, hiddenLayerLength), 
				-getCoord(y, hiddenLayerLength), 
				0
			];
			hiddenLayerPositions.push(p);
			worldPosData.set(p, bi); bi += 3;
		}
	}

	for (let y = 0; y < outputLayerSize; y++) {
		const p = [
			0, 
			-getCoord(y, outputLayerSize), 
			100
		];
		outputLayerPositions.push(p);
		worldPosData.set(p, bi); bi += 3;
	}

	let li = 0;

	for (let i = 0; i < hiddenLayerSize; i++) {
		for (let j = 0; j < inputLayerSize; j++) {
			const a = inputLayerPositions[j];
			const b = hiddenLayerPositions[i];

			linePosData[li++] = a[0];
			linePosData[li++] = a[1];
			linePosData[li++] = a[2] + 0.5;

			linePosData[li++] = b[0];
			linePosData[li++] = b[1];
			linePosData[li++] = b[2] - 0.5;
		}
	}

	for (let i = 0; i < outputLayerSize; i++) {
		for (let j = 0; j < hiddenLayerSize; j++) {
			const a = hiddenLayerPositions[j];
			const b = outputLayerPositions[i];

			linePosData[li++] = a[0];
			linePosData[li++] = a[1];
			linePosData[li++] = a[2] + 0.5;

			linePosData[li++] = b[0];
			linePosData[li++] = b[1];
			linePosData[li++] = b[2] - 0.5;
		}
	}

	worldPosBuffer = createBuffer(worldPosData);
	activationBuffer = createBuffer(activationData, true);
	objectIdBuffer = createBuffer(objectIdData);

	linePosBuffer = createBuffer(linePosData);
	lineDistanceAndAlphaBuffer = createBuffer(lineDistanceAndAlphaData, true);
}

function getCoord(x, n) {
	return n > 1 ? (x / (n - 1) * 2 - 1) * n * gap : 0;
}

initObjects();

let nrx = 0.1;
let nry = -7;
let nDepth = 100;

let rx = 0.6;
let ry = -0.5;
let depth = 180;

const minDepth = 2;
const maxDepth = 300;

canvas.onwheel = function (event) {
	nDepth *= (event.deltaY > 0 ? 1.1 : 0.9);
	nDepth = Math.max(minDepth, Math.min(nDepth, maxDepth));
}

canvas.oncontextmenu = () => false;

let picked;

let lastPoint;
canvas.onmousedown = function (event) {
	if (event.button === 0) {
		lastPoint = [event.clientX, event.clientY];
	}
}
window.onmousemove = function (event) {
	if (lastPoint) {
		const p = [event.clientX, event.clientY];
		const dx = p[0] - lastPoint[0];
		const dy = p[1] - lastPoint[1];
		nrx += dy * 0.01;
		nry -= dx * 0.01;
		nrx = Math.max(-Math.PI / 2, Math.min(nrx, Math.PI / 2));
		lastPoint = p;
	} else if (params) {
		picked = pick();
	}
}
window.onmouseup = function (event) {
	if (event.button === 0) {
		lastPoint = null;
	}
}

function pick() {
	const px = event.clientX;
	const py = event.clientY;

	const W = window.innerWidth;
	const H = window.innerHeight;
	const r = getScreenSpaceSize() * H + 10;

	for (let i = 0; i < hiddenLayerSize; i++) {
		const pos = hiddenLayerPositions[i];
		const p = project2(...pos);
		
		if (p && Math.hypot(px - p[0] * W, py - p[1] * H) < r) {
			const x = i % hiddenLayerLength;
			const y = Math.floor(i / hiddenLayerLength);

			return {
				name: `HiddenLayer (${x + 1}, ${y + 1})`, 
				image: createImage(
					params.w1.slice(i * inputLayerSize, (i + 1) * inputLayerSize), 
					inputLayerLength
				), 
				pos
			};
		}
	}

	for (let i = 0; i < outputLayerSize; i++) {
		const pos = outputLayerPositions[i];
		const p = project2(...pos);

		if (p && Math.hypot(px - p[0] * W, py - p[1] * H) < r) {
			return {
				name: `OutputLayer (1, ${i + 1})`, 
				image: createImage(
					params.w2.slice(i * hiddenLayerSize, (i + 1) * hiddenLayerSize), 
					hiddenLayerLength
				), 
				pos
			};
		}
	}
}

function createImage(data, size) {
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext('2d');

	const imageData = ctx.createImageData(size, size);

	let min = Infinity;
	let max = -Infinity;
	for (let i = 0; i < data.length; i++) {
		min = Math.min(data[i], min);
		max = Math.max(data[i], max);
	}

	for (let i = 0; i < data.length; i++) {
		const f = (data[i] - min) / (max - min) * 255;
		imageData.data.set([f, f, f, 255], i * 4);
	}

	ctx.putImageData(imageData, 0, 0);

	return canvas;
}

function inView(x, y, z) {
	return x > 0 && x < 1 && y > 0 && y < 1 && z > 0 && z < 1;
}

const sketchEl = SketchUI();

let predictT = 0;

let projectionMatrix, viewMatrix;
let showingLossLandscape = false;

let now = 0;
let lastTime = Date.now();
let dt = 0;
let dts = 0;

function update() {
	now = Date.now();
	dt = now - lastTime;
	dts = dt / 1000;
	lastTime = now;

	if (!lastPoint) {
		nry += 0.015 * settings.orbitSpeed * dts;
	}

	let lf = getLerpFactor(0.05);
	rx = lerpAngle(rx, nrx, lf);
	ry = lerpAngle(ry, nry, lf);
	depth = lerp(depth, nDepth, lf);

	showingLossLandscape = settings.lossLandscape && lossLandscape && !isTraining();
	graphs.lossCurve.visible = showingLossLandscape;

	lf = getLerpFactor(0.1);
	showT = lerp(showT, 1, lf);
	headerEl.style.transform = `translateX(${(1 - showT) * 200}%)`;
	settingsEl.style.transform = `translateY(${(1 - showT) * -200}%)`;
	sketchEl.style.transform = `translateY(${(1 - showT) * 200}%)`;

	if (loaded) {
		if (isTraining()) {
			predictT = 0;
			if (!epoching) {
				epoching = true;
				worker.postMessage({
					id: 'train'
				});
			}
		} else if (settings.lossLandscape) {
			if (!lossCurving && graphs.lossCurve.points.length <= lossCurveLength) {
				lossCurving = true;
				worker.postMessage({
					id: 'lossCurve', 
					x: toRange(lossCurveRange, graphs.lossCurve.points.length / lossCurveLength)
				});
			}

			if (!lossLandscaping && lossLandscapePoints.length < lossLandscapeSize) {
				lossLandscaping = true;
				const i = lossLandscapePoints.length;
				worker.postMessage({
					id: 'lossLandscape', 
					x: toRange(lossLandscapeRange, (i % lossLandscapeLength) / lossLandscapeLength), 
					y: toRange(lossLandscapeRange, Math.floor(i / lossLandscapeLength) / lossLandscapeLength)
				});
			}
		} else if (!predicting) {
			if (predictT === 0) {
				predicting = true;
				worker.postMessage({
					id: 'predict', 
					x: userInput
				});
				userInput = null;
			}

			predictT += dts / settings.predictAnimTime;
			if (predictT >= 1.5) {
				predictT = 0;
				prediction = null;
			}
		}
	}
}

function toRange([min, max], f) {
	return min + (max - min) * f;
}

function render() {
	const cosX = Math.cos(rx);
	const sinX = Math.sin(rx);
	const cosY = Math.cos(ry);
	const sinY = Math.sin(ry);

	viewMatrix = [
		cosY, sinY * -sinX, sinY * cosX, 0, 
		0, cosX, sinX, 0, 
		-sinY, cosY * -sinX, cosY * cosX, 0, 
		0, 0, -depth, 1
	];

	const near = 0.1;
	const far = 1000;
	const fov = 60;
	const f = 1 / Math.tan(fov * Math.PI / 360);
	const nf = 1 / (near - far);
	const aspect = canvas.width / canvas.height;

	projectionMatrix = [
		f / aspect, 0, 0, 0, 
		0, f, 0, 0, 
		0, 0, (near + far) * nf, -1, 
		0, 0, 2 * far * near * nf, 1
	];

	drawHud(hudCtx);

	gl.viewport(0, 0, canvas.width, canvas.height);

	gl.clearColor(0, 0, 0, 1);
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	renderBg();
	
	if (showingLossLandscape) {
		renderLossLandscape();	
	} else {
		renderBoxes();
		settings.showLines && renderLines();
	}
}

function renderBg() {
	gl.useProgram(bgProgram);
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, map);
	gl.uniform1i(bgProgram.uniforms.map, 0);

	gl.uniformMatrix4fv(bgProgram.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(bgProgram.uniforms.viewMatrix, false, viewMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
	gl.enableVertexAttribArray(bgProgram.attributes.position);
	gl.vertexAttribPointer(bgProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);

	gl.cullFace(gl.FRONT);
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
	gl.drawElements(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0);			
	gl.cullFace(gl.BACK);

	gl.disableVertexAttribArray(bgProgram.attributes.position);

	gl.clear(gl.DEPTH_BUFFER_BIT);
}

function renderBoxes() {
	gl.useProgram(program);

	let n = 0;
	if (predictT > 0) {
		n = inputLayerSize;
		if (predictT >= 0.5) n += hiddenLayerSize;
		if (predictT >= 1) n += outputLayerSize;
	}

	gl.uniform1f(program.uniforms.maxActiveId, n);
	gl.uniformMatrix4fv(program.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(program.uniforms.viewMatrix, false, viewMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
	gl.enableVertexAttribArray(program.attributes.position);
	gl.vertexAttribPointer(program.attributes.position, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.position, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
	gl.enableVertexAttribArray(program.attributes.normal);
	gl.vertexAttribPointer(program.attributes.normal, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.normal, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, worldPosBuffer);
	gl.enableVertexAttribArray(program.attributes.worldPos);
	gl.vertexAttribPointer(program.attributes.worldPos, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.worldPos, 1);

	gl.bindBuffer(gl.ARRAY_BUFFER, activationBuffer);
	gl.enableVertexAttribArray(program.attributes.active);
	gl.vertexAttribPointer(program.attributes.active, 1, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.active, 1);

	gl.bindBuffer(gl.ARRAY_BUFFER, objectIdBuffer);
	gl.enableVertexAttribArray(program.attributes.objectId);
	gl.vertexAttribPointer(program.attributes.objectId, 1, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.objectId, 1);

	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
	ext.drawElementsInstancedANGLE(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0, objectCount);

	gl.disableVertexAttribArray(program.attributes.position);
	gl.disableVertexAttribArray(program.attributes.normal);
	gl.disableVertexAttribArray(program.attributes.worldPos);
	gl.disableVertexAttribArray(program.attributes.active);
	gl.disableVertexAttribArray(program.attributes.objectId);
}

function renderLines() {
	gl.useProgram(lineProgram);

	let t = 0;

	if (isTraining()) {
		t = now / (settings.trainAnimTime * 1000) % 1;
		if (t < 0.5) t = t / 0.5;
		else t = 1 - (t - 0.5) / 0.5;
		t = t * 2;
	} else {
		t = predictT * 2;
	}

	gl.uniform1f(lineProgram.uniforms.d, t);
	gl.uniform1f(lineProgram.uniforms.alpha, 0.03);

	gl.uniformMatrix4fv(lineProgram.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(lineProgram.uniforms.viewMatrix, false, viewMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuffer);
	gl.enableVertexAttribArray(lineProgram.attributes.position);
	gl.vertexAttribPointer(lineProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, lineDistanceAndAlphaBuffer);
	gl.enableVertexAttribArray(lineProgram.attributes.distanceAndAlpha);
	gl.vertexAttribPointer(lineProgram.attributes.distanceAndAlpha, 2, gl.FLOAT, false, 0, 0);

	gl.drawArrays(gl.LINES, 0, lineCount);

	gl.disableVertexAttribArray(lineProgram.attributes.position);
	gl.disableVertexAttribArray(lineProgram.attributes.distance);
	gl.disableVertexAttribArray(lineProgram.attributes.distanceAndAlpha);
}

function renderLossLandscape() {
	if (lossLandscape.vertexCount > 0) {
		gl.useProgram(planeProgram);

		gl.uniformMatrix4fv(planeProgram.uniforms.projectionMatrix, false, projectionMatrix);
		gl.uniformMatrix4fv(planeProgram.uniforms.viewMatrix, false, viewMatrix);

		gl.bindBuffer(gl.ARRAY_BUFFER, lossLandscape.posBuffer);
		gl.enableVertexAttribArray(planeProgram.attributes.position);
		gl.vertexAttribPointer(planeProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);

		gl.bindBuffer(gl.ARRAY_BUFFER, lossLandscape.intensityBuffer);
		gl.enableVertexAttribArray(planeProgram.attributes.intensity);
		gl.vertexAttribPointer(planeProgram.attributes.intensity, 1, gl.FLOAT, false, 0, 0);

		gl.disable(gl.CULL_FACE);
		gl.drawArrays(gl.TRIANGLES, 0, lossLandscape.vertexCount);
		gl.enable(gl.CULL_FACE);

		gl.disableVertexAttribArray(planeProgram.attributes.position);
		gl.disableVertexAttribArray(planeProgram.attributes.intensity);
	}

	// 

	gl.useProgram(planeLineProgram);

	gl.uniformMatrix4fv(planeLineProgram.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(planeLineProgram.uniforms.viewMatrix, false, viewMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, lossLandscape.lineBuffer);
	gl.enableVertexAttribArray(planeLineProgram.attributes.position);
	gl.vertexAttribPointer(planeLineProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);

	gl.drawArrays(gl.LINES, 0, lossLandscape.lineCount);

	gl.disableVertexAttribArray(planeLineProgram.attributes.position);
}

function drawHud(ctx) {
	const canvas = ctx.canvas;
	const scale = Math.max(canvas.width / 1366, canvas.height / 768);

	ctx.clearRect(0, 0, canvas.width, canvas.height);

	const W = canvas.width / scale;
	const H = canvas.height / scale;

	ctx.save();
	ctx.scale2(scale);

	if (666) {
		const p = project2(69, 699, 420);
		if (p) {
			ctx.save();
			ctx.translate(p[0] * W, p[1] * H);
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.font = 'bolder 50px monospace';
			ctx.fillStyle = 'brown';
			ctx.fillText('⛧OD⛧', 0, 4);

			ctx.beginPath();
			ctx.moveTo(-60, -20);
			ctx.lineTo(60, 20);
			ctx.moveTo(60, -20);
			ctx.lineTo(-60, 20);
			ctx.strokeStyle = 'red';
			ctx.lineWidth = 5;
			ctx.stroke();

			ctx.font = 'normal 10px monospace';
			ctx.fillStyle = getHoverColor();
			ctx.fillText('⛧6⛧6⛧6⛧', 0, 0);

			if (hiddenLayerSize === 1) {
				ctx.fillStyle = 'white';
				ctx.textBaseline = 'top';
				ctx.fillText(`▲f: dats ur brain u dum p of poo xp`, 0, 28);
				ctx.fillText(`z: so tru bbg :3`, 0, 28 + 13);
			}

			ctx.restore();
		} 
	}

	if (showingLossLandscape) {
		ctx.fillStyle = colors.activation;
		ctx.textAlign = 'center';
		ctx.textBaseline = 'bottom';

		for (let i = 0; i < lossLandscape.points.length; i++) {
			const loss = lossLandscapePoints[i];
			const p = project2(...lossLandscape.points[i]);
			if (p) {
				ctx.fillText(loss.toFixed(2), p[0] * W, p[1] * H);
			}
		}
	} else {
		drawNetworkHud(ctx, W, H);
	}

	//

	ctx.save();
	ctx.translate(12, H - 12 - 16);

	const graphWidth = 90;
	const graphHeight = 50;

	for (const key in graphs) {
		const graph = graphs[key];
		if (!graph.visible) continue;

		ctx.save();
		showT < 1 && ctx.translate(0, (1 - Math.pow(showT, graph.i)) * graphHeight * 4);

		ctx.beginPath();
		let y = 0;
		if (graph.points.length === 0) {
			y = -graphHeight;
			ctx.lineTo(0, y);
		} else {
			const l = Math.max(1, graph.points.length - 1);
			for (let i = 0; i < graph.points.length; i++) {
				const v = graph.points[i];
				const x = i / l * graphWidth;
				y = -v / graph.max * graphHeight;
				ctx.lineTo(x, y);
			}
		}
		ctx.lineTo(graphWidth, y);
		ctx.lineTo(graphWidth, 0);
		ctx.lineTo(0, 0);
		ctx.closePath();
		ctx.fillStyle = '#333';
		ctx.globalAlpha = 0.3;
		ctx.fill();
		ctx.lineWidth = 1;
		ctx.strokeStyle = '#888';
		ctx.globalAlpha = 1;
		ctx.stroke();

		ctx.fillStyle = '#888';
		ctx.font = 'normal 16px monospace';
		ctx.textBaseline = 'bottom';
		ctx.textAlign = 'right';
		const n = graph.points.length > 0 ? graph.points[graph.points.length - 1] : 0;
		ctx.fillText(n.toFixed(2), graphWidth, 0);

		ctx.fillStyle = '#fff';
		ctx.font = 'normal 10px monospace';
		ctx.textBaseline = 'top';
		ctx.textAlign = 'left';
		ctx.fillText(graph.name, 0, 7);

		ctx.restore();

		ctx.translate(graphWidth + 15, 0);
	}

	ctx.restore();

	//

	ctx.save();
	ctx.translate(-400 * (1 - showT), H - 130);

	ctx.beginPath();
	ctx.rect(-5, -18, 250 + 5, 36);
	ctx.fillStyle = '#333';
	ctx.globalAlpha = 0.3;
	ctx.fill();
	ctx.lineWidth = 1;
	ctx.strokeStyle = '#888';
	ctx.lineCap = ctx.lineJoin = 'round';
	ctx.globalAlpha = 1;
	ctx.stroke();

	ctx.globalAlpha = 0.1;
	ctx.fillStyle = '#fff';
	ctx.fillRect(0, -14, 246 * progress, 28);

	ctx.globalAlpha = 1;
	ctx.fillStyle = '#fff';
	ctx.font = 'normal 10px monospace';
	ctx.textBaseline = 'middle';
	ctx.textAlign = 'left';
	ctx.fillText(progressText + (progress !== 1 ? '.'.repeat((now / 1000 % 1) * 10) : ''), 10, 0);

	ctx.restore();

	ctx.restore();
}

function drawNetworkHud(ctx, W, H) {
	ctx.lineCap = ctx.lineJoin = 'round';
	ctx.font = 'normal 10px monospace';
	ctx.textBaseline = 'middle';	
	ctx.textAlign = 'center';
	ctx.fillStyle = colors.activation;

	const r = getScreenSpaceSize() * H;
	const offset = r + 10;

	if (prediction && predictT > 0.5 && depth < 200) {
		for (let i = 0; i < hiddenLayerSize; i++) {
			const p = project2(...hiddenLayerPositions[i]);
			if (p) {
				const dir = (i % hiddenLayerLength) % 2 === 0 ? -1 : 1;
				ctx.fillText(prediction.a1[i].toFixed(2), p[0] * W, p[1] * H + dir * offset);
			}
		}
	}

	for (let i = 0; i < outputLayerSize; i++) {
		const p = project2(...outputLayerPositions[i]);
		if (p) {
			const [x, y] = p;
			const dir = x > 0.5 ? 1 : -1;
			ctx.save();
			ctx.translate(x * W, y * H);
			ctx.textAlign = x > 0.5 ? 'left' : 'right';
			ctx.fillStyle = colors.label;
			ctx.fillText(i, dir * offset, 0);

			if (prediction && predictT > 1) {
				ctx.textAlign = x < 0.5 ? 'left' : 'right';
				ctx.fillStyle = colors.activation;
				ctx.fillText(prediction.a2[i].toFixed(2), -dir * offset, 0);

				if (i === prediction.result) {
					ctx.scale(dir, 1);
					ctx.translate(offset + 15 + (Math.sin((now / 200 % 1) * PI2) * 0.5 + 0.5) * 5, -2);
					ctx.beginPath();
					ctx.moveTo(0, 0);
					ctx.lineTo(15, 12);
					ctx.lineTo(15, 5);
					ctx.lineTo(35, 5);
					ctx.lineTo(35, -5);
					ctx.lineTo(15, -5);
					ctx.lineTo(15, -12);
					ctx.closePath();
					ctx.fillStyle = getHoverColor();
					ctx.fill();
				}
			}

			ctx.restore();
		}
	}

	if (picked) {
		const p = project2(...picked.pos);
		if (p) {
			ctx.save();
			ctx.translate(p[0] * W, p[1] * H);
			ctx.lineWidth = 2;
			ctx.strokeStyle = getHoverColor();
			const s = r + (Math.sin(now / 100) * 0.5 + 0.5) * Math.min(r, 20);
			ctx.strokeRect(-s, -s, s * 2, s * 2);

			const size = 90;

			ctx.translate(0, -r - 10 - size);

			ctx.imageSmoothingEnabled = false;
			ctx.drawImage(picked.image, -size / 2, 0, size, size);

			ctx.translate(0, -5);
			ctx.fillStyle = colors.label;
			ctx.textAlign = 'center';
			ctx.textBaseline = 'bottom';
			ctx.fillText(picked.name, 0, 0);

			ctx.restore();
		}
	}
}

function getHoverColor() {
	return `hsl(${(now / 400 % 1) * 360}deg, 100%, 70%)`;
}

function getScreenSpaceSize() {
	const a = project(1, 1, 1);
	const b = project(0, 0, 0);
	return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

function animate() {
	update();
	render();
	window.requestAnimationFrame(animate);
}

animate();

function createProgram(vert, frag) {
	const vShader = createShader(vert, true);
	const fShader = createShader(frag, false);

	const program = gl.createProgram();
	gl.attachShader(program, vShader);
	gl.attachShader(program, fShader);
	gl.linkProgram(program);

	gl.deleteShader(vShader);
	gl.deleteShader(fShader);

	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		throw new Error(`failed to link program: ${gl.getProgramInfoLog(program)}`);
	}

	const attributeCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
	program.attributes = {};
	for (let i = 0; i < attributeCount; i++) {
		const info = gl.getActiveAttrib(program, i);
		program.attributes[info.name] = gl.getAttribLocation(program, info.name);
	}

	const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
	program.uniforms = {};
	for (let i = 0; i < uniformCount; i++) {
		const info = gl.getActiveUniform(program, i);
		program.uniforms[info.name] = gl.getUniformLocation(program, info.name);
	}

	return program;
}

function createShader(src, isVertex) {
	const shader = gl.createShader(isVertex ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER);
	gl.shaderSource(shader, src);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		throw new Error(`failed to compile ${isVertex ? 'vertex' : 'fragment'} shader! ${gl.getShaderInfoLog(shader)}`);
	}

	return shader;
}

function createBuffer(data, dynamic) {
	const buffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
	gl.bufferData(gl.ARRAY_BUFFER, data, dynamic ? gl.DYNAMIC_DRAW :  gl.STATIC_DRAW);
	return buffer;
}

function Grid(size = 20) {
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext('2d');

	ctx.fillStyle = '#151515';
	ctx.fillRect(0, 0, size, size);

	ctx.beginPath();
	const s = size / 2;
	ctx.moveTo(s, 0);
	ctx.lineTo(s, size);
	ctx.moveTo(0, s);
	ctx.lineTo(size, s);
	ctx.lineWidth = size * 0.04;
	ctx.strokeStyle = 'hsla(0, 0%, 100%, 0.02)';
	ctx.stroke();

	return canvas;
}

function lerpAngle(a, b, t) {
	let da = (b - a) % PI2;
	da = 2 * da % PI2 - da;
	return a + da * t;
}

function lerp(start, target, t) {
	const d = target - start;
	if (Math.abs(d) < 1e-4) return target;
	return start + d * t;
}

function getLerpFactor(f) {
	return 1 - Math.exp(-f * dt / 16);
}

function project(x, y, z) {
	const p = transformVector(transformVector([x, y, z, 1], viewMatrix), projectionMatrix);
	x = p[0] / p[3] * 0.5 + 0.5;
	y = 0.5 - p[1] / p[3] * 0.5;
	z = p[2] / p[3] * 0.5 + 0.5;
	return [x, y, z];
}

function project2(x, y, z) {
	const p = project(x, y, z);
	if (inView(...p)) return p;
}

function transformVector(p, matrix) {
	return [
		p[0] * matrix[0] + p[1] * matrix[4] + p[2] * matrix[8] + p[3] * matrix[12], 
		p[0] * matrix[1] + p[1] * matrix[5] + p[2] * matrix[9] + p[3] * matrix[13], 
		p[0] * matrix[2] + p[1] * matrix[6] + p[2] * matrix[10] + p[3] * matrix[14], 
		p[0] * matrix[3] + p[1] * matrix[7] + p[2] * matrix[11] + p[3] * matrix[15]
	];
}

function SketchUI() {
	const size = 150;
	const el = fromHtml(`<div style="
		position: absolute;
		right: 12px;
		bottom: 12px;
		display: flex;
		flex-direction: column;
		align-items: end;
		grid-gap: 5px;
		pointer-events: all;
	">
		<div class="row">
			<div class="btn predict-btn">predict</div>
			<div class="btn clear-btn">clear</div>
		</div>
		<canvas style="
			width: ${size}px;
			height: ${size}px;
			border-radius: 5px;
			background: hsla(0, 0%, 20%, 0.3);
			border: 2px solid hsla(0, 0%, 100%, 0.2);
			pointer-events: all;
		"></canvas>
		<div>draw a digit xp</div>
	</div>`);
	uiEl.appendChild(el);

	const canvas = el.querySelector('canvas');
	canvas.width = canvas.height = size * window.devicePixelRatio;
	const ctx = canvas.getContext('2d');

	el.querySelector('.clear-btn').onclick = function () {
		paths.length = 0;
		path = null;
		draw();
	}

	el.querySelector('.predict-btn').onclick = function () {
		setUserInput(canvas);
	}

	const paths = [];

	let path;
	canvas.onmousedown = function (event) {
		if (event.button === 0 && !path) {
			path = [getPointer(event)];
			paths.push(path);
			draw();
		}
	}
	window.addEventListener('mousemove', event => {
		if (path) {
			path.push(getPointer(event));
			draw();
		}
	});
	window.addEventListener('mouseup', event => {
		if (event.button === 0) {
			path = null;
		}
	});

	function draw() {
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		ctx.save();
		ctx.scale2(window.devicePixelRatio);

		ctx.filter = 'blur(2px)';
		
		ctx.beginPath();
		for (const path of paths) {
			ctx.moveTo(...path[0]);
			ctx.lineTo(...path[0]);
			for (let i = 1; i < path.length; i++) {
				ctx.lineTo(...path[i])
			}
		}

		ctx.lineWidth = 15;
		ctx.strokeStyle = '#fff';
		ctx.lineCap = ctx.lineJoin = 'round';
		ctx.stroke();

		ctx.restore();
	}

	function getPointer(event) {
		const box = canvas.getBoundingClientRect();
		return [
			(event.clientX - box.x) / box.width * size, 
			(event.clientY - box.y) / box.height * size
		];
	}

	return el;
}