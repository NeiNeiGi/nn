let inputLayerSize = 28 * 28;
let hiddenLayerSize = 4 * 4;
let outputLayerSize = 10;

let learningRate = 0.5;

class Model {
	constructor() {
		this.w1 = createParams(hiddenLayerSize * inputLayerSize);
		this.b1 = createParams(hiddenLayerSize * 1);
		this.w2 = createParams(outputLayerSize * hiddenLayerSize);
		this.b2 = createParams(outputLayerSize * 1);
	}

	forward(x) {
		const numSamples = x.length / inputLayerSize;

		this.x = x;
		this.z1 = new Float32Array(numSamples * hiddenLayerSize);
		this.a1 = new Float32Array(numSamples * hiddenLayerSize);

		// z1 = w1*x + b1
		// a1 = relu(z1)

		for (let i = 0; i < numSamples; i++) {
			for (let j = 0; j < hiddenLayerSize; j++) {
				const ni = i * hiddenLayerSize + j;
				this.z1[ni] = this.b1[j];
				for (let k = 0; k < inputLayerSize; k++) {
					this.z1[ni] += this.w1[j * inputLayerSize + k] * x[i * inputLayerSize + k];
				}

				this.a1[ni] = relu(this.z1[ni]);
			}
		}

		// z2 = w2*z1 + b2
		// a2 = softmax(z2)

		this.z2 = new Float32Array(numSamples * outputLayerSize);
		this.a2 = new Float32Array(numSamples * outputLayerSize);

		for (let i = 0; i < numSamples; i++) {
			let max = -Infinity;

			for (let j = 0; j < outputLayerSize; j++) {
				const ni = i * outputLayerSize + j;
				let z = this.b2[j];
				for (let k = 0; k < hiddenLayerSize; k++) {
					z += this.w2[j * hiddenLayerSize + k] * this.z1[i * hiddenLayerSize + k];
				}
				this.z2[ni] = z;
				z > max && (max = z);
			}

			// softmax XD

			let sum = 0;
			for (let j = 0; j < outputLayerSize; j++) {
				const ni = i * outputLayerSize + j;
				const e = Math.exp(this.z2[ni] - max);
				this.a2[ni] = e;
				sum += e;
			}

			sum = 1 / sum;
			for (let j = 0; j < outputLayerSize; j++) {
				this.a2[i * outputLayerSize + j] *= sum;
			}
		}

		return this.a2;
	}

	backward(y) {
		const numSamples = y.length / outputLayerSize;

		// dZ2 = (A2 - Y)/n

		const dZ2 = new Float32Array(y.length);
		for (let i = 0; i < y.length; i++) {
			dZ2[i] = (this.a2[i] - y[i]) / numSamples;
		}

		const dW2 = new Float32Array(this.w2.length);

		// dW2 = dZ2 @ A1
		// dZ2 @ A1 -> 10xN @ 16xN -> 10xN @ Nx16 -> 10x16 -> dZ2 @ transpose(A1)

		for (let i = 0; i < outputLayerSize; i++) {
			for (let j = 0; j < hiddenLayerSize; j++) {
				const ni = i * hiddenLayerSize + j;
				for (let k = 0; k < numSamples; k++) {
					dW2[ni] += dZ2[i + k * outputLayerSize] * this.a1[j + k * hiddenLayerSize];
				}
			}
		}

		// dB2 = sum dZ2 for each neuron across all samples

		const dB2 = new Float32Array(this.b2.length);

		for (let i = 0; i < outputLayerSize; i++) {
			for (let j = 0; j < numSamples; j++) {
				dB2[i] += dZ2[j * outputLayerSize + i];
			}
		}

		// dZ1 = W2 @ dZ2 * reluPrime(Z1)
		// W2 @ dZ2 -> 10x16 @ 10xN -> 16x10 @ 10xN -> 16xN -> transpose(W2) @ dZ2

		const dZ1 = new Float32Array(numSamples * hiddenLayerSize);

		for (let i = 0; i < hiddenLayerSize; i++) {
			for (let j = 0; j < numSamples; j++) {
				const ni = j * hiddenLayerSize + i;
				for (let k = 0; k < outputLayerSize; k++) {
					dZ1[ni] += this.w2[k * hiddenLayerSize + i] * dZ2[j * outputLayerSize + k];
				}

				dZ1[ni] *= reluPrime(this.z1[ni]);
			}
		}

		// dW1 = dZ1 @ X
		// dZ1 @ X -> 16xN @ 784xN -> 16xN @ Nx784 -> 16x784

		const dW1 = new Float32Array(this.w1.length);

		for (let i = 0; i < hiddenLayerSize; i++) {
			for (let j = 0; j < inputLayerSize; j++) {
				const ni = i * inputLayerSize + j;
				for (let k = 0; k < numSamples; k++) {
					dW1[ni] += dZ1[k * hiddenLayerSize + i] * this.x[k * inputLayerSize + j];
				}
			}
		}

		// dB1 = sum dZ1 for each neuron across all samples

		const dB1 = new Float32Array(this.b1.length);

		for (let i = 0; i < hiddenLayerSize; i++) {
			for (let j = 0; j < numSamples; j++) {
				dB1[i] += dZ1[j * hiddenLayerSize + i];
			}
		}

		// update params

		updateParams(this.w1, dW1);
		updateParams(this.b1, dB1);
		updateParams(this.w2, dW2);
		updateParams(this.b2, dB2);
	}
}

function createParams(n) {
	return Float32Array.from({ length: n }, () => Math.random() - 0.5);
}

function updateParams(params, grad) {
	for (let i = 0; i < params.length; i++) {
		let g = grad[i];
		g < -1 && (g = -1);
		g > 1 && (g = 1);
		params[i] -= g * learningRate;
	}
}

function relu(x) {
	return x > 0 ? x : 0;
}

function reluPrime(x) {
	return x > 0 ? 1 : 0;
}

function crossEntropy(targets, predictions) {
	const n = targets.length / outputLayerSize;
	
	let sum = 0;
	for (let i = 0; i < targets.length; i++) {
		const p = predictions[i];
		if (isFinite(p)) {
			sum += targets[i] * -Math.log(p + 1e-12);
		}
	}

	return sum / n;
}

// dataset

const xhr = new XMLHttpRequest();

xhr.onprogress = function (event) {
	const total = event.total || 69e6;
	const percent = Math.min(1, event.loaded / total);

	postMessage({
		id: 'progress', 
		percent
	});
}

xhr.onload = function () {
	data = [];

	const text = this.responseText;

	const lines = text.split('\n');
	lines.shift();

	for (let line of lines) {
		line = line.trim();
		if (!line) continue;

		const items = line.split(',');
		const label = parseInt(items.shift());
		for (let i = 0; i < items.length; i++) {
			items[i] = parseInt(items[i]) / 255;
		}

		data.push({
			x: new Float32Array(items), 
			y: label
		});
	}

	data.sort(() => Math.random() - 0.5);

	console.log(`dataset loaded! (${data.length} samples)`);

	postMessage({
		id: 'loaded'
	});
}

xhr.onerror = function () {
	postMessage({
		id: 'failed'
	});
}

xhr.open('GET', 'mnist_train.csv');
xhr.send();

let model, modelId;
let data, datasets;

function createDatasets(dataSplit, trainSplit) {
	const partialData = data.slice(0, Math.floor(dataSplit * data.length));

	const n = Math.floor(trainSplit * partialData.length);
	const trainData = partialData.slice(0, n);
	const valData = partialData.slice(n);

	const train = prepareData(trainData);
	const val = prepareData(valData);

	return { train, val };
}

function prepareData(data) {
	const x = new Float32Array(data.length * inputLayerSize);
	const y = new Uint8Array(data.length * outputLayerSize);

	const counter = {};

	for (let i = 0; i < data.length; i++) {
		const item = data[i];
		x.set(item.x, i * inputLayerSize);
		y[i * outputLayerSize + item.y] = 1;
	
		counter[item.y] = (counter[item.y] || 0) + 1;
	}

	let text = `${data.length} total samples:\n`;

	for (const key in counter) {
		const n = counter[key];
		const percent = n / data.length * 100;
		text += `${key} / ${n} / ${percent.toFixed(2)}%\n`;
	}

	console.log(text);

	return [x, y];
}

function train() {
	const startTime = performance.now();

	const [trainX, trainY] = datasets.train;
	const [valX, valY] = datasets.val;

	const trainPreds = model.forward(trainX);
	model.backward(trainY);

	const trainLoss = crossEntropy(trainY, trainPreds);
	const trainAccuracy = getAccuracy(trainY, trainPreds);

	const valPreds = model.forward(valX);
	const valLoss = crossEntropy(valY, valPreds);
	const valAccuracy = getAccuracy(valY, valPreds);

	const timeTaken = performance.now() - startTime;

	postMessage({
		id: 'epoch', 
		modelId, 
		trainLoss, 
		trainAccuracy, 
		valLoss, 
		valAccuracy, 
		timeTaken
	});

	sendParams();
}

function sendParams() {
	postMessage({
		id: 'params', 
		modelId, 
		w1: model.w1, 
		b1: model.b1, 
		w2: model.w2, 
		b2: model.b2
	});
}

function getAccuracy(targets, predictions) {
	let correct = 0;

	for (let i = 0; i < targets.length; i += outputLayerSize) {
		let max = -Infinity;
		let maxIndex = 0;
		for (let j = 0; j < outputLayerSize; j++) {
			const prob = predictions[i + j];
			if (prob > max) {
				max = prob;
				maxIndex = j;
			}
		}

		if (targets[i + maxIndex] === 1) {
			correct++;
		}
	}

	const n = targets.length / outputLayerSize;
	return correct / n;
}

function predict(x) {
	model.forward(x);

	postMessage({
		id: 'prediction', 
		modelId, 
		x, 
		a1: model.a1, 
		a2: model.a2
	});
}

const paramKeys = {
	w1: 1, 
	b1: 1, 
	w2: 1, 
	b2: 1
};

function getLossCurve(x) {
	const startParams = model.params || (model.params = {});

	const oldParams = {};

	for (const key in paramKeys) {
		const params = model[key];
		oldParams[key] = params;

		const start = startParams[key] || (startParams[key] = createParams(params.length));

		const newParams = new Float32Array(params.length);
		for (let j = 0; j < params.length; j++) {
			newParams[j] = start[j] * x + (1 - x) * params[j];
		}

		model[key] = newParams;
	}

	const preds = model.forward(datasets.val[0]);
	const loss = crossEntropy(datasets.val[1], preds);

	for (const key in oldParams) {
		model[key] = oldParams[key];
	}

	postMessage({
		id: 'lossCurve',
		modelId, 
		value: loss
	});
}

function getlossLandscape(x, y) {
	const dirX = model.dirX || (model.dirX = {});
	const dirY = model.dirY || (model.dirY = {});

	const oldParams = {};

	for (const key in paramKeys) {
		const params = model[key];
		oldParams[key] = params;

		const dx = dirX[key] || (dirX[key] = createParams(params.length));
		const dy = dirY[key] || (dirY[key] = createParams(params.length));

		const newParams = new Float32Array(params.length);
		for (let i = 0; i < params.length; i++) {
			newParams[i] = params[i] + dx[i] * x + dy[i] * y;
		}

		model[key] = newParams;
	}

	const preds = model.forward(datasets.val[0]);
	const loss = crossEntropy(datasets.val[1], preds);

	for (const key in oldParams) {
		model[key] = oldParams[key];
	}

	postMessage({
		id: 'lossLandscape',
		modelId, 
		value: loss
	});
}

addEventListener('message', event => {
	const msg = event.data;

	switch (msg.id) {
		case 'createDataset':
			datasets = createDatasets(msg.dataSplit, msg.trainSplit);
			break;

		case 'setLearningRate':
			learningRate = msg.value;
			break;

		case 'createModel':
			inputLayerSize = msg.inputLayerSize;
			hiddenLayerSize = msg.hiddenLayerSize;
			outputLayerSize = msg.outputLayerSize;
			modelId = msg.modelId;
			model = new Model();

			if (msg.params) {
				for (const key in msg.params) {
					model[key] = msg.params[key];
				}
			}
			
			sendParams();
			break;

		case 'train':
			train();
			break;

		case 'predict':
			const x = msg.x || data[Math.floor(Math.random() * data.length)].x;
			predict(x);
			break;

		case 'lossCurve':
			getLossCurve(msg.x);
			break;

		case 'lossLandscape':
			getlossLandscape(msg.x, msg.y);
			break;

		default:
			console.log(`Unknown message from parent: ${msg.id}`);
	}
});