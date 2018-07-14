// Daniel Shiffman
// Intelligence and Learning
// The Coding Train

// Full tutorial playlist:
// https://www.youtube.com/playlist?list=PLRqwX-V7Uu6bmMRCIoTi72aNWHo7epX4L

// Community version:
// https://codingtrain.github.io/ColorClassifer-TensorFlow.js
// https://github.com/CodingTrain/ColorClassifer-TensorFlow.js

let data;
let model;
let xs, ys;
let rSlider, gSlider, bSlider;
let labelP;
let lossP;
let canvas;
let graph;
let lossX = [];
let lossY = [];
let accY = [];

let labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]

function preload() {
  data = loadJSON('./colorData.json');
}

function setup() {
  // Crude interface
  canvas = createCanvas(200, 200);
  graph = document.getElementById('graph');
  labelP = select('#prediction');
  lossP = select('#loss');
  rSlider = select('#red-slider');
  gSlider = select('#green-slider');
  bSlider = select('#blue-slider');

  canvas.parent('rgb-Canvas');
  let colors = [];
  let labels = [];
  for (let record of data.entries) {
    let col = [record.r / 255, record.g / 255, record.b / 255];
    colors.push(col);
    labels.push(labelList.indexOf(record.label));
  }

  xs = tf.tensor2d(colors);
  let labelsTensor = tf.tensor1d(labels, 'int32');

  ys = tf.oneHot(labelsTensor, 9).cast('float32');
  labelsTensor.dispose();

  model = buildModel();

  train();
}

async function train() {
  // This is leaking https://github.com/tensorflow/tfjs/issues/457
  await model.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.1,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch);
        lossY.push(logs.val_loss.toFixed(2));
        accY.push(logs.val_acc.toFixed(2));
        lossX.push(epoch + 1);
        lossP.html('Loss: ' + logs.loss.toFixed(5));
      },
      onBatchEnd: async (batch, logs) => {
        await tf.nextFrame();
      },
      onTrainEnd: () => {
        console.log('finished')
      },
    },
  });
}

function buildModel() {
  let md = tf.sequential();
  const hidden = tf.layers.dense({
    units: 15,
    inputShape: [3],
    activation: 'sigmoid'
  });

  const output = tf.layers.dense({
    units: 9,
    activation: 'softmax'
  });
  md.add(hidden);
  md.add(output);

  const LEARNING_RATE = 0.25;
  const optimizer = tf.train.sgd(LEARNING_RATE);

  md.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return md
}

function plotTraining() {
  let layout = {
    width: 600,
    height: 300,
    title: 'Graph of learning progress',
    xaxis: {
      title: 'No. of Epochs'
    }
  };

  let loss = {
    x: lossX,
    y: lossY,
    name: 'Val Loss'
  };

  let acc = {
    x: lossX,
    y: accY,
    name: 'Val Accuracy'
  };

  Plotly.newPlot(graph, [loss, acc], layout);
}
function draw() {
  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();
  background(r, g, b);
  //strokeWeight(2);
  //stroke(255);
  //line(frameCount % width, 0, frameCount % width, height);
  tf.tidy(() => {
    const input = tf.tensor2d([
      [r, g, b]
    ]);
    let results = model.predict(input);
    let argMax = results.argMax(1);
    let index = argMax.dataSync()[0];
    let label = labelList[index];
    labelP.html("Color: " + label);
  });

  plotTraining();
}
