import tf from "@tensorflow/tfjs";
import * as processor from "./filters.js";

export const runModel = async (model, jsonData) => {
  let modelArray = [];
  const n = 150;
  const len = jsonData["imp1"].length;
  console.log("len", len);
  if (len < n) {
    return new Array(jsonData["imp1"].length).fill(1);
  }
  let signal;
  let replace = false;
  for (let i = 0; i < 4; i++) {
    if (n * (i + 1) < len) {
      signal = tf.tensor([
        jsonData["imp1"].slice(i * n, n * (i + 1)),
        jsonData["imp2"].slice(i * n, n * (i + 1)),
      ]);
      //console.log("imp1", jsonData["imp1"].slice(i * n, n * (i + 1)));
      //console.log("imp2", jsonData["imp2"].slice(i * n, n * (i + 1)));
    } else {
      signal = tf.tensor([
        jsonData["imp1"].slice(len - n, len),
        jsonData["imp2"].slice(len - n, len),
      ]);
      //console.log("imp1", jsonData["imp1"].slice(len - n, len));
      //console.log("imp2", jsonData["imp2"].slice(len - n, len));
      replace = true;
    }
    const filtered_signal = await filter_signal(signal);
    const predicted_delay = await estimate_time_delay(model, filtered_signal);
    const delay = await post_process(
      predicted_delay,
      new Array(predicted_delay.dataSync().length).fill(1) //optional or just put a 0 zero
    );
    //console.log("predicted_delay", predicted_delay.dataSync());
    //console.log("delay", delay.dataSync());
    if (!replace) {
      delay.dataSync().map((x) => modelArray.push(x));
    } else {
      delay
        .dataSync()
        .slice(
          delay.dataSync().length - (len - modelArray.length),
          delay.dataSync().length
        )
        .map((x) => modelArray.push(x));
    }
  }

  return modelArray;
};

export function fillDetection(channel1, channel2, runtime_mode, delay, idx) {
  let buffer = 10;
  let fill_1 = get_fill_index(channel1);
  let fill_2 = get_fill_index(channel2);

  if (fill_1 > 0 && fill_2 > 0) {
    runtime_mode = 0;
    idx = fill_1 + buffer;
    delay = fill_2 - fill_1;
  } else {
    idx = channel2.dataSync().length;
    delay = 0;
  }

  return {
    fill_1: fill_1,
    fill_2: fill_2,
    runtime_mode: runtime_mode,
    current_index: idx,
    current_delay: delay,
  };
}

function get_fill_index(signal) {
  let upper_threshold = 113880;
  let lower_threshold = 5000;

  //let diff = signal.slice(1, -1).sub(signal.slice(0, signal.shape[0] - 1));

  let detecting = true;
  while (detecting) {
    let diff = signal.slice(1, -1).sub(signal.slice(0, signal.shape[0] - 1));
    var candidate = tf
      .argMax(
        tf.logicalAnd(
          tf.less(signal, upper_threshold),
          tf.greater(signal, lower_threshold)
        )
      )
      .dataSync()[0];
    
    // Edge Removal is intentionally turned off until further testing (22/03/31)
    // Do edge removal if it is a real candidate
    if (candidate > 0) {
      signal, (detecting = removeRisingEdges(signal, diff, candidate));

      // If not a real candidate stop detecting
    } else {
      detecting = false;
    }
  }

  var candidate = tf
    .argMax(
      tf.logicalAnd(
        tf.less(signal, upper_threshold),
        tf.greater(signal, lower_threshold)
      )
    )
    .dataSync();

  return candidate[0];
}

export function filter_signal(signal) {
  let median_filter_size = 15;
  let signal1 = signal.slice([0, 0], [1, -1]);
  let signal2 = signal.slice([1, 0], [-1, -1]);
  signal1 = processor.median_filter(signal1, median_filter_size);
  signal2 = processor.median_filter(signal2, median_filter_size);
  signal1 = processor.normalize(signal1);
  signal2 = processor.normalize(signal2);
  return signal1.concat(signal2).squeeze();
}

export function estimate_time_delay(model, signal) {
  let output;
  output = model.predict(signal.transpose([1, 0]).expandDims(0));
  // console.dir(Array.from(output.round().dataSync()), {'maxArrayLength': null})
  return tf.squeeze(output);
}

export function post_process(time_delay, current_delay) {
  let min_delay = 40;

  // Check for and replace NaNs with 0
  time_delay = tf.mul(time_delay, tf.logicalNot(time_delay.isNaN()));

  // Enforce min delay time
  let mask = tf.less(tf.add(current_delay, time_delay), min_delay); // find where array is less than min
  let min_masked_time_delay = tf.mul(time_delay, tf.logicalNot(mask)); // everything below min is now 0
  time_delay = tf.add(min_masked_time_delay, tf.mul(mask, min_delay)).round();

  // Get last valid time delay
  let range_vec = tf.linspace(time_delay.shape[0] - 1, 0, time_delay.shape[0]);
  let out_bounds_idx = tf.argMax(
    tf.less(tf.sub(range_vec, tf.abs(time_delay)), 0)
  );

  out_bounds_idx = out_bounds_idx.dataSync()[0];

  if (out_bounds_idx == 0) {
    out_bounds_idx = time_delay.shape;
  }

  time_delay = tf.cast(time_delay.slice(0, out_bounds_idx).round(), "int32");

  return time_delay;
}

//batch version
export const runLegacyModel = async (
  model,
  jsonData,
  runtime_mode,
  idx,
  delay,
  n
) => {
  if (runtime_mode === -1) {
    let channel1 = tf.tensor(jsonData["imp1"]);
    let channel2 = tf.tensor(jsonData["imp2"]);
    const results = await fillDetection(
      channel1,
      channel2,
      runtime_mode,
      delay,
      idx
    );
    return results;
  } else if (runtime_mode === 0) {
    let signal = tf.tensor([jsonData["imp1"], jsonData["imp2"]]);
    signal = signal.slice([0, 0], [1, n]).concat(signal.slice([1, 0], [-1, n]));
    const filtered_signal = await filter_signal(signal);
    const time_delay = await estimate_time_delay(model, filtered_signal);
    const data = await post_process(time_delay, delay).dataSync();
    return {
      estimated_time_delay: data.map((x) => x + delay),
      current_index: idx + data.length,
      current_delay: delay + data.slice(-1)[0],
    };
  }
};

function removeRisingEdges(signal, diff, candidate) {
  let extension = 50;
  let rising_edge_threshold = 10000;

  // Chceck to see if there is a rising edge
  let rising_tensor = tf.argMax(tf.greater(diff, rising_edge_threshold));
  let rising_index = rising_tensor.dataSync()[0];

  // If the signal has a rising edge, slice past the rising edge and run another loop iteration
  if (rising_tensor.cast("bool").any().dataSync()[0]) {
    console.log("Rising edges");
    return signal.slice(rising_index + 1, -1), rising_index, true;

    // If the signal does not have a rising edge, candidate is good, exit detection
  } else {
    console.log("No rising edges");
    return signal, false;
  }
}
