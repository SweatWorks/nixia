import { impedance_reference, osmo_reference } from "./constants.js";
import * as processor from "./filters.js";

export function getBSA(height, weight) {
  const capturedHeight =
    height.unit === "cm" ? height.measure : height.measure * 2.54;
  const capturedWeight =
    weight.unit === "kg" ? height.measure : height.measure * 0.4535923699954641; // just apply to lbs
  const bsa =
    Math.pow(capturedWeight, 0.425) *
    Math.pow(capturedHeight, 0.725) *
    //0.007184; // my change
    71.84; //current formula
  return bsa;
}

export function correct_temp(imp, temp) {
  const a = -0.021012382970746594;
  imp = imp.map((x, i) => x / Math.exp(a * temp[i]));
  //imp = imp.map((x, i) => x / Math.exp(a * (temp[i] - 273.15)));
  return imp;
}

function binarySearch(arr, x, start, end) {
  let mid = Math.floor((start + end) / 2);
  if (start > end) return mid;
  if (arr[mid] === x) return mid;
  if (arr[mid] < x) return binarySearch(arr, x, start, mid - 1);
  else return binarySearch(arr, x, mid + 1, end);
}

export function convert_osmo(impedance) {
  let osmos = [];
  for (let i = 0; i < impedance.length; i++) {
    /*osmos.push(
      osmo_reference[
        binarySearch(
          impedance_reference,
          impedance[i],
          0,
          impedance_reference.length - 1
        )
      ]
    );*/ // my change
    osmos.push(
      binarySearch(
        impedance_reference,
        impedance[i],
        0,
        impedance_reference.length - 1
      )
    ); //current formula
  }
  return osmos;
}

//tensors operations

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
  var output;
  // console.log(signal.slice([0,0],[1,10]).dataSync())
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

  // Slice set_time_delay
  time_delay = tf.cast(time_delay.slice(0, out_bounds_idx).round(), "int32");

  return time_delay;
}
