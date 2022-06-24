/* eslint-disable */

// import * as np from 'numpy';
import tf from "@tensorflow/tfjs";

function replace_infinite(signal, replace_with = 0) {
  /*
  Description:
  -----------
  Replace infinite artifact sensor values with replace_with

  Params:
  -------
  signal (np.array): intended for either osmolality or temperature
  replace_with (int): value to replace infinite values with

  Returns:
  --------
  signal (np.array): correct infinite values
  */
  signal[tf.isInf(signal)] = replace_with;
  return signal;
}

function clip_threshold(signal, threshold) {
  /*
  Description:
  -----------
  Clip artifact sensor values with above a threshold to the threshold

  Params:
  -------
  signal (np.array): intended for either osmolality or temperature
  threshold (int): value to apply a clip to

  Returns:
  --------
  signal (np.array): thresholded values
  */
  signal[signal > threshold] = threshold;
  return signal;
}

function pad_signal(t, sig) {
  /*
  Description:
  -----------
  Zero pad a signal to desired length

  Params:
  -------
  t (int): desired signal size
  sig (np.array): input signal

  Returns:
  --------
  padded_signal (np.array): zero padded signal
  */
  var pad_size, padded_sig;
  pad_size = t - sig.shape[0];
  padded_sig = tf.zeros(t);
  // padded_sig.slice(pad_size) = sig;
  return padded_sig;
}

export function normalize(signal) {
  let min = tf.min(signal);
  let max = tf.max(signal);
  return tf.div(tf.sub(signal, min), tf.sub(max, min));
}

export function moving_average_filter(signal, N) {
  /*
   * Apply a moving average filter of rank N to signal
   */
  let ones_filter = tf.ones([N]).expandDims(1).expandDims(2);
  ones_filter = ones_filter.mul(1 / N);

  signal = signal.expandDims(2);
  signal = tf.conv1d(signal, ones_filter, 1, "same");
  return signal;
}

export function median_filter(values, N) {
  // Check that filter size if odd
  if ((N - 1) % 2 != 0) throw "Median filter size must be odd.";

  // Create a padding array
  let pad_size = (N - 1) / 2;
  let padding = new Array(pad_size).fill(0);

  // Change from Tensor to Array and pad
  values = Array.from(values.dataSync());
  values = padding.concat(values).concat(padding);

  // Apply median filter
  let filtered_signal = [];
  for (var idx = pad_size; idx < values.length - pad_size; idx++) {
    filtered_signal.push(
      median(values.slice(idx - pad_size, idx + pad_size + 1))
    );
  }
  return tf.tensor(filtered_signal).expandDims(0).expandDims(2);
}

function median(values) {
  if (values.length === 0) return 0;

  values.sort(function (a, b) {
    return a - b;
  });

  var half = Math.floor(values.length / 2);

  if (values.length % 2) return values[half];

  return (values[half - 1] + values[half]) / 2.0;
}

export function trapz(array) {
  let n = array.length;
  let area = 0;
  for (let i = 1; i < n + 1; i++) {
    let tmp = (array[i] + array[i - 1]) / 2;
    area = area + tmp;
  }
  return area;
}
