import { correct_temp, convert_osmo, getBSA } from "./formulas.js";
import { runModel, fillDetection, runLegacyModel } from "./model.js";
import workoutObject from "../../workouts/input.json";
//import workoutObject from "../workout1.json";
import * as fs from "fs";
import tf from "@tensorflow/tfjs-node";

function saveJson(file, array) {
  fs.writeFile(file, JSON.stringify(array), function (err) {
    if (err) {
      console.error("Crap happens");
    }
  });
}

function estimateSweatRate(td, imp_signal, temp, bsa) {
  imp_signal = correct_temp(imp_signal, temp);

  //console.log("imp_signal", imp_signal);

  let osmo_signal = convert_osmo(imp_signal);

  //console.log("osmo_signal", osmo_signal);

  //saveJson("./outputs/osmolality.json", osmo_signal);

  var cross_sectional_area = 0.1284; // cross sectional area of the channnel in mm^2
  var channel_length = 20.03; // length of the sweat channel in mm
  var fs = 1; // effective sampling rate of the osmo sensors (actuall 100Hz averaged to 2Hz)
  var r = 0.6; // radius of the sweat resorvoir (cm)
  var sweat_rate_bound = 5; // theoretical bound of human instantanous sweat rate

  var sweat_area = Math.PI * Math.pow(r, 2);

  var predicted_flow = td.map((x) => (channel_length / x) * fs);

  let clip_rate = false;

  var vfr = predicted_flow.map((x) => x * cross_sectional_area);

  vfr = vfr.map((x) => (x * 60) / sweat_area);

  // Clip the sweat rate to that of the human theoretical limit
  if (clip_rate) {
    vfr[vfr > sweat_rate_bound] = sweat_rate_bound;
  }

  var wbsr_ml_min = vfr.map((x) => (bsa * (x * 0.353 + 0.199)) / 1000);

  var wbsr_ml_sec = wbsr_ml_min.map((x) => x / 60);

  var na_loss = osmo_signal.map(
    (x, i) => ((x * wbsr_ml_sec[i]) / 1000) * 22.98 * 0.3125
  );

  var cl_loss = osmo_signal.map(
    (x, i) => ((x * wbsr_ml_sec[i]) / 1000) * 35.453 * 0.3125
  );
  var k_loss = osmo_signal.map(
    (x, i) => ((x * wbsr_ml_sec[i]) / 1000) * 39.0983 * 0.0312
  );
  var total_electrolyte_loss = na_loss.map(
    (x, i) => x + cl_loss[i] + k_loss[i]
  );

  const results = {
    wbsr_ml_sec: wbsr_ml_sec,
    electrolytes: total_electrolyte_loss,
  };
  return results;
}

const getSweatStats = (time_delay, imp_signal, temp, height, weight) => {
  const bsa = getBSA(height, weight);
  return estimateSweatRate(time_delay, imp_signal, temp, bsa);
};

const aiSweatProcess = (impedance1, skinTemp, ai_delay) => {
  const estimated_time_delay = new Array(impedance1.length).fill(ai_delay);
  const profile = {
    height: { unit: "cm", measure: "173" },
    weight: { unit: "kg", measure: "73" },
  };
  return getSweatStats(
    estimated_time_delay,
    impedance1,
    skinTemp,
    profile.height,
    profile.weight
  );
};

/*const estimated_time_delay = new Array(impedance1.length).fill(ai_delay);
let estimateElectrolytes = 0;
let estimateFluid = 0;
let estimatedLength = 1;*/

/**
 * 
 *const ai_n = 150;
let ai_index = 0;
let ai_mode = -1;
let ai_delay = 1;
let ai_processing = false;
let lastLength = 0;
let allParsedRecords = [];
 */

function test_stats() {
  //console.log("execute main: ", convert_osmo([0, 50000, 70000, 150000]));
  //should return [599.5,124,96,61]
  /*const stats = aiSweatProcess(
    [],
    [150000, 150000, 150000, 7558.7998046875, 150000, 150000],
    [0, 0, 8, 10, 12, 16],
    ai_delay
  );*/

  let ai_delay = 1;
  const ai_temp = Array.from(workoutObject, (record) => record.st);
  const ai_imp1 = Array.from(workoutObject, (record) => record.i1);
  const stats = aiSweatProcess(ai_imp1, ai_temp, ai_delay);

  console.log("stats", stats);

  saveJson("./outputs/electrolytes.json", stats);
}

const test_fill = async () => {
  const channel1 = Array.from(workoutObject, (record) => record.i1);
  const channel2 = Array.from(workoutObject, (record) => record.i2);
  const runtime_mode = 0;
  const delay = 0;
  const idx = 1;

  const result = await fillDetection(
    channel1,
    channel2,
    runtime_mode,
    delay,
    idx
  );

  console.log(result);
};

async function test_model() {
  const ai_imp1 = Array.from(workoutObject, (record) => record.i1);
  const ai_imp2 = Array.from(workoutObject, (record) => record.i2);
  const model = await tf.loadLayersModel("file://./predict/model.json");
  model.summary();
  const result = await runModel(model, { imp1: ai_imp1, imp2: ai_imp2 });
  console.log("result model", result);
  const profile = {
    height: { unit: "cm", measure: "173" },
    weight: { unit: "kg", measure: "73" },
  };
  const ai_temp = Array.from(workoutObject, (record) => record.st);
  const stats = getSweatStats(
    result,
    ai_imp1,
    ai_temp,
    profile.height,
    profile.weight
  );
  console.log("stats", stats);
  saveJson("./outputs/electrolytes.json", stats);
}

const ai_n = 150;
let ai_index = 0;
let ai_mode = -1;
let ai_delay = 0;
let lastLength = 0;

const main = async () => {
  //test_stats();
  //test_model();
  //const i1 = Array.from(workoutObject, (record) => record.i1);
  //const st = Array.from(workoutObject, (record) => record.st);
  //console.log("i1", i1);
  //console.log("st", st);
  //console.log("corrected", correct_temp(i1, st));
  //test_fill();
  let metaParsedRecords = JSON.parse(JSON.stringify(workoutObject));
  const model = await tf.loadLayersModel("file://./predict/model.json");
  model.summary();
  for (let i = 0; i < workoutObject.length; i++) {
    //console.log("workoutObject",workoutObject.slice(0,i).length)
    const data = metaParsedRecords.slice(0, i);
    const i1 = Array.from(data, (record) => record.i1);
    const i2 = Array.from(data, (record) => record.i2);
    const st = Array.from(data, (record) => record.st);
    let { processed: result } = await aiSweatProcess2(data, i1, i2, st, model);

    if (result.length > 0) {
      for (let j = 0; j < result.length; j++) {
        /*if (j === 186) {
          console.log("on loop", result[j], metaParsedRecords[j]);
        }*/
        metaParsedRecords[j] = {
          ...{ td: 0, el: 0, sr: 0 },
          ...metaParsedRecords[j],
          ...result[j],
        };
      }
      //lastLength = +result.length + 1;
    }
    //console.log("finalResult", finalResult);

    //console.log("result", result.length);

    await new Promise((r) => setTimeout(r, 250));
  }
  //console.log("finalResult", finalResult);
  saveJson("./outputs/output.json", metaParsedRecords);
};

main();

/** const result = runModel(
    model,
    { imp1: ai_imp1, imp2: ai_imp2 },
    0, // runtime_mode
    0, // index
    ai_delay,
    150 // n
  ); */

const aiSweatProcess2 = (metaParsedRecords, imp1, imp2, st, ai_model) => {
  let estimated_time_delay = new Array(imp1.length).fill(-1);
  let estimateElectrolytes = 0;
  let estimateFluid = 0;
  let estimatedLength = 1;
  return new Promise(async (resolve, reject) => {
    //if (lastLength - ai_index >= ai_n + ai_delay) {
    if (
      metaParsedRecords[+ai_index + ai_delay] &&
      metaParsedRecords[+ai_index + ai_n + ai_delay]
    ) {
      const { impedance1, impedance2, skinTemp } = get_next_data_batch(
        ai_index,
        ai_index + ai_n,
        ai_delay,
        ai_mode,
        imp1,
        imp2,
        st
      );

      const modelResult = await runLegacyModel(
        ai_model,
        { imp1: impedance1, imp2: impedance2 },
        ai_mode,
        ai_index,
        ai_delay,
        ai_n
      );
      console.log("modelResult", modelResult);
      if (
        modelResult.runtime_mode != null &&
        modelResult.runtime_mode != undefined
      ) {
        ai_mode = modelResult.runtime_mode;
      }
      if (modelResult.current_index) {
        ai_index = modelResult.current_index;
      }
      if (modelResult.current_delay) {
        ai_delay = modelResult.current_delay;
      }
      if (modelResult.estimated_time_delay) {
        estimated_time_delay = modelResult.estimated_time_delay;
      }
      const profile = {
        height: { unit: "cm", measure: "173" },
        weight: { unit: "kg", measure: "73" },
      };
      const resultSweat = getSweatStats(
        [...estimated_time_delay],
        impedance1,
        skinTemp,
        profile.height,
        profile.weight
      );
      console.log(
        "resultSweat.electrolytes",
        resultSweat.electrolytes.length,
        estimated_time_delay.length,
        resultSweat.electrolytes
      );
      resultSweat.electrolytes.forEach((electrolytes, i) => {
        if (i >= lastLength) {
          estimateElectrolytes += electrolytes;
          estimateFluid += resultSweat.wbsr_ml_sec[i];
          estimatedLength++;
        }

        /*console.log(
          electrolytes,
          resultSweat.wbsr_ml_sec[i],
          estimated_time_delay[i],
          ai_index,
          +i + ai_index
        );*/
        metaParsedRecords[+i + ai_index] = {
          ...metaParsedRecords[+i + ai_index],
          //metaParsedRecords[i] = {
          //  ...metaParsedRecords[i],
          el: electrolytes,
          sr: resultSweat.wbsr_ml_sec[i],
          td: estimated_time_delay[i],
        };
      });
      lastLength = resultSweat.electrolytes.length;
      estimateElectrolytes = estimateElectrolytes / estimatedLength;
      estimateFluid = estimateFluid / estimatedLength / 29.5735;
      resolve({ processed: metaParsedRecords });
    } else {
      resolve({ processed: [] });
    }
  });
};

/**
 * 
 * 
 * data = get_next_data_batch(s = res_data['current_index'],
                        e = res_data['current_index']+n,
                        current_index = res_data['current_index'],
                        current_delay = res_data['current_delay'],
                        runtime_mode = res_data['runtime_mode'],
                        estimated_time_delay = res_data['estimated_time_delay'])
 */

const get_next_data_batch = (
  s,
  e,
  current_delay = -1,
  runtime_mode = -1,
  impedance1,
  impedance2,
  skinTemp
) => {
  if (runtime_mode === -1) {
    impedance1 = impedance1.slice(s, e);
    impedance2 = impedance2.slice(s, e);
  }
  if (runtime_mode === 0) {
    impedance1 = impedance1.slice(s, e);
    impedance2 = impedance2.slice(+s + current_delay, +e + current_delay);
    skinTemp = skinTemp.slice(s, e);
    //     data['time_delay'] = data['time_delay'][s:e]
    // data['current_index'] = current_index
    //data['current_delay'] = current_delay
    //data['runtime_mode'] = runtime_mode
    //data['estimated_time_delay'] = estimated_time_delay
    //data['imp_signal'] = data['impedance1'][s:e]
  }
  return { impedance1, impedance2, skinTemp };
};
