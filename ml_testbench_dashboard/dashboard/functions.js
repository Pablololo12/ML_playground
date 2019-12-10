// Copyright (c) 2019, ARM Limited and Contributors
//
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

var app = require('electron').remote; 
var dialog = app.dialog;
var fs = require('fs');

var Edata = null;
var selModel = document.getElementById("listModels");
var selOption = document.getElementById("listOptions");
var selModel2 = document.getElementById("listModels2");
var selOption2 = document.getElementById("listOptions2");
var textTime = document.getElementById("text_time");
var textTime2 = document.getElementById("text_time2");

const buttOpen = document.getElementById("buttonOpen");
buttOpen.addEventListener('click', function () {
    console.log("Opening file");
    dialog.showOpenDialog((fileNames) => {
    // fileNames is an array that contains all the selected
    if(fileNames === undefined){
        console.log("No file selected");
        return;
    }

    fileName = fileNames[0]
    fs.readFile(fileName, 'utf-8', (err, data) => {
        if(err){
            alert("An error ocurred reading the file :" + err.message);
            return;
        }
        Edata = JSON.parse(data);
        loadData();
    });
    });
});

function loadData() {
    selModel.options.length = 0;
    selModel2.options.length = 0;
    var option = document.createElement("OPTION");
    option.innerHTML = "Empty";
    option.value = 0;
    selModel2.options.add(option);
    keys = Object.keys(Edata);
    keys.forEach(function(key, i, a) {
        option = document.createElement("OPTION");
        option.innerHTML = key;
        option.value = key;
        selModel.options.add(option);
        option = document.createElement("OPTION");
        option.innerHTML = key;
        option.value = key;
        selModel2.options.add(option);
    });
    populOption();
    plotData();
}

function populOption() {
    if (Edata == null) return;
    l = Edata[selModel.value];
    selOption.options.length = 0;
    if (l == null) return;
    console.log(l)
    l.forEach(function(k, i, a) {
        var option = document.createElement("OPTION");
        option.innerHTML = k["type"];
        option.value = i;
        selOption.options.add(option);
    })
}

function populOption2() {
    if (Edata == null) return;
    l = Edata[selModel2.value];
    selOption2.options.length = 0;
    if (l == null) return;
    console.log(l)
    l.forEach(function(k, i, a) {
        var option = document.createElement("OPTION");
        option.innerHTML = k["type"];
        option.value = i;
        selOption2.options.add(option);
    })
}

var helper_popul_plot = function () {
    populOption();
    plotData();
}
var helper_popul_plot2 = function () {
    populOption2();
    plotData();
}
var helper_plot = function () {
    plotData();
}
selModel.addEventListener('change', helper_popul_plot);
selOption.addEventListener('change', helper_plot);
selModel2.addEventListener('change', helper_popul_plot2);
selOption2.addEventListener('change', helper_plot);

Number.prototype.pad = function(size) {
    var s = String(this);
    while (s.length < (size || 2)) {s = "0" + s;}
    return s;
}

function plotData() {
    tpl = document.getElementById('timperlayer');
    model = selModel.value;
    type = selOption.value;
    tavg = Edata[model][type]["mean_time"];
    textTime.innerText = "Mean execution time: "+tavg+"ms";
    Plotly.purge(tpl);
    timperlayer(model, type);
    timlayer(model, type);

    model = selModel2.value;
    type = selOption2.value;
    if (Edata[model] != null) {
        tavg = Edata[model][type]["mean_time"];
        textTime2.innerText = "Mean execution time: "+tavg+"ms";
        timperlayer(model, type);
    } else {
        textTime2.innerText = "";
    }
}

function timlayer (model, type) {
    tpl = document.getElementById('timlayer');
    tpl2 = document.getElementById('timlayer2');
    Plotly.purge(tpl);
    Plotly.purge(tpl2);
    console.log("Model: "+model+" Type: "+type);
    times = Edata[model][type]["times"];
    ls = {};
    times.forEach(function (k,i,o) {
        l=k["layer"]; t=k["time"];
        if (l in ls) {
            ls[l] = ls[l]+t;
        } else {
            ls[l] = t;
        }
    })
    x = []; y = []
    k = Object.keys(ls);
    k.forEach(function (q, i, d) {
        x.push(q);
        y.push(ls[q]);
    });
    Plotly.plot(tpl, [{
        x: x, y:y, type: 'bar'
    }], {
        title: {
            text: 'Time Summary Per Layer'
        },
        xaxis: {
            title: {
                text: 'Layers'
            }
        },
        yaxis: {
            title: {
                text: 'Time (ms)'
            }
        }
    });
    x = []; y = [];
    l = Edata[selModel.value];
    l.forEach(function(k,i,o) {
        x.push(k["type"]);
        y.push(k["mean_time"]);
    })
    Plotly.plot(tpl2, [{
        x: x, y:y, type: 'bar'
    }], {
        title: {
            text: 'Comparison between execution places'
        },
        xaxis: {
            title: {
                text: 'Type of execution'
            }
        },
        yaxis: {
            title: {
                text: 'Time (ms)'
            }
        }
    });
}

function timperlayer (model, type) {
    tpl = document.getElementById('timperlayer');
    console.log("Model: "+model+" Type: "+type);
    times = Edata[model][type]["times"];
    x = []; y = []
    times.forEach(function (k, i, d) {
        x.push(k["layer"]+"_"+(i).pad());
        y.push(k["time"])
    });
    Plotly.plot(tpl, [{
        x: x, y:y, type: 'bar', name: model+" "+Edata[model][type]["type"]
    }], {
        margin: {t: 50, b: -50},
        title: {
            text: 'Time Per Layer'
        },
        xaxis: {
            title: {
                text: 'Layers'
            }
        },
        yaxis: {
            title: {
                text: 'Time (ms)'
            }
        }
    });
}
