#!/usr/bin/env python3
# Copyright (c) 2019, ARM Limited and Contributors
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import csv
import json
import os
from random import random
import subprocess
import sys
import yaml

# Default values
MODEL_FOLDER = "/data/local/tmp/models/"
BENCH_BIN_PATH = "/data/local/tmp/binaries/"
TFLITE_BIN = "benchmark_model"
ARMNN_BIN = "ExecuteNetwork"
NUM_LOOPS = 10
NUM_THREADS = [4]
WHERE_EXEC = ['cpu']

def execute_command(comm, shell=False):
    if shell:
        comm = ["adb", "shell"] + comm
    process = subprocess.run(
        comm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    return process.returncode, process.stdout, process.stderr

# Uploads the file only if it does not exist
def upload_if(localP, remoteP, executable=False):
    r, o, e = execute_command(["ls", remoteP], shell=True)
    if r==0:
        return 0 #because the file exist
    r, o, e = execute_command(["adb", "push", localP, remoteP])
    if r!=0:
        print("Error: Uploading file " + localP)
        return 1
    if not executable:
        return 0
    r, o, e = execute_command(["chmod", "u+x", remoteP], shell=True)
    if r!=0:
        print("Error: Making it executable")
        return 1
    return 0

def upload_executable():
    print("Uploading TFLite executable...", end='', flush=True)
    r, o, e = execute_command(["mkdir", "-p", BENCH_BIN_PATH], shell=True)
    if r!=0:
        print("Error: There was a problem with mkdir binaries folder")
        print(o, e)
        return 1
    dirs = os.listdir("binaries")
    for fil in dirs:
        ret = upload_if(
            os.path.join("binaries", fil), BENCH_BIN_PATH+fil, executable=True)
        if ret!=0:
            print("Error: There was a problem uploading "+fil)
            return 1
    r, o, e = execute_command(["mkdir", "-p", MODEL_FOLDER], shell=True)
    if r!=0:
        print("Error: There was a problem with mkdir model folder")
        return 1
    print("Done")
    return 0

# Execute the TFLite benchmark
def bench_exec(file, where, thr, loops):
    opt = ""
    if where == "gpu":
        opt = "--use_gpu=true"
    elif where == "nnapi":
        opt = "--use_nnapi=true"

    r, o, e = execute_command([BENCH_BIN_PATH + TFLITE_BIN,
                              "--graph=" + MODEL_FOLDER + file,
                              "--num_runs=" + str(loops),
                              "--num_threads=" + str(thr),
                              "--enable_op_profiling=true",
                              opt], shell=True)
    if r!=0:
        print("Error executing " + file + " on " +  where + " mode")
        print(o)
        print(e)
        return None, None
    return o, e

# Parser for the TFLite output
def parse(doc):
    doc = doc.split('\n')
    in_avg = False
    fir = False
    l_s = 0
    l_e = 0
    for i in range(len(doc)):
        line = doc[i]
        if len(line) == 0:
            continue
        if "Average" == line[0:7]:
            l_s = i + 2
            in_avg = True
        if line[0] == "=" and in_avg and fir:
            l_e = i
            in_avg = False
        if line[0] == "=" and in_avg and not fir:
            fir = True
    table = doc[l_s + 1:l_e]
    if len(table)==0:
        print("\nError extracting results table")
        return [], None
    x = csv.reader(table, delimiter="\t")
    li = []
    mean_time = 0.0
    for row in x:
        if len(row) < 4:
            continue
        row = [x for x in row if x]
        t = float(row[3].strip())
        mean_time = mean_time + t
        li.append({"layer":row[0].strip(), "time":t})
    return li, mean_time

# Parser for the ArmNN output
def parse_results(inp):
    st = ""
    lines = inp.split('\n')
    is_json = 0
    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == '{':
            is_json = is_json + 1
        if is_json != 0:
            st = st + line
        if line[0] == '}':
            is_json = is_json - 1
    if st == "":
        return None
    data = json.loads(st)
    data = data["ArmNN"]
    outd = {}
    k = list(data.keys())
    if len(k) > 1:
        print("There is more than 1 run")
    data = data[k[0]]
    if not "Execute_#2" in data:
        print("Execution not found")
        return None
    data = data["Execute_#2"]
    outd["type"] = "ArmNN"
    outd["mean_time"] = data["Wall clock time_#2"]["raw"][0] / 1000.0
    outd["times"] = []

    keys = list(data.keys())
    for k in keys[4:]:
        ent = data[k]
        ks = list(ent.keys())
        outd["times"].append(
            {"layer": k, "time": ent[ks[1]]["raw"][0] / 1000.0})

    return outd

def execute_tflite(work_config, model):
    TH = NUM_THREADS
    WH = WHERE_EXEC
    LP = NUM_LOOPS
    out = []
    model = os.path.split(model)[-1]
    if 'threads' in work_config.keys():
        TH = work_config['threads']
    if 'options' in work_config.keys():
        WH = work_config['options']
    if 'loops' in work_config.keys():
        LP = work_config['loops']
    for T in TH:
        for W in WH:
            ret, err = bench_exec(model, W, T, LP)
            if ret is not None:
                ret = ret.decode('ASCII')
                li, m_t = parse(ret)
                if len(li) == 0:
                    print("Error")
                    continue
                out.append(
                    {'type':W + "_" + str(T) + "Threads", "mean_time": m_t,
                    "times": li, "threads": T})
    return out

def execute_armnn(conf, model):
    LP = 1
    shap = []
    inName = ""
    outName = ""
    concurrent = " "
    quant = " "
    turbo = " "
    acc = " "
    model = os.path.split(model)[-1]
    # Handle options
    if 'input_shape' not in conf.keys():
        print("Error: Input shape required on ArmNN")
        return None
    if 'input_name' not in conf.keys() or 'output_name' not in conf.keys():
        print("Error: Input and Output names required")
        return None
    shap = conf['input_shape']
    inName = conf['input_name']
    outName = conf['output_name']
    if 'concurrent' in conf.keys():
        if conf['concurrent']:
            concurrent = "-n"
    if 'quantized' in conf.keys():
        if conf['quantized']:
            quant = "-q"
    if 'fp16' in conf.keys():
        if conf['fp16']:
            turbo = "-h"
    if 'loops' in conf.keys():
        LP = conf['loops']
    if 'accelerator' in conf.keys():
        if 'Gpu' == conf['accelerator']:
            acc = "-c GpuAcc"
        if 'Cpu' == conf['accelerator']:
            acc = "-c CpuAcc"
    with open("temp_input_file","w") as f:
        t = 1
        for s in shap:
            t = t * s
        for i in range(t):
            f.write(str(random()) + "\n")
    upload_if("temp_input_file", MODEL_FOLDER + "intemp")
    out = None
    for i in range(LP):
        r, o, e = execute_command(["export LD_LIBRARY_PATH=" + BENCH_BIN_PATH,
                                  "&&",
                                  BENCH_BIN_PATH + ARMNN_BIN,
                                  concurrent,
                                  quant,
                                  turbo,
                                  "-e",
                                  "-f tflite-binary",
                                  "-m " + MODEL_FOLDER + model,
                                  "-i " + inName,
                                  "-o " + outName,
                                  acc,
                                  "-c CpuRef",
                                  "-d " + MODEL_FOLDER + "intemp"],
                                  shell = True)
        if r != 0:
            print("Error: Executing armnn")
            print(o)
            print(e)
            break
        o = o.decode('ascii')
        o = parse_results(o)
        if o == None:
            print("Error: Parsing ArmNN")
            break
        if out == None:
            out = o
        else:
            out['mean_time'] = out['mean_time']+o['mean_time']
            for i in range(len(out['times'])):
                out['times'][i]['time'] = (out['times'][i]['time']
                                           + o['times'][i]['time'])
    if out != None:
        out['mean_time'] = out['mean_time'] / LP
        for i in range(len(out['times'])):
            out['times'][i]['time'] = out['times'][i]['time'] / LP
    r, o, e = execute_command(["rm",MODEL_FOLDER+"intemp"], shell = True)
    return out

def loop_workloads(data):
    if 'workloads' not in data.keys():
        print("Error: Parsing yaml")
        return 1
    r = upload_executable()
    if r != 0:
        return 1
    dic = {}
    for work in data['workloads']:
        if 'model' not in work.keys():
            print("Error: Model option not found")
            continue
        name = os.path.split(work['model'])[-1]
        if 'name' in work.keys():
            print("\tRunning workload " + work['name'])
        else:
            print("\tRunning workload " + name)
        upload_if(
            os.path.join(
                *list(os.path.split(work['model']))),
            MODEL_FOLDER + name)
        res = []

        if 'tflite' in work.keys():
            print("\t\tExecuting with tflite")
            out = execute_tflite(work['tflite'], work['model'])
            res.extend(out)
        
        if 'armnn' in work.keys():
            print("\t\tExecuting with armnn")
            out = execute_armnn(work['armnn'], work['model'])
            if out != None:
                res.append(out)

        if 'name' in work.keys():
            name = work['name']
        dic[name] = res
    r, o, e = execute_command(["rm", "-r", BENCH_BIN_PATH], shell = True)
    r, o, e = execute_command(["rm", "-r", MODEL_FOLDER], shell = True)
    return dic

def main(args):
    if len(args) < 2:
        print("Error: Argument error")
        print("Usage:\n./testbench.py [yaml file with the config] | -h")
        sys.exit(128)

    if args[1] == '-h':
        print("Usage:\n./testbench.py [yaml file with the config] | -h")
        return
    
    if not os.path.isfile(args[1]):
        print("Error: File not found")
        sys.exit(128)

    with open(args[1], 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        out = loop_workloads(data)
        outfile = "results.json"
        if 'global' in data.keys():
            if 'outputfile' in data['global'].keys():
                outfile = data['global']['outputfile']
        print("Writting output file...", end='', flush=True)
        with open(outfile, "w") as f:
            json.dump(out, f)
            print("Done")

if __name__ == "__main__":
    main(sys.argv)