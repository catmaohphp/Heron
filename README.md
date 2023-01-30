# Heron

## Introduction

Automatically generation of performant programs for deep learning accelerators (DLA) extremely difficult. Heron solve this problem by automatically generating a constrained search space and explore it with a novel constrained genetic algorithm(CGA). 

## Installation

**Build for Tensor Core code generation.** Make sure that your platform has GPUs with Tensor Core available. Tested platform includes V100, T4, and RTX8000, experiments on these platforms should not inccur any problem.

First, check the dependancies needed.

```
cublas v10.2
cudnn v7.6.5
```

Second, build TVM for gpu. You can follow [TVM build from source instructions](http://tmp.syfeng.net:9090/install/from_source.html#install-from-source) for details.

```shell
cd third_parties/tvm
mkdir build
cp cmake/config_tc.cmake build
cd build
cmake ..
make -j8

# You can add following configurations to you ~/.bashrc file
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```



**Build for  DL Boost  code generation.** Make sure that your platform has CPUs with DL Boost available. We conducted our experiments on Intel’s Xeon Gold 6240 CPU.

First, check the dependancies needed.

```
# other versions may have errors related to intrinsic functions.
llvm 8.0.0
```

Second, build TVM for cpu. You can follow [TVM build from source instructions](http://tmp.syfeng.net:9090/install/from_source.html#install-from-source) for details.

```
cd third_parties/tvm
mkdir build
cp cmake/config_cpu.cmake build
cd build
cmake ..
make -j8

# You can add following configurations to you ~/.bashrc file
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```



## Tuning

**A quick start for Tensor Core tuning.** Take gemm operation with (m,k,n) =(32, 4096, 4096) as an example, we show how to automatically schedule with Heron.

```shell
cd tests
python tests.py -p tensorcore -n gemm -c ./configs/quick_start.json
```

Then, you can get the following results in 1 or 2 minutes.

...

All configurations locates in ./configs folder. You can try your own configuration if needed. Take quick_start.json for example,  the comments show the functionality of each configuration.

...

To obtain results in our paper, you can type the following instruction. These will take you ... minutes to finish.

```shell
python tests.py -p tensorcore -n gemm -c ./configs/config_G4.json
```

If your platform is same with our's, you can try the tuned solutions by

```shell
python tests.py -p tensorcore -n gemm -c ./configs/config_G4.json -t ./tuned/tuned_G4.pkl
```

